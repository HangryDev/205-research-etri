"""
CCTV Action Recognition — 영상 프레임 추출 및 안전위반 탐지
=============================================================
데이터셋: CCTV Action Recognition Dataset (Kaggle, Jonathan Nield)
    - 13개 액션: fall, grab, gun, hit, kick, lying_down,
                 run, sit, sneak, stand, struggle, throw, walk
    - 소스: UCF Crime, NTU, YouTube CCTV
    - split 파일: filename.mp4 label (1 or 2)
    - 실제 mp4는 별도 다운로드 필요

참고 캐글 메달 노트북:
    1. "Action Recognition for Real-Time Safety Monitoring" (Kaggle Competition)
       — 실시간 CCTV 안전위반 탐지 (fall, fight, anomaly)
    2. "train-movinet-a1-on-cctv-dataset" (caturbudisantoso)
       — MoViNet-A1 영상 분류 모델 학습
    3. Jonathan Ledur-Nield 논문 "Human pose estimation and skeletal
       action recognition in CCTV" — CNN+LSTM 아키텍처

안전위반 분류:
    위반(VIOLATION): fall, grab, gun, hit, kick, sneak, struggle, throw
    정상(NORMAL):    walk, stand, sit, run, lying_down

파이프라인:
    1. Split 파일 파싱 & 데이터셋 구성
    2. 영상 → 프레임 추출 (OpenCV)
    3. 프레임 전처리 (리사이즈, 정규화)
    4. CNN 특징 추출 + LSTM 시계열 분류
    5. 안전위반 탐지 & 시각화
"""

import os
import sys
import glob
import logging
import warnings
import time
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Input, TimeDistributed, GlobalAveragePooling2D)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical

# ============================================================
# 로깅
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("lecture/h3_debug.log", mode="w"),
    ]
)
log = logging.getLogger("CCTV-Safety")

# ============================================================
# [설정] CCTV 안전위반 탐지 실험 하이퍼파라미터
# ============================================================
# [리뷰] IMG_SIZE=112: 프레임을 112x112으로 리사이즈 (연산량과 정보량의 균형)
#        SEQ_LENGTH=16: 16프레임을 하나의 시퀀스로 처리
#        SPLIT_RATIO="75%": 학습/테스트 비율 (75% train split 파일 사용)
SPLIT_DIR   = "lecture/h3_data/Test_Train_Splits"
VIDEO_DIR   = "lecture/h3_data/Videos"        # mp4 파일 위치
SPLIT_ID    = 1                                # split 1~5 중 선택
SPLIT_RATIO = "75%"                            # "75%" or "50%"

IMG_SIZE    = 112                              # 프레임 리사이즈
SEQ_LENGTH  = 16                               # 시퀀스당 프레임 수
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
RANDOM_STATE = 42

# [리뷰] 안전위반 분류 기준 — 13개 액션을 이진 분류로 매핑
#        위반(8개): 폭행/무기/침입 관련 액션 → label=1
#        정상(5개): 일상적 이동/자세 액션 → label=0
#        이 분류는 Kaggle 경진대회 "Action Recognition for Real-Time Safety
#        Monitoring"의 평가 기준을 반영
SAFETY_VIOLATIONS = {"fall", "grab", "gun", "hit", "kick",
                     "sneak", "struggle", "throw"}
NORMAL_ACTIONS    = {"walk", "stand", "sit", "run", "lying_down"}
ALL_ACTIONS = sorted(SAFETY_VIOLATIONS | NORMAL_ACTIONS)

VIOLATION_LABEL = 1  # 위반
NORMAL_LABEL    = 0  # 정상

ACTION_TO_SAFETY = {}
for a in SAFETY_VIOLATIONS:
    ACTION_TO_SAFETY[a] = VIOLATION_LABEL
for a in NORMAL_ACTIONS:
    ACTION_TO_SAFETY[a] = NORMAL_LABEL

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 유틸
# ============================================================
class Timer:
    def __init__(self, name):
        self.name = name
        self.t0 = time.time()
        log.info(f"[START] {self.name}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        log.info(f"[DONE] {self.name} — {time.time() - self.t0:.2f}s")


# ============================================================
# 1. Split 파일 파싱
# ============================================================
with Timer("Split 파일 파싱"):
    split_dir = os.path.join(SPLIT_DIR, SPLIT_RATIO)

    records = []
    for action in ALL_ACTIONS:
        fpath = os.path.join(split_dir, f"{action}_test_split{SPLIT_ID}.txt")
        if not os.path.exists(fpath):
            log.warning(f"  파일 없음: {fpath}")
            continue
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    fname = parts[0]
                    label = int(parts[1])
                    records.append({
                        "filename": fname,
                        "action": action,
                        "source_label": label,
                        "safety": ACTION_TO_SAFETY[action],
                        "safety_str": "VIOLATION" if ACTION_TO_SAFETY[action] == 1 else "NORMAL",
                        "video_path": os.path.join(VIDEO_DIR, fname),
                    })

    df = pd.DataFrame(records)
    log.info(f"  전체 클립: {len(df)}")
    log.info(f"  액션 분포:\n{df['action'].value_counts().to_string()}")
    log.info(f"  안전위반: {len(df[df['safety'] == 1])} (VIOLATION), "
             f"{len(df[df['safety'] == 0])} (NORMAL)")

    # 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    order = df['action'].value_counts().index
    sns.countplot(data=df, y="action", order=order, ax=axes[0],
                  palette="viridis")
    axes[0].set_title("액션별 클립 수")

    safety_colors = {"VIOLATION": "#F44336", "NORMAL": "#4CAF50"}
    for label, grp in df.groupby("safety_str"):
        axes[1].barh(label, len(grp), color=safety_colors[label], alpha=0.8)
    axes[1].set_title("안전위반 vs 정상")
    axes[1].set_xlabel("클립 수")

    plt.tight_layout()
    plt.savefig("lecture/h3_data_distribution.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h3_data_distribution.png")


# ============================================================
# 2. 영상 프레임 추출
# ============================================================
def extract_frames(video_path, seq_length=SEQ_LENGTH, img_size=IMG_SIZE):
    """
    비디오에서 균등 간격으로 seq_length개 프레임을 추출하여 리사이즈.
    """
    # [리뷰] 프레임 추출 전략 — 균등 샘플링(uniform sampling)
    #        연속 프레임을 가져오면 중복 정보가 많고, 시간 범위가 좁음
    #        대신 전체 영상에 걸쳐 균등하게 분포된 프레임을 선택
    #        예: 160프레임 영상에서 16프레임 추출 → 0, 10, 20, ..., 150
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None

    # np.linspace로 시작~끝 프레임을 seq_length개로 균등 분할
    if total_frames >= seq_length:
        indices = np.linspace(0, total_frames - 1, seq_length, dtype=int)
    else:
        # 짧은 영상은 마지막 프레임 반복 (zero-padding 대신)
        indices = list(range(total_frames))
        while len(indices) < seq_length:
            indices.append(total_frames - 1)
        indices = np.array(indices)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames, dtype=np.float32)


def extract_frames_debug(video_path, seq_length=SEQ_LENGTH, img_size=IMG_SIZE):
    """프레임 추출 + 디버그 로그."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"  열기 실패: {video_path}")
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    log.info(f"  영상 정보: {os.path.basename(video_path)} | "
             f"{w}x{h} @ {fps:.1f}fps | {total}프레임 | {duration:.1f}초")
    cap.release()

    return extract_frames(video_path, seq_length, img_size)


# ============================================================
# 3. 데이터셋 로드 (실제 비디오 또는 시뮬레이션)
# ============================================================
with Timer("데이터셋 로드"):
    # [리뷰] 이중 모드 설계: 실제 비디오가 있으면 실제 모드, 없으면 시뮬레이션
    #        시뮬레이션 모드에서는 랜덤 데이터로 파이프라인 전체를 검증
    #        → 실제 비디오를 배치하기 전에 코드 동작을 확인 가능
    #        → 강의/데모 환경에서도 정상 실행됨
    has_videos = os.path.isdir(VIDEO_DIR) and len(
        glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))) > 0

    if has_videos:
        # 실제 비디오에서 프레임 추출
        log.info("  모드: 실제 비디오 프레임 추출")
        X_frames = []
        y_labels = []
        y_actions = []
        skipped = 0

        for _, row in df.iterrows():
            frames = extract_frames(row["video_path"])
            if frames is None:
                skipped += 1
                continue
            X_frames.append(frames)
            y_labels.append(row["safety"])
            y_actions.append(row["action"])

        X_frames = np.array(X_frames) / 255.0  # [0,1] 정규화
        y_labels = np.array(y_labels)
        y_actions = np.array(y_actions)
        log.info(f"  로드 완료: {len(X_frames)}개, 스킵: {skipped}")

    else:
        # 비디오 없음 → 시뮬레이션 모드 (랜덤 프레임으로 파이프라인 검증)
        log.warning("  비디오 파일 없음 → 시뮬레이션 모드")
        log.warning(f"  비디오를 {VIDEO_DIR}/ 에 배치하면 실제 모드로 전환됩니다")

        n_samples = len(df)
        X_frames = np.random.rand(n_samples, SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        y_labels = df["safety"].values
        y_actions = df["action"].values
        log.info(f"  시뮬레이션 데이터: {X_frames.shape}")

    # [리뷰] 클래스 가중치: 위반(8종류) vs 정상(5종류)의 불균형 보정
    #        위반 클립이 더 많으므로 정상 클래스에 더 높은 가중치 부여
    #        공식: weight = total / (2 * class_count)
    #        학습 시 loss 계산에서 가중치가 곱해져 소수 클래스를 무시하지 않게 함
    n_violation = np.sum(y_labels == VIOLATION_LABEL)
    n_normal    = np.sum(y_labels == NORMAL_LABEL)
    total       = len(y_labels)
    class_weight = {
        NORMAL_LABEL:    total / (2 * n_normal)    if n_normal > 0 else 1.0,
        VIOLATION_LABEL: total / (2 * n_violation) if n_violation > 0 else 1.0,
    }
    log.info(f"  클래스 가중치: NORMAL={class_weight[NORMAL_LABEL]:.2f}, "
             f"VIOLATION={class_weight[VIOLATION_LABEL]:.2f}")


# ============================================================
# 4. 프레임 시각화 (샘플)
# ============================================================
with Timer("프레임 시각화"):
    n_show = min(4, len(X_frames))
    fig, axes = plt.subplots(n_show, SEQ_LENGTH, figsize=(SEQ_LENGTH * 1.5, n_show * 2))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        for j in range(SEQ_LENGTH):
            ax = axes[i, j]
            ax.imshow(X_frames[i, j])
            ax.axis("off")
            if j == 0:
                safety_tag = "VIOLATION" if y_labels[i] == 1 else "NORMAL"
                ax.set_ylabel(f"{y_actions[i]}\n({safety_tag})", fontsize=8)

    fig.suptitle("추출된 프레임 시퀀스 샘플", fontsize=14)
    plt.tight_layout()
    plt.savefig("lecture/h3_frame_samples.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h3_frame_samples.png")


# ============================================================
# 5. 학습/검증 분할
# ============================================================
with Timer("데이터 분할"):
    X_train, X_val, y_train, y_val = train_test_split(
        X_frames, y_labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_labels
    )
    log.info(f"  학습: {X_train.shape} (VIOLATION={np.sum(y_train==1)}, NORMAL={np.sum(y_train==0)})")
    log.info(f"  검증: {X_val.shape}   (VIOLATION={np.sum(y_val==1)}, NORMAL={np.sum(y_val==0)})")


# ============================================================
# 6. CNN + LSTM 모델 구축
# ============================================================
with Timer("모델 구축"):
    # 6-1. CNN 특징 추출기 (MobileNetV2 backbone)
    # [리뷰] 전이학습(Transfer Learning): ImageNet으로 사전학습된 모델을 특징 추출기로 사용
    #        MobileNetV2 선택 이유: 경량(3.4M 파라미터), 모바일/실시간 환경에 적합
    #        include_top=False: 분류 레이어 제거, 특징 맵만 출력
    #        trainable=False: 백본 가중치 동결 → 학습 시간 단축, 오버피팅 방지
    cnn_base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    cnn_base.trainable = False  # 초기에는 동결

    # [리뷰] GlobalAveragePooling2D: 특징 맵을 1D 벡터로 압축
    #        (7, 7, 1280) → (1280,) — 공간 정보는 평균으로 요약
    #        Flatten 대비 파라미터 수 대폭 감소
    cnn_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = cnn_base(cnn_input, training=False)
    x = GlobalAveragePooling2D()(x)
    cnn_feat = Model(cnn_input, x, name="cnn_feature_extractor")

    feat_dim = cnn_feat.output_shape[-1]
    log.info(f"  CNN 특징 차원: {feat_dim}")

    # 6-2. CNN 특징 미리 추출 (메모리 효율)
    # [리뷰] 핵심 최적화: CNN+LSTM을 end-to-end로 학습하지 않고 2단계로 분리
    #        1단계: CNN으로 각 프레임의 특징 벡터(1280차원)를 미리 추출
    #        2단계: 추출된 특징 시퀀스만 LSTM에 입력
    #        장점: 메모리 절약, 학습 속도 향상 (CNN 재계산 방지)
    #        캐글 메달 노트북들의 표준 접근법
    log.info("  CNN 특징 추출 중...")
    X_train_feat = np.zeros((X_train.shape[0], SEQ_LENGTH, feat_dim), dtype=np.float32)
    X_val_feat   = np.zeros((X_val.shape[0],   SEQ_LENGTH, feat_dim), dtype=np.float32)

    for s in range(X_train.shape[0]):
        for t in range(SEQ_LENGTH):
            X_train_feat[s, t] = cnn_feat.predict(
                X_train[s:t+1, t], verbose=0
            )[0] if False else cnn_feat(
                X_train[s:t+1][t:t+1], training=False
            ).numpy()[0]

    for s in range(X_val.shape[0]):
        for t in range(SEQ_LENGTH):
            X_val_feat[s, t] = cnn_feat(
                X_val[s:t+1][t:t+1], training=False
            ).numpy()[0]

    log.info(f"  Train 특징: {X_train_feat.shape}")
    log.info(f"  Val   특징: {X_val_feat.shape}")

    # 6-3. LSTM 분류기
    # [리뷰] LSTM이 시계열 분류에 적합한 이유:
    #        프레임 특징의 "시간적 순서"가 액션을 결정
    #        예: 앞사람을 쫓아감(seek→sneak) vs 그냥 걸음(walk)은
    #            시간에 따른 특징 변화 패턴으로 구분 가능
    #        LSTM(128→64): 128유닛이 전체 시퀀스를 읽고, 64유닛이 요약
    #        Dense(1, sigmoid): 0~1 사이 확률 출력 (위반 확률)
    lstm_model = Sequential([
        LSTM(128, input_shape=(SEQ_LENGTH, feat_dim), return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")  # 이진 분류: 위반 확률 [0, 1]
    ])

    lstm_model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    log.info(f"  LSTM 파라미터 수: {lstm_model.count_params():,}")


# ============================================================
# 7. 모델 학습 (실시간 디버깅)
# ============================================================
with Timer("LSTM 학습"):
    class DebugCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info(
                    f"  Epoch {epoch+1:03d}/{EPOCHS} — "
                    f"loss={logs['loss']:.4f} acc={logs['accuracy']:.4f} "
                    f"val_loss={logs['val_loss']:.4f} val_acc={logs['val_accuracy']:.4f}"
                )

    history = lstm_model.fit(
        X_train_feat, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_feat, y_val),
        class_weight=class_weight,
        callbacks=[DebugCallback()],
        verbose=0
    )

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lecture/h3_training_curves.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h3_training_curves.png")


# ============================================================
# 8. 평가
# ============================================================
with Timer("평가"):
    y_pred_prob = lstm_model.predict(X_val_feat, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred, average="weighted")

    log.info(f"  Accuracy: {acc:.4f}")
    log.info(f"  F1 (weighted): {f1:.4f}")
    log.info(f"\n{classification_report(y_val, y_pred, target_names=['NORMAL', 'VIOLATION'])}")

    # 혼동 행렬
    cm = confusion_matrix(y_val, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["NORMAL", "VIOLATION"],
                yticklabels=["NORMAL", "VIOLATION"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix (Acc={acc:.4f}, F1={f1:.4f})")

    # 위반 확률 분포
    normal_probs    = y_pred_prob[y_val == NORMAL_LABEL]
    violation_probs = y_pred_prob[y_val == VIOLATION_LABEL]
    axes[1].hist(normal_probs,    bins=30, alpha=0.6, label="NORMAL",    color="#4CAF50")
    axes[1].hist(violation_probs, bins=30, alpha=0.6, label="VIOLATION", color="#F44336")
    axes[1].axvline(x=0.5, color="black", linestyle="--", label="Threshold=0.5")
    axes[1].set_xlabel("위반 확률")
    axes[1].set_ylabel("빈도")
    axes[1].set_title("안전위반 확률 분포")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lecture/h3_evaluation.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h3_evaluation.png")


# ============================================================
# 9. 실시간 안전위반 탐지 데모
# ============================================================
with Timer("실시간 탐지 데모"):
    def detect_safety_violation(video_path, model, cnn_model,
                                seq_length=SEQ_LENGTH, img_size=IMG_SIZE,
                                threshold=0.5):
        """
        단일 비디오에 대해 안전위반 탐지 수행.
        Returns: (is_violation, confidence, frames)
        """
        # [리뷰] 실시간 추론 파이프라인 — 새로운 비디오 한 개를 처리하는 함수
        #        1. 프레임 추출 (균등 샘플링)
        #        2. CNN 특징 추출 (MobileNetV2)
        #        3. LSTM 분류 (위반 확률 예측)
        #        4. 임계값(0.5)과 비교하여 최종 판정
        #        실제 서비스에서는 이 함수를 스트리밍 프레임에 대해 반복 호출
        frames = extract_frames(video_path, seq_length, img_size)
        if frames is None:
            return None, 0.0, None

        frames_norm = frames / 255.0

        # CNN 특징 추출
        feats = np.zeros((1, seq_length, feat_dim), dtype=np.float32)
        for t in range(seq_length):
            feats[0, t] = cnn_model(
                frames_norm[t:t+1], training=False
            ).numpy()[0]

        # LSTM 예측
        prob = model.predict(feats, verbose=0)[0, 0]
        is_violation = prob >= threshold

        return is_violation, float(prob), frames

    # 검증셋에서 위반 탐지 데모
    demo_indices = np.random.choice(len(X_val), size=min(10, len(X_val)), replace=False)

    results = []
    for idx in demo_indices:
        actual = "VIOLATION" if y_val[idx] == 1 else "NORMAL"
        prob = y_pred_prob[idx]
        pred = "VIOLATION" if y_pred[idx] == 1 else "NORMAL"
        match = "O" if y_pred[idx] == y_val[idx] else "X"
        results.append({
            "actual": actual,
            "predicted": pred,
            "violation_prob": f"{prob:.3f}",
            "match": match,
        })
        log.info(f"  [{match}] 실제={actual:10s} | 예측={pred:10s} | 위반확률={prob:.3f}")

    results_df = pd.DataFrame(results)
    log.info(f"\n{results_df.to_string(index=False)}")

    # 탐지 결과 시각화
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#F44336" if r["actual"] == "VIOLATION" else "#4CAF50" for r in results]
    probs  = [float(r["violation_prob"]) for r in results]
    labels = [f"{r['actual']}\n{'O' if r['match']=='O' else 'X'}" for r in results]

    bars = ax.bar(range(len(probs)), probs, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=2, label="Threshold=0.5")
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("위반 확률")
    ax.set_title("안전위반 탐지 결과 (빨강=VIOLATION, 초록=NORMAL)")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig("lecture/h3_detection_demo.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h3_detection_demo.png")


# ============================================================
# 10. 비디오 프레임 추출 유틸리티 (독립 사용 가능)
# ============================================================
def extract_and_save_frames(video_path, output_dir, fps=5, img_size=224):
    """
    비디오에서 지정 FPS로 프레임을 추출하여 이미지로 저장.
    실시간 디버깅용 진행 상황 출력.

    Args:
        video_path: 입력 비디오 경로
        output_dir: 프레임 저장 디렉토리
        fps:        추출할 초당 프레임 수
        img_size:   출력 이미지 크기
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        log.error(f"비디오 열기 실패: {video_path}")
        return 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip = max(1, int(orig_fps / fps))

    log.info(f"입력: {os.path.basename(video_path)} | {w}x{h} @ {orig_fps:.1f}fps | "
             f"총 {total}프레임")
    log.info(f"추출: {fps}fps (매 {skip}프레임마다) → 약 {total // skip}프레임")

    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            frame = cv2.resize(frame, (img_size, img_size))
            out_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(out_path, frame)
            count += 1
            if count % 50 == 0:
                log.info(f"  {count}프레임 추출 완료...")
        frame_idx += 1

    cap.release()
    log.info(f"완료: 총 {count}프레임 → {output_dir}")
    return count


# ============================================================
# 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("결과 요약")
print("=" * 60)
print(f"  데이터셋: CCTV Action Recognition (split {SPLIT_ID}, {SPLIT_RATIO})")
print(f"  총 클립: {len(df)} (VIOLATION: {n_violation}, NORMAL: {n_normal})")
print(f"  위반 액션: {sorted(SAFETY_VIOLATIONS)}")
print(f"  정상 액션: {sorted(NORMAL_ACTIONS)}")
print(f"  모델: MobileNetV2(CNN) + LSTM")
print(f"  프레임: {SEQ_LENGTH}프레임/시퀀스, {IMG_SIZE}x{IMG_SIZE}")
print(f"")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 Score: {f1:.4f}")
mode_str = "시뮬레이션" if not has_videos else "실제 비디오"
print(f"  모드: {mode_str}")
if not has_videos:
    print(f"\n  ※ 실제 비디오를 {VIDEO_DIR}/ 에 배치하면 실제 모드로 실행됩니다")
print(f"\n  생성된 파일:")
for f in sorted(
    x for x in os.listdir("lecture")
    if x.startswith("h3_") and (x.endswith(".png") or x.endswith(".log"))
):
    print(f"    lecture/{f}")
print("\n디버그 로그: lecture/h3_debug.log")
print("완료.")

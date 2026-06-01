"""
MVTec AD — 산업 이미지 이상탐지 (Autoencoder + PatchCore)
==========================================================
데이터셋: MVTec AD (MVTec Anomaly Detection)
    - 15개 카테고리: bottle, cable, capsule, carpet, grid, hazelnut,
      leather, metal_nut, pill, screw, tile, toothbrush, transistor,
      wood, zipper
    - train/: 정상(good) 이미지만
    - test/:  정상(good) + 다양한 결함 유형
    - ground_truth/: 결함 마스크 (픽셀 수준)
    - 이미지: 700~1024px, RGB PNG

참고 캐글 메달 노트북:
    1. akshaysom, "Anomaly Detection using PatchCore from Scratch"
       — WideResNet backbone + memory bank + kNN 이상 점수
    2. ohseokkim, "Dectecting Anomaly using Autoencoder!"
       — Autoencoder reconstruction error 기반 임계값 탐지
    3. ipythonx, "MVTec-AD: Anomaly Detection with Anomalib Library"
       — Anomalib PatchCore/PaDiM

파이프라인:
    1. 데이터 로드 & 전처리
    2. EDA — 카테고리별 분포 시각화
    3. Autoencoder 학습 (정상 데이터만)
    4. 재구성 오차 기반 이상 탐지
    5. PatchCore 스타일 특징 기반 탐지
    6. 평가 — AUROC, 시각화
"""

import os
import sys
import logging
import warnings
import time
from glob import glob
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Conv2D, Conv2DTranspose,
                                     Flatten, Reshape, MaxPooling2D, UpSampling2D,
                                     BatchNormalization, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# ============================================================
# 로깅
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("lecture/h4_debug.log", mode="w"),
    ]
)
log = logging.getLogger("MVTec-AD")

# ============================================================
# 설정
# ============================================================
DATA_DIR    = "lecture/h4_data"
CATEGORY    = "bottle"          # 분석할 카테고리
IMG_SIZE    = 256               # 리사이즈 크기
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4
RANDOM_STATE = 42

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


def load_image(path, size=IMG_SIZE):
    """이미지 로드 → 리사이즈 → [0,1] float32."""
    # [리뷰] 전처리 파이프라인: 원본(700~1024px) → 256px 리사이즈 → [0,1] 정규화
    #        resize로 이미지 크기를 통일해야 배열로 결합 가능
    #        /255.0: 픽셀값을 0~255에서 0.0~1.0으로 변환 (신경망 학습 안정화)
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ============================================================
# 1. 데이터 로드
# ============================================================
with Timer("데이터 로드"):
    cat_dir = os.path.join(DATA_DIR, CATEGORY)

    # [리뷰] MVTec AD의 특이한 구조: 학습 데이터에는 "정상(good)" 이미지만 있음!
    #        이것이 비지도 이상탐지(unsupervised anomaly detection)의 핵심 가정:
    #        "정상 샘플만으로 모델을 학습하고, 정상에서 벗어난 것을 이상으로 간주"
    #        실제 산업 환경에서 결함 데이터가 매우 귀소하다는 현실을 반영
    train_paths = sorted(glob(os.path.join(cat_dir, "train", "good", "*.png")))
    train_images = np.array([load_image(p) for p in train_paths])

    # [리뷰] 테스트 데이터는 정상(good) + 다양한 결함 유형이 혼합
    #        결함 유형은 카테고리마다 다름 (bottle: broken_large, broken_small, contamination)
    #        label: 0=정상(good), 1=결함(나머지)
    test_dir = os.path.join(cat_dir, "test")
    defect_types = [d for d in os.listdir(test_dir)
                    if os.path.isdir(os.path.join(test_dir, d))]

    test_images = []
    test_labels = []       # 0=정상, 1=결함
    test_defect = []       # 결함 유형명

    for dtype in defect_types:
        dtype_dir = os.path.join(test_dir, dtype)
        paths = sorted(glob(os.path.join(dtype_dir, "*.png")))
        label = 0 if dtype == "good" else 1
        for p in paths:
            test_images.append(load_image(p))
            test_labels.append(label)
            test_defect.append(dtype)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    log.info(f"  카테고리: {CATEGORY}")
    log.info(f"  학습 (정상): {train_images.shape}")
    log.info(f"  테스트: {test_images.shape} (정상={np.sum(test_labels==0)}, 결함={np.sum(test_labels==1)})")
    log.info(f"  결함 유형: {[d for d in defect_types if d != 'good']}")

    # 결함 유형별 분포
    defect_counts = defaultdict(int)
    for d in test_defect:
        defect_counts[d] += 1
    log.info(f"  결함 분포: {dict(defect_counts)}")


# ============================================================
# 2. EDA
# ============================================================
with Timer("EDA 시각화"):
    fig, axes = plt.subplots(2, len(defect_types), figsize=(3 * len(defect_types), 6))
    if len(defect_types) == 1:
        axes = axes[:, np.newaxis]

    for idx, dtype in enumerate(sorted(defect_types)):
        # 첫 번째 샘플 이미지
        dtype_dir = os.path.join(test_dir, dtype)
        sample = sorted(glob(os.path.join(dtype_dir, "*.png")))[0]
        img = load_image(sample)
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(dtype, fontsize=10)
        axes[0, idx].axis("off")

        # 결함 마스크 (있으면)
        gt_dir = os.path.join(cat_dir, "ground_truth", dtype)
        gt_paths = sorted(glob(os.path.join(gt_dir, "*.png"))) if os.path.isdir(gt_dir) else []
        if gt_paths:
            mask = np.array(Image.open(gt_paths[0]).convert("L").resize((IMG_SIZE, IMG_SIZE)))
            axes[1, idx].imshow(mask, cmap="gray")
            axes[1, idx].set_title("mask", fontsize=9)
        else:
            axes[1, idx].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[1, idx].axis("off")

    fig.suptitle(f"MVTec AD — {CATEGORY} (테스트 샘플)", fontsize=13)
    plt.tight_layout()
    plt.savefig("lecture/h4_eda_samples.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h4_eda_samples.png")

    # 전체 카테고리 분포 요약
    all_cats = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "license.txt"])
    cat_stats = []
    for cat in all_cats:
        n_train = len(glob(os.path.join(DATA_DIR, cat, "train", "good", "*.png")))
        n_test  = len(glob(os.path.join(DATA_DIR, cat, "test", "*", "*.png")))
        n_defect_types = len([d for d in os.listdir(os.path.join(DATA_DIR, cat, "test"))
                              if os.path.isdir(os.path.join(DATA_DIR, cat, "test", d)) and d != "good"])
        cat_stats.append({"category": cat, "train_good": n_train,
                          "test_total": n_test, "defect_types": n_defect_types})

    stats_df = pd.DataFrame(cat_stats)
    log.info(f"\n{stats_df.to_string(index=False)}")


# ============================================================
# 3. Autoencoder 모델 구축
# ============================================================
with Timer("Autoencoder 구축"):
    # [리뷰] 컨볼루셔널 Autoencoder 아키텍처
    #        인코더: 256x256x3 → 128x128x64 → 64x64x128 → 32x32x256 → 16x16x512
    #        잠재공간: 16x16x512 = 131,072차원 (Flatten 후)
    #        디코더: 대칭 구조로 원본 크기로 복원
    #
    #        Conv2D(strides=2): 2배 다운샘플링 (MaxPooling 대신 stride 사용)
    #        Conv2DTranspose(strides=2): 2배 업샘플링
    #        BatchNormalization: 각 레이어 출력 정규화 → 학습 안정화
    #        최종 sigmoid: 출력을 [0,1]로 제한 (정규화된 입력과 동일 범위)
    encoder_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="encoder_input")

    x = Conv2D(64, 3, strides=2, padding="same", activation="relu")(encoder_input)    # 256→128
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)               # 128→64
    x = BatchNormalization()(x)
    x = Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)               # 64→32
    x = BatchNormalization()(x)
    x = Conv2D(512, 3, strides=2, padding="same", activation="relu")(x)               # 32→16
    x = BatchNormalization()(x)

    latent = Flatten(name="latent")(x)  # 잠재공간: 16x16x512 → 131072

    # 디코더: 인코더의 거울상(대칭) 구조
    x = Reshape((16, 16, 512))(latent)
    x = Conv2DTranspose(512, 3, strides=2, padding="same", activation="relu")(x)      # 16→32
    x = BatchNormalization()(x)
    x = Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)      # 32→64
    x = BatchNormalization()(x)
    x = Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)      # 64→128
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)       # 128→256
    x = BatchNormalization()(x)

    decoder_output = Conv2D(3, 3, padding="same", activation="sigmoid", name="decoder_output")(x)

    autoencoder = Model(encoder_input, decoder_output, name="Autoencoder")
    autoencoder.compile(optimizer=Adam(learning_rate=LR), loss="mse", metrics=["mae"])

    log.info(f"  파라미터 수: {autoencoder.count_params():,}")
    autoencoder.summary(print_fn=lambda x: log.info(f"  {x}"))


# ============================================================
# 4. Autoencoder 학습 (정상 데이터만)
# ============================================================
with Timer("Autoencoder 학습"):
    # 정상 데이터를 train/val 분할
    X_train_ae, X_val_ae = train_test_split(
        train_images, test_size=0.1, random_state=RANDOM_STATE
    )

    class DebugCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info(
                    f"  Epoch {epoch+1:03d}/{EPOCHS} — "
                    f"loss={logs['loss']:.6f} mae={logs['mae']:.6f} "
                    f"val_loss={logs['val_loss']:.6f} val_mae={logs['val_mae']:.6f}"
                )

    history = autoencoder.fit(
        X_train_ae, X_train_ae,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_ae, X_val_ae),
        callbacks=[DebugCallback()],
        verbose=0
    )

    # 학습 곡선
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss (MSE)")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train")
    axes[1].plot(history.history["val_mae"], label="Val")
    axes[1].set_title("MAE")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lecture/h4_ae_training.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h4_ae_training.png")


# ============================================================
# 5. 재구성 오차 계산 & 이상 탐지
# ============================================================
with Timer("이상 탐지 (Autoencoder)"):
    # [리뷰] 이상 점수 = 재구성 오차(MSE)
    #        정상 이미지: 학습했으므로 잘 재구성 → 낮은 MSE
    #        결함 이미지: 처음 보는 패턴 → 재구성 실패 → 높은 MSE
    #        axis=(1,2,3): 각 이미지별로 모든 픽셀의 MSE를 평균
    train_recon = autoencoder.predict(train_images, verbose=0)
    train_errors = np.mean((train_images - train_recon) ** 2, axis=(1, 2, 3))

    test_recon = autoencoder.predict(test_images, verbose=0)
    test_errors = np.mean((test_images - test_recon) ** 2, axis=(1, 2, 3))

    # [리뷰] 임계값 산출: 정상 데이터의 mean + 2*std (ohseokkim 노트북 방식)
    #        정규분포 가정 시 약 97.7%의 정상 샘플이 임계값 이하에 위치
    #        즉, 정상 중 약 2.3%는 오탐지(false positive) 가능
    threshold = train_errors.mean() + 2 * train_errors.std()
    log.info(f"  학습 오차: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
    log.info(f"  임계값 (mean+2*std): {threshold:.6f}")

    # 이상 판정
    predictions = (test_errors >= threshold).astype(int)

    # AUROC
    auroc = roc_auc_score(test_labels, test_errors)

    log.info(f"  AUROC: {auroc:.4f}")
    log.info(f"\n{classification_report(test_labels, predictions, target_names=['Normal', 'Defect'])}")

    # 오차 분포 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 5-1. 오차 분포
    normal_errors = test_errors[test_labels == 0]
    defect_errors = test_errors[test_labels == 1]
    axes[0].hist(normal_errors, bins=30, alpha=0.6, label="Normal", color="#4CAF50")
    axes[0].hist(defect_errors, bins=30, alpha=0.6, label="Defect", color="#F44336")
    axes[0].axvline(x=threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
    axes[0].set_title("재구성 오차 분포")
    axes[0].set_xlabel("MSE")
    axes[0].legend()

    # 5-2. ROC 곡선
    fpr, tpr, _ = roc_curve(test_labels, test_errors)
    axes[1].plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    # 5-3. 결함 유형별 오차
    defect_df = pd.DataFrame({"defect": test_defect, "error": test_errors, "label": test_labels})
    defect_order = defect_df.groupby("defect")["error"].median().sort_values(ascending=False).index
    sns.boxplot(data=defect_df, x="error", y="defect", order=defect_order, ax=axes[2])
    axes[2].axvline(x=threshold, color="red", linestyle="--", label="Threshold")
    axes[2].set_title("결함 유형별 재구성 오차")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("lecture/h4_anomaly_scores.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h4_anomaly_scores.png")


# ============================================================
# 6. 재구성 결과 시각화
# ============================================================
with Timer("재구성 시각화"):
    n_show = min(6, len(test_images))
    # 정상 3개 + 결함 3개 선택
    normal_idx = np.where(test_labels == 0)[0][:3]
    defect_idx = np.where(test_labels == 1)[0][:3]
    show_idx = np.concatenate([normal_idx, defect_idx])

    fig, axes = plt.subplots(3, len(show_idx), figsize=(3 * len(show_idx), 9))

    for col, idx in enumerate(show_idx):
        # 원본
        axes[0, col].imshow(test_images[idx])
        axes[0, col].set_title(f"{test_defect[idx]}\n(label={'Defect' if test_labels[idx] else 'Normal'})",
                               fontsize=8)
        axes[0, col].axis("off")

        # 재구성
        axes[1, col].imshow(test_recon[idx])
        axes[1, col].set_title(f"MSE={test_errors[idx]:.4f}", fontsize=8)
        axes[1, col].axis("off")

        # 오차 맵
        diff = np.abs(test_images[idx] - test_recon[idx])
        diff_vis = np.mean(diff, axis=-1)
        axes[2, col].imshow(diff_vis, cmap="hot")
        axes[2, col].set_title("Error Map", fontsize=8)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
    axes[2, 0].set_ylabel("Error Map", fontsize=10)

    plt.suptitle(f"Autoencoder 재구성 결과 — {CATEGORY}", fontsize=13)
    plt.tight_layout()
    plt.savefig("lecture/h4_reconstruction.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h4_reconstruction.png")


# ============================================================
# 7. PatchCore 스타일 특징 기반 탐지
# ============================================================
with Timer("PatchCore 스타일 탐지"):
    # [리뷰] PatchCore: 캐글 메달 노트북에서 가장 인기 있는 이미지 이상탐지 기법
    #        핵심 아이디어: 정상 이미지의 특징을 "메모리 뱅크"에 저장
    #        새 이미지가 들어오면 메모리 뱅크의 가장 가까운 특징과 비교
    #        거리가 멀면 → 이상! (정상 패턴에서 벗어남)
    #
    #        원본 PatchCore은 WideResNet-50을 사용하지만
    #        여기서는 MobileNetV2로 경량화 (실시간 환경 고려)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    # [리뷰] 중간 레이어("out_relu")의 출력을 특징으로 사용
    #        최종 분류 레이어가 아닌 중간 특징이 더 풍부한 정보를 담음
    feat_extractor = Model(
        backbone.input,
        backbone.get_layer("out_relu").output  # 8x8x1280
    )

    # [리뷰] Memory Bank 구축: 모든 정상 이미지의 특징 벡터를 모음
    #        (N, 8, 8, 1280) → (N, 8*8*1280) = (N, 81920)으로 평탄화
    #        이 벡터들이 "정상 패턴의 사전" 역할을 함
    train_feats = feat_extractor.predict(train_images, verbose=0)
    train_feats_flat = train_feats.reshape(train_feats.shape[0], -1)  # (N, 8*8*1280)
    memory_bank = train_feats_flat
    log.info(f"  Memory bank: {memory_bank.shape}")

    # 테스트 특징 추출
    test_feats = feat_extractor.predict(test_images, verbose=0)
    test_feats_flat = test_feats.reshape(test_feats.shape[0], -1)

    # [리뷰] kNN 거리 기반 이상 점수 — PatchCore의 핵심 로직
    #        각 테스트 이미지에 대해:
    #        1. 특징 벡터 추출
    #        2. 메모리 뱅크의 모든 정상 특징과 L2 거리 계산
    #        3. 가장 가까운 k=3개 이웃의 거리 평균 = 이상 점수
    #        점수가 높을수록 = 정상 패턴에서 멀리 떨어짐 = 이상일 가능성 높음
    def compute_anomaly_scores(test_feats, memory_bank, k=3):
        scores = []
        for i in range(len(test_feats)):
            dists = np.sqrt(np.sum((memory_bank - test_feats[i]) ** 2, axis=1))
            top_k_dists = np.sort(dists)[:k]  # k개 최근접 이웃
            scores.append(np.mean(top_k_dists))
        return np.array(scores)

    patchcore_scores = compute_anomaly_scores(test_feats_flat, memory_bank, k=3)

    # AUROC
    pc_auroc = roc_auc_score(test_labels, patchcore_scores)
    log.info(f"  PatchCore AUROC: {pc_auroc:.4f}")

    # 임계값
    pc_train_scores = compute_anomaly_scores(
        feat_extractor.predict(train_images, verbose=0).reshape(train_images.shape[0], -1),
        memory_bank, k=3
    )
    pc_threshold = pc_train_scores.mean() + 2 * pc_train_scores.std()
    pc_predictions = (patchcore_scores >= pc_threshold).astype(int)

    from sklearn.metrics import f1_score
    pc_f1 = f1_score(test_labels, pc_predictions, average="weighted")
    ae_f1 = f1_score(test_labels, predictions, average="weighted")

    log.info(f"  PatchCore 임계값: {pc_threshold:.4f}")
    log.info(f"  PatchCore F1: {pc_f1:.4f}")

    # 비교 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AE 오차 분포
    axes[0].hist(normal_errors, bins=20, alpha=0.6, label="Normal", color="#4CAF50")
    axes[0].hist(defect_errors, bins=20, alpha=0.6, label="Defect", color="#F44336")
    axes[0].axvline(x=threshold, color="black", linestyle="--")
    axes[0].set_title(f"Autoencoder (AUROC={auroc:.4f}, F1={ae_f1:.4f})")
    axes[0].set_xlabel("Reconstruction Error")
    axes[0].legend()

    # PatchCore 점수 분포
    normal_pc = patchcore_scores[test_labels == 0]
    defect_pc = patchcore_scores[test_labels == 1]
    axes[1].hist(normal_pc, bins=20, alpha=0.6, label="Normal", color="#4CAF50")
    axes[1].hist(defect_pc, bins=20, alpha=0.6, label="Defect", color="#F44336")
    axes[1].axvline(x=pc_threshold, color="black", linestyle="--")
    axes[1].set_title(f"PatchCore (AUROC={pc_auroc:.4f}, F1={pc_f1:.4f})")
    axes[1].set_xlabel("Anomaly Score (kNN distance)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lecture/h4_method_comparison.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h4_method_comparison.png")


# ============================================================
# 8. 결함 로컬라이제이션 (픽셀 수준)
# ============================================================
with Timer("결함 로컬라이제이션"):
    # [리뷰] 로컬라이제이션: "이상한지"뿐 아니라 "어디가 이상한지"를 찾는 기술
    #        Autoencoder의 재구성 오차를 픽셀 단위로 계산 → 오차가 큰 위치 = 결함 위치
    #        GT 마스크(ground_truth)와 비교하여 탐지 정확도를 시각적으로 확인
    #        4행 구성: 원본 → GT 마스크 → 오차 히트맵 → 이진화 맵
    n_localize = min(5, len(defect_idx))
    fig, axes = plt.subplots(4, n_localize, figsize=(4 * n_localize, 12))

    for col in range(n_localize):
        idx = defect_idx[col]

        # 원본
        axes[0, col].imshow(test_images[idx])
        axes[0, col].set_title(test_defect[idx], fontsize=9)
        axes[0, col].axis("off")

        # GT 마스크
        gt_dir = os.path.join(cat_dir, "ground_truth", test_defect[idx])
        gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))
        # 테스트 이미지 순서와 GT 순서 매칭
        test_path = glob(os.path.join(test_dir, test_defect[idx], "*.png"))
        gt_idx_in_folder = test_path.index(
            [p for p in test_path if os.path.basename(p) in
             [os.path.basename(p2) for p2 in test_path]][0]
        ) if gt_paths else -1

        if gt_paths and gt_idx_in_folder < len(gt_paths):
            mask = np.array(Image.open(gt_paths[min(gt_idx_in_folder, len(gt_paths)-1)]).convert("L"))
            mask = np.array(Image.fromarray(mask).resize((IMG_SIZE, IMG_SIZE)))
            axes[1, col].imshow(mask, cmap="gray")
        else:
            axes[1, col].text(0.5, 0.5, "N/A", ha="center")
        axes[1, col].set_title("GT Mask", fontsize=9)
        axes[1, col].axis("off")

        # 재구성 오차 맵
        diff = np.mean(np.abs(test_images[idx] - test_recon[idx]), axis=-1)
        axes[2, col].imshow(diff, cmap="hot")
        axes[2, col].set_title("AE Error Map", fontsize=9)
        axes[2, col].axis("off")

        # 이진화된 오차 맵
        # [리뷰] percentile(95): 오차 상위 5%만 결함으로 간주
        #        임계값을 너무 낮추면 정상 영역도 결함으로 오탐지
        #        95퍼센타일은 보수적(conservative)인 설정
        pixel_threshold = np.percentile(diff, 95)
        binary_map = (diff > pixel_threshold).astype(np.float32)
        axes[3, col].imshow(binary_map, cmap="gray")
        axes[3, col].set_title("Binary (p95)", fontsize=9)
        axes[3, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("GT Mask", fontsize=10)
    axes[2, 0].set_ylabel("AE Error", fontsize=10)
    axes[3, 0].set_ylabel("Binary", fontsize=10)

    plt.suptitle(f"결함 로컬라이제이션 — {CATEGORY}", fontsize=13)
    plt.tight_layout()
    plt.savefig("lecture/h4_localization.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h4_localization.png")


# ============================================================
# 9. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("결과 요약")
print("=" * 60)
print(f"  데이터셋: MVTec AD — {CATEGORY}")
print(f"  학습 (정상): {train_images.shape[0]}장")
print(f"  테스트: {test_images.shape[0]}장 (정상={int(np.sum(test_labels==0))}, 결함={int(np.sum(test_labels==1))})")
print(f"  결함 유형: {[d for d in sorted(defect_types) if d != 'good']}")
print(f"")
print(f"  Autoencoder:")
print(f"    임계값: {threshold:.6f}")
print(f"    AUROC:  {auroc:.4f}")
print(f"    F1:     {ae_f1:.4f}")
print(f"")
print(f"  PatchCore (kNN):")
print(f"    임계값: {pc_threshold:.4f}")
print(f"    AUROC:  {pc_auroc:.4f}")
print(f"    F1:     {pc_f1:.4f}")
print(f"")
print(f"  생성된 파일:")
for f in sorted(
    x for x in os.listdir("lecture")
    if x.startswith("h4_") and (x.endswith(".png") or x.endswith(".log"))
):
    print(f"    lecture/{f}")
print("\n디버그 로그: lecture/h4_debug.log")
print("완료.")

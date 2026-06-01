"""
NASA Turbofan Engine Dataset — RUL 라벨링, 클리핑, 실시간 디버깅
================================================================
데이터셋: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
    - 4개 하위셋 (FD001~FD004)
    - 26컬럼: unit, cycle, 3개 operational setting, 21개 센서
    - 학습: run-to-failure, 테스트: failure 이전 시점까지

참고 캐글 메달 노트북:
    1. Damir Kamalov, "RUL engines" — Piecewise Linear RUL, 클리핑, LSTM
    2. fabf001, "NASA Turbofan Jet Engine LSTM Loss s-score" — S-Score 평가
    3. Wassim Derbel, "NASA Predictive Maintenance (RUL)" — Feature Engineering
    4. sumitpr96, "AIML Project" — RUL cap=125

파이프라인:
    1. 데이터 로드 및 컬럼명 부여
    2. EDA — 센서 트렌드, 상수 센서 탐지
    3. RUL 라벨 생성 (Linear + Piecewise Clipped)
    4. 상수 센서 제거 & MinMax 정규화
    5. Sliding Window 시퀀스 생성
    6. LSTM 모델 학습 (실시간 디버깅)
    7. 평가 — RMSE, NASA S-Score
    8. 결과 시각화
"""

import os
import sys
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

# ============================================================
# 실시간 디버깅 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("lecture/h2_debug.log", mode="w"),
    ]
)
log = logging.getLogger("NASA-RUL")

# ============================================================
# [설정] NASA C-MAPSS 데이터셋의 실험 하이퍼파라미터
# ============================================================
# [리뷰] RUL_CAP=125: 캐글 메달 노트북에서 가장 널리 쓰이는 클리핑 상한값
#        초기 엔진 수명에서는 열화 패턴이 관측되지 않으므로
#        RUL이 125 이상이면 모두 125로 처리 → 모델이 의미 있는 열화에 집중
#        SEQUENCE_LENGTH=30: 30사이클(=시간단위)의 과거 데이터를 보고 RUL 예측
DATA_DIR = "lecture/h2_data/CMaps"
DATASET_ID = "FD001"        # FD001~FD004 중 선택
RUL_CAP = 125               # Piecewise Linear RUL 클리핑 상한
SEQUENCE_LENGTH = 30        # Sliding window 길이
BATCH_SIZE = 256
EPOCHS = 50
LR = 0.001
RANDOM_STATE = 42

# [리뷰] 원본 데이터에는 컬럼명이 없음 (공백으로 구분된 숫자만 있음)
#        readme.txt에 명시된 26개 컬럼명을 수동으로 정의하여 부여
COL_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 유틸: 실시간 진행 로그 헬퍼
# ============================================================
# [리뷰] Timer: with문과 함께 사용하는 컨텍스트 매니저
#        with Timer("작업명"): 블록 진입 시 START 로그, 종료 시 DONE + 소요시간 출력
#        실시간 디버깅의 핵심 — 각 단계별 성능 병목 파악 가능
class Timer:
    def __init__(self, name):
        self.name = name
        self.t0 = time.time()
        log.info(f"[START] {self.name}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.t0
        log.info(f"[DONE] {self.name} — {elapsed:.2f}s")


def debug_df(df, name="DataFrame"):
    log.info(f"  [{name}] shape={df.shape}, units={df['unit'].nunique()}, "
             f"cycles_range=({df['cycle'].min()},{df['cycle'].max()}), "
             f"nulls={df.isnull().sum().sum()}")


# ============================================================
# 1. 데이터 로드
# ============================================================
with Timer("데이터 로드"):
    # [리뷰] 데이터 파일은 공백으로 구분된 텍스트 파일
    #        sep=r"\s+": 하나 이상의 공백(스페이스/탭)을 구분자로 인식
    #        header=None: 파일에 헤더 행이 없음
    #        RUL_FD001.txt: 테스트 엔진 100개의 실제 잔여수명 (정답)
    train_path = os.path.join(DATA_DIR, f"train_{DATASET_ID}.txt")
    test_path  = os.path.join(DATA_DIR, f"test_{DATASET_ID}.txt")
    rul_path   = os.path.join(DATA_DIR, f"RUL_{DATASET_ID}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COL_NAMES)
    test_df  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=COL_NAMES)
    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul_true"])

    debug_df(train_df, "Train")
    debug_df(test_df,  "Test")
    log.info(f"  RUL true: shape={rul_true.shape}, range=({rul_true['rul_true'].min()},{rul_true['rul_true'].max()})")


# ============================================================
# 2. EDA — 센서 분포 및 트렌드
# ============================================================
with Timer("EDA"):
    # 2-1. 센서별 표준편차 → 상수 센서 탐지
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    sensor_std = train_df[sensor_cols].std().sort_values()

    log.info("  센서별 표준편차 (낮은 순):")
    for col, std in sensor_std.items():
        flag = " *** 상수/거의 상수" if std < 0.5 else ""
        log.info(f"    {col}: std={std:.4f}{flag}")

    # [리뷰] 상수 센서 자동 감지: 표준편차 < 0.5인 센서는 정보량이 없음
    #        예: sensor_1(518.67 고정), sensor_5(14.62 고정) 등
    #        이런 센서를 그대로 사용하면 노이즈만 추가되므로 제거
    #        캐글 메달 노트북들의 공통 전처리 단계
    constant_sensors = sensor_std[sensor_std < 0.5].index.tolist()
    useful_sensors   = [s for s in sensor_cols if s not in constant_sensors]
    log.info(f"  제거 대상 상수 센서 ({len(constant_sensors)}개): {constant_sensors}")
    log.info(f"  유효 센서 ({len(useful_sensors)}개): {useful_sensors}")

    # 2-2. 센서 트렌드 시각화 (샘플 엔진)
    sample_unit = train_df["unit"].iloc[0]
    sample_df   = train_df[train_df["unit"] == sample_unit]

    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 20))
    for idx, sensor in enumerate(sensor_cols):
        ax = axes[idx // 3, idx % 3]
        ax.plot(sample_df["cycle"], sample_df[sensor], linewidth=0.8)
        ax.set_title(sensor, fontsize=9)
        if sensor in constant_sensors:
            ax.set_facecolor("#ffcccc")
    fig.suptitle(f"센서 트렌드 — Unit {sample_unit} (빨간 배경=상수 센서)", fontsize=14)
    plt.tight_layout()
    plt.savefig("lecture/h2_sensor_trends.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h2_sensor_trends.png")


# ============================================================
# 3. RUL 라벨 생성 — Linear + Piecewise Clipped
# ============================================================
with Timer("RUL 라벨 생성"):
    def add_rul_label(df, cap=None):
        """
        각 엔진(유닛)의 각 시점(cycle)에 대해 RUL 라벨을 생성.
        - Linear RUL: max_cycle - current_cycle
        - Clipped RUL (cap): min(linear_rul, cap)
        """
        # [리뷰] RUL 라벨링의 핵심 로직
        #        1. 각 엔진의 마지막 사이클(max_cycle)을 구함
        #        2. Linear RUL = max_cycle - 현재 cycle (단순 카운트다운)
        #        3. Clipped RUL = min(linear_rul, cap) → 상한선 적용
        #
        #        예시 (cap=125):
        #          cycle 1:  linear_RUL=192, clipped_RUL=125
        #          cycle 68: linear_RUL=125, clipped_RUL=125  ← 여기서부터 감소 시작
        #          cycle 193: linear_RUL=0,   clipped_RUL=0
        #
        #        클리핑 효과: 초기에 RUL이 너무 높으면 모델이 의미 없는 학습을 함
        #        열화가 관측되는 구간(대략 마지막 125사이클)에 집중하게 만듦
        result = df.copy()
        max_cycles = result.groupby("unit")["cycle"].max().reset_index()
        max_cycles.columns = ["unit", "max_cycle"]
        result = result.merge(max_cycles, on="unit")

        result["rul_linear"] = result["max_cycle"] - result["cycle"]

        if cap is not None:
            # [리뷰] .clip(upper=cap): 값이 cap을 초과하면 cap으로 잘라냄
            result[f"rul_clip_{cap}"] = result["rul_linear"].clip(upper=cap)
            log.info(f"  RUL 클리핑 적용: cap={cap} "
                     f"(linear range: [{result['rul_linear'].min()}, {result['rul_linear'].max()}] "
                     f"→ clipped range: [{result[f'rul_clip_{cap}'].min()}, {result[f'rul_clip_{cap}'].max()}])")
        return result

    train_df = add_rul_label(train_df, cap=RUL_CAP)
    debug_df(train_df, "Train (RUL added)")

    # RUL 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(train_df["rul_linear"], bins=50, color="#2196F3", alpha=0.7)
    axes[0].set_title("Linear RUL 분포")
    axes[0].set_xlabel("RUL (cycles)")
    axes[0].axvline(x=RUL_CAP, color="red", linestyle="--", label=f"cap={RUL_CAP}")
    axes[0].legend()

    axes[1].hist(train_df[f"rul_clip_{RUL_CAP}"], bins=50, color="#4CAF50", alpha=0.7)
    axes[1].set_title(f"Clipped RUL 분포 (cap={RUL_CAP})")
    axes[1].set_xlabel("RUL (cycles)")

    plt.tight_layout()
    plt.savefig("lecture/h2_rul_distribution.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h2_rul_distribution.png")

    # 샘플 엔진 RUL 비교
    fig, ax = plt.subplots(figsize=(12, 4))
    sample_rul = train_df[train_df["unit"] == sample_unit]
    ax.plot(sample_rul["cycle"], sample_rul["rul_linear"], label="Linear RUL", linewidth=2)
    ax.plot(sample_rul["cycle"], sample_rul[f"rul_clip_{RUL_CAP}"], label=f"Clipped RUL (cap={RUL_CAP})",
            linewidth=2, linestyle="--", color="red")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_title(f"RUL 비교 — Unit {sample_unit}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("lecture/h2_rul_comparison.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h2_rul_comparison.png")


# ============================================================
# 4. 전처리 — 상수 센서 제거 & 정규화
# ============================================================
with Timer("전처리"):
    feature_cols = useful_sensors  # 상수 센서 제거된 센서만 사용

    # [리뷰] MinMaxScaler: 데이터를 [0, 1] 범위로 변환
    #        중요: fit은 학습 데이터에만, transform은 학습/테스트 모두에 적용
    #        이유: 테스트 데이터로 fit하면 데이터 누수(data leakage) 발생
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    log.info(f"  정규화 완료: {len(feature_cols)}개 센서 (상수 {len(constant_sensors)}개 제거)")
    log.info(f"  Train 센서 범위: [{train_df[feature_cols].min().min():.4f}, {train_df[feature_cols].max().max():.4f}]")
    log.info(f"  Test  센서 범위: [{test_df[feature_cols].min().min():.4f}, {test_df[feature_cols].max().max():.4f}]")


# ============================================================
# 5. Sliding Window 시퀀스 생성
# ============================================================
with Timer("Sliding Window 시퀀스 생성"):
    def generate_sequences(df, feature_cols, target_col, seq_length):
        """
        각 엔진별로 sliding window 시퀀스 생성.
        길이가 seq_length 미만인 엔진은 패딩 없이 건너뜀.
        """
        # [리뷰] Sliding Window가 RUL 예측에 필수적인 이유:
        #        LSTM은 시계열 패턴을 학습하므로 과거 N개 사이클을 시퀀스로 입력
        #        stride=1 (한 칸씩 이동) → 데이터 증강 효과
        #        타겟은 시퀀스의 "마지막 시점"의 RUL (i + seq_length - 1)
        X, y, units = [], [], []
        for unit_id, group in df.groupby("unit"):
            data  = group[feature_cols].values
            label = group[target_col].values
            if len(data) < seq_length:
                log.warning(f"  Unit {unit_id}: 시퀀스 길이 {len(data)} < {seq_length}, 건너뜀")
                continue
            # [리뷰] 슬라이딩 윈도우: 매 시점마다 seq_length길이의 시퀀스 생성
            #         총 (데이터길이 - seq_length + 1)개의 샘플이 한 엔진에서 생성됨
            for i in range(len(data) - seq_length + 1):
                X.append(data[i : i + seq_length])
                y.append(label[i + seq_length - 1])  # 시퀀스 마지막 시점의 RUL
                units.append(unit_id)
        return np.array(X), np.array(y), np.array(units)

    target_col = f"rul_clip_{RUL_CAP}"

    X_train, y_train, _ = generate_sequences(train_df, feature_cols, target_col, SEQUENCE_LENGTH)
    log.info(f"  Train 시퀀스: X={X_train.shape}, y={y_train.shape}")
    log.info(f"  y_train 통계: mean={y_train.mean():.1f}, std={y_train.std():.1f}, "
             f"range=[{y_train.min()}, {y_train.max()}]")


# ============================================================
# 6. 테스트 데이터 준비 (시퀀스의 마지막 것만 사용)
# ============================================================
with Timer("테스트 데이터 준비"):
    def get_test_sequences(df, feature_cols, seq_length):
        """각 엔진의 마지막 시퀀스만 추출 (최종 시점 RUL 예측용)."""
        # [리뷰] 학습과 테스트의 시퀀스 생성 방식이 다름!
        #        학습: 모든 sliding window 생성 (데이터 증강)
        #        테스트: 각 엔진의 "마지막 시퀀스" 1개만 사용
        #        이유: Kaggle 평가 방식이 마지막 시점의 RUL 예측이기 때문
        X_test, units = [], []
        for unit_id, group in df.groupby("unit"):
            data = group[feature_cols].values
            if len(data) < seq_length:
                # [리뷰] 시퀀스보다 짧은 엔진: 앞을 0으로 패딩
                #        실제 데이터를 시퀀스 뒤쪽에 배치 → 최근 데이터 우선
                padded = np.zeros((seq_length, len(feature_cols)))
                padded[-len(data):] = data
                X_test.append(padded)
            else:
                X_test.append(data[-seq_length:])  # 마지막 seq_length개 시간단계
            units.append(unit_id)
        return np.array(X_test), np.array(units)

    X_test_final, test_units = get_test_sequences(test_df, feature_cols, SEQUENCE_LENGTH)
    y_test_final = rul_true["rul_true"].values

    log.info(f"  Test 시퀀스: X={X_test_final.shape}, y_true={y_test_final.shape}")
    log.info(f"  y_test 통계: mean={y_test_final.mean():.1f}, std={y_test_final.std():.1f}, "
             f"range=[{y_test_final.min()}, {y_test_final.max()}]")


# ============================================================
# 7. LSTM 모델 학습 (실시간 디버깅 콜백)
# ============================================================
with Timer("LSTM 모델 학습"):
    # 7-1. 커스텀 콜백: 실시간 디버깅
    class RealtimeDebugCallback(Callback):
        """매 에포크마다 상세 로그 출력."""
        def __init__(self, log_every=5):
            super().__init__()
            self.log_every = log_every

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if (epoch + 1) % self.log_every == 0 or epoch == 0:
                log.info(
                    f"  Epoch {epoch+1:03d}/{EPOCHS} — "
                    f"loss={logs.get('loss', 0):.4f} "
                    f"mae={logs.get('mae', 0):.4f} "
                    f"val_loss={logs.get('val_loss', 0):.4f} "
                    f"val_mae={logs.get('val_mae', 0):.4f} "
                    f"lr={float(self.model.optimizer.learning_rate):.6f}"
                )

    # 7-2. NASA S-Score 커스텀 메트릭
    # [리뷰] NASA S-Score: RMSE와 다른 관점의 평가 지표
    #        d = 예측 - 실제 (오차의 부호가 중요!)
    #        d < 0 (조기 예측): exp(-d/10) - 1 → 덜 벌점
    #        d >= 0 (늦은 예측): exp(d/13) - 1 → 더 벌점
    #        의미: 늦게 고장을 예측하는 것이 더 위험하므로 가중치를 높게 부여
    #        캐글 메달 노트북에서 RMSE와 함께 필수적으로 사용
    def nasa_s_score(y_true, y_pred):
        """
        NASA S-Score: s = exp(-d/10) - 1    if d < 0 (조기 예측)
                       = exp(d/13) - 1       if d >= 0 (늦은 예측)
        d = y_pred - y_true
        """
        d = y_pred - y_true
        s = tf.where(d < 0,
                     tf.exp(-d / 10.0) - 1.0,
                     tf.exp(d / 13.0) - 1.0)
        return tf.reduce_mean(s)

    # 7-3. 모델 구성
    # [리뷰] LSTM 기반 RUL 예측 모델 아키텍처
    #        입력: (30사이클, n_features개 센서값)
    #        LSTM(64, return_sequences=True): 모든 시간단계 출력 (다음 LSTM에 전달)
    #        LSTM(32, return_sequences=False): 마지막 시간단계만 출력 (시퀀스 요약)
    #        Dense(1, linear): RUL 회귀값 출력 (활성화 없음 = 선형)
    #        BatchNormalization: 학습 안정화,梯度 폭발/소실 방지
    n_features = len(feature_cols)
    model = Sequential([
        LSTM(64, input_shape=(SEQUENCE_LENGTH, n_features), return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")  # 회귀: RUL 값 그대로 출력
    ])

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="mse",
        metrics=["mae", nasa_s_score]
    )

    log.info(f"  모델 파라미터 수: {model.count_params():,}")
    model.summary(print_fn=lambda x: log.info(f"  {x}"))

    # 7-4. 학습
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[RealtimeDebugCallback(log_every=5)],
        verbose=0  # 콜백으로 로그 출력
    )

    # 학습 곡선 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, metric, title in [
        (axes[0], "loss", "Loss (MSE)"),
        (axes[1], "mae", "MAE"),
        (axes[2], "nasa_s_score", "NASA S-Score"),
    ]:
        ax.plot(history.history[metric], label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig("lecture/h2_training_curves.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h2_training_curves.png")


# ============================================================
# 8. 평가 — RMSE, S-Score, 시각화
# ============================================================
with Timer("평가"):
    y_pred = model.predict(X_test_final, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, None)  # 음수 RUL 방지

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))

    # NASA S-Score (numpy)
    d = y_pred - y_test_final
    s_score = np.mean(np.where(d < 0, np.exp(-d / 10.0) - 1.0, np.exp(d / 13.0) - 1.0))

    log.info(f"  RMSE: {rmse:.2f}")
    log.info(f"  NASA S-Score: {s_score:.4f}")

    # 8-1. 예측 vs 실제 산점도
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(y_test_final, y_pred, alpha=0.5, s=20)
    axes[0].plot([0, max(y_test_final)], [0, max(y_test_final)], "r--", label="Perfect")
    axes[0].set_xlabel("True RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title(f"RUL 예측 vs 실제 (RMSE={rmse:.2f}, S-Score={s_score:.4f})")
    axes[0].legend()

    # 8-2. 엔진별 RUL 비교 바차트
    n_show = min(30, len(y_test_final))
    x_pos = np.arange(n_show)
    width = 0.35
    axes[1].bar(x_pos - width/2, y_test_final[:n_show], width, label="True RUL", alpha=0.7)
    axes[1].bar(x_pos + width/2, y_pred[:n_show], width, label="Predicted RUL", alpha=0.7)
    axes[1].set_xlabel("Unit Index")
    axes[1].set_ylabel("RUL")
    axes[1].set_title(f"엔진별 RUL (처음 {n_show}개)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lecture/h2_rul_prediction.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h2_rul_prediction.png")

    # 8-3. 오차 분포
    errors = y_pred - y_test_final
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(errors, bins=30, color="#FF9800", alpha=0.7, edgecolor="black")
    axes[0].axvline(x=0, color="red", linestyle="--")
    axes[0].set_title("예측 오차 분포 (Pred - True)")
    axes[0].set_xlabel("Error (cycles)")

    axes[1].scatter(y_test_final, np.abs(errors), alpha=0.5, s=20)
    axes[1].set_xlabel("True RUL")
    axes[1].set_ylabel("|Error|")
    axes[1].set_title("오차 크기 vs True RUL")

    plt.tight_layout()
    plt.savefig("lecture/h2_error_analysis.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h2_error_analysis.png")


# ============================================================
# 9. 클리핑 효과 비교 (Linear vs Clipped RUL 학습)
# ============================================================
with Timer("클리핑 효과 비교"):
    # [리뷰] 이 단계가 이 코드의 핵심 시연 포인트!
    #        동일한 LSTM 모델을 두 가지 타겟으로 학습하여 성능 비교:
    #        1. Linear RUL (클리핑 없음): RUL = 192, 191, ... , 1, 0
    #        2. Clipped RUL (cap=125):    RUL = 125, 125, ... , 1, 0
    #
    #        예상 결과: Clipped RUL이 더 낮은 RMSE, S-Score를 보여야 함
    #        이유: 모델이 초기 고RUL 값의 노이즈에 방해받지 않고
    #              의미 있는 열화 패턴(125 이하)에 집중할 수 있기 때문
    # Linear RUL로 동일 모델 학습 (비교용)
    target_linear = "rul_linear"
    X_train_lin, y_train_lin, _ = generate_sequences(
        train_df, feature_cols, target_linear, SEQUENCE_LENGTH
    )

    model_linear = Sequential([
        LSTM(64, input_shape=(SEQUENCE_LENGTH, n_features), return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])
    model_linear.compile(optimizer=Adam(learning_rate=LR), loss="mse", metrics=["mae"])
    model_linear.fit(X_train_lin, y_train_lin, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_split=0.2, verbose=0)

    y_pred_lin = model_linear.predict(X_test_final, verbose=0).flatten()
    y_pred_lin = np.clip(y_pred_lin, 0, None)
    rmse_lin   = np.sqrt(mean_squared_error(y_test_final, y_pred_lin))
    d_lin      = y_pred_lin - y_test_final
    s_score_lin = np.mean(np.where(d_lin < 0, np.exp(-d_lin/10.0)-1.0, np.exp(d_lin/13.0)-1.0))

    log.info(f"  [Linear RUL]    RMSE: {rmse_lin:.2f}, S-Score: {s_score_lin:.4f}")
    log.info(f"  [Clipped {RUL_CAP}]  RMSE: {rmse:.2f}, S-Score: {s_score:.4f}")

    # 비교 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, pred, title, rmse_val, s_val in [
        (axes[0], y_pred_lin, f"Linear RUL (RMSE={rmse_lin:.2f})", rmse_lin, s_score_lin),
        (axes[1], y_pred,    f"Clipped RUL cap={RUL_CAP} (RMSE={rmse:.2f})", rmse, s_score),
    ]:
        ax.scatter(y_test_final, pred, alpha=0.5, s=20)
        ax.plot([0, max(y_test_final)], [0, max(y_test_final)], "r--", label="Perfect")
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig("lecture/h2_clipping_comparison.png", dpi=150)
    plt.close()
    log.info("  → 저장: lecture/h2_clipping_comparison.png")


# ============================================================
# 10. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("결과 요약")
print("=" * 60)
print(f"  데이터셋: NASA C-MAPSS {DATASET_ID}")
print(f"  사용 센서: {len(feature_cols)}개 (상수 {len(constant_sensors)}개 제거)")
print(f"  RUL 클리핑: cap={RUL_CAP}")
print(f"  시퀀스 길이: {SEQUENCE_LENGTH}")
print(f"  학습 샘플: {X_train.shape[0]}")
print(f"  테스트 엔진: {X_test_final.shape[0]}")
print(f"")
print(f"  Linear RUL    → RMSE: {rmse_lin:.2f}, S-Score: {s_score_lin:.4f}")
print(f"  Clipped({RUL_CAP}) RUL → RMSE: {rmse:.2f}, S-Score: {s_score:.4f}")
print(f"")
print(f"  생성된 파일:")
for f in sorted(
    x for x in os.listdir("lecture")
    if x.startswith("h2_") and (x.endswith(".png") or x.endswith(".log"))
):
    print(f"    lecture/{f}")
print("\n디버그 로그: lecture/h2_debug.log")
print("완료.")

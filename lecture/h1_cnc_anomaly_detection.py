"""
CNC 밀링머신 진동데이터셋 — 전처리 및 이상탐지
================================================
데이터셋: CNC Machining Data (Kaggle, Maximilian Fellhuber)
    - 3축 가속도계 (acc_x, acc_y, acc_z), 2 kHz 샘플링
    - 3대의 CNC 머신 (M01, M02, M03)
    - label_AB.csv 형식 (A=머신번호, B=프로세스/상태)

참고 분석:
    1. Cendikia Ishmatuka, "Autoencoder: Anomaly Detection for Vibration Data"
       — Autoencoder + IQR threshold 기반 이상탐지
    2. ResearchGate, "Anomaly Detection in CNC Machining Using Dynamic-Input LSTM"
       — LSTM 기반 시계열 이상탐지
    3. MDPI Applied Sciences (2025), "Deep Learning for Anomaly Detection in CNC
       Machine Vibration Data" — RoughLSTM, 94.3% 정확도

파이프라인:
    1. 데이터 로드 및 EDA
    2. 윈도잉 & 다운샘플링
    3. 정규화 (Min-Max)
    4. 특징 추출 (시간 도메인 + 주파수 도메인)
    5. 이상탐지 — Autoencoder + Isolation Forest
    6. 결과 시각화
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from scipy import stats
from scipy.signal import welch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ============================================================
# [설정] 실험 하이퍼파라미터와 데이터 경로를 정의합니다
# ============================================================
# [리뷰] SAMPLE_RATE=2000은 가속도계의 원본 샘플링 주파수(2kHz)입니다
#        WINDOW_SEC=10은 10초 단위로 데이터를 잘라 윈도우를 만든다는 의미
#        즉, 하나의 윈도우 = 10초 × 2000Hz = 20,000개 샘플
DATA_DIR = "lecture/h1_data/Dataset/Dataset"
SAMPLE_RATE = 2000          # 2 kHz
WINDOW_SEC = 10             # 10초 윈도우
DOWNSAMPLE_FACTOR = 10      # 다운샘플링 계수
WINDOW_SIZE = WINDOW_SEC * SAMPLE_RATE  # 20000 샘플/윈도우
RANDOM_STATE = 42

# [리뷰] 파일명 label_AB.csv에서 A=머신번호, B=상태번호를 의미
#        M01(label_0X): 정상 운전 데이터 → 정상 라벨로 사용
#        M02(label_1X): 혼합 데이터 → 학습에 사용하지 않음
#        M03(label_2X): 다양한 이상 패턴 포함 → 이상 라벨로 사용
NORMAL_FILES = ["label_00", "label_01", "label_02"]       # M01 (정상)
MIXED_FILES  = ["label_11", "label_12"]                   # M02
ANOMALY_FILES = ["label_20", "label_21", "label_22",      # M03 (이상 혼합)
                 "label_23", "label_24", "label_25", "label_26"]

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 1. 데이터 로드
# ============================================================
def load_data(data_dir, files, max_windows=None):
    """CSV 파일들을 로드하여 윈도우 단위로 분할."""
    # [리뷰] 연속 시계열을 고정 길이 윈도우로 자르는 이유:
    #        1) 메모리 관리: 파일당 2.4M 행 → 120개 윈도우로 분할
    #        2) 특징 추출: 윈도우 단위로 통계/주파수 특징을 계산
    #        3) max_windows: 실험 속도를 위해 샘플 수 제한
    windows = []
    labels = []
    for fname in files:
        path = os.path.join(data_dir, f"{fname}.csv")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} 없음")
            continue
        print(f"  로드 중: {fname}.csv")
        df = pd.read_csv(path)
        data_np = df.to_numpy()  # (N, 3) — acc_x, acc_y, acc_z

        n_samples = data_np.shape[0]
        n_full_windows = n_samples // WINDOW_SIZE

        # [리뷰] 슬라이딩 윈도우가 아닌 비겹침(non-overlapping) 분할
        #        → 윈도우 간 독립성 보장, 데이터 누수 방지
        for i in range(n_full_windows):
            if max_windows and len(windows) >= max_windows:
                break
            start = i * WINDOW_SIZE
            end = start + WINDOW_SIZE
            windows.append(data_np[start:end])

    # 최종 shape: (윈도우 수, 20000, 3)
    return np.array(windows)  # (n_windows, WINDOW_SIZE, 3)


print("=" * 60)
print("1. 데이터 로드")
print("=" * 60)

normal_windows = load_data(DATA_DIR, NORMAL_FILES, max_windows=100)
mixed_windows  = load_data(DATA_DIR, MIXED_FILES,  max_windows=50)
anomaly_windows = load_data(DATA_DIR, ANOMALY_FILES, max_windows=100)

print(f"\n정상 윈도우:   {normal_windows.shape}")
print(f"혼합 윈도우:   {mixed_windows.shape}")
print(f"이상 윈도우:   {anomaly_windows.shape}")


# ============================================================
# 2. EDA (탐색적 데이터 분석)
# ============================================================
print("\n" + "=" * 60)
print("2. EDA — 원시 진동 데이터 시각화")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 8))
axis_names = ["acc_x", "acc_y", "acc_z"]
colors = ["#2196F3", "#4CAF50", "#F44336"]

for i, ax in enumerate(axes):
    # 정상 데이터 첫 윈도우 (처음 0.1초 = 200포인트만 표시)
    ax.plot(normal_windows[0, :200, i], color=colors[0], alpha=0.7, label="Normal (M01)")
    # 이상 데이터 첫 윈도우
    ax.plot(anomaly_windows[0, :200, i], color=colors[2], alpha=0.7, label="Anomaly (M03)")
    ax.set_title(f"{axis_names[i]} — 첫 0.1초 비교")
    ax.set_ylabel("가속도 (mg)")
    ax.legend()

axes[-1].set_xlabel("샘플 인덱스")
plt.tight_layout()
plt.savefig("lecture/h1_eda_raw.png", dpi=150)
plt.close()
print("  → 저장: lecture/h1_eda_raw.png")

# [리뷰] KDE(커널 밀도 추정) 플롯으로 정상/이상 데이터의 분포 차이를 확인
#        두 분포가 겹치면 분류가 어렵고, 분리되어 있으면 이상탐지가 용이
#        [::100]은 메모리 절약을 위해 100개 간격으로 서브샘플링
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, ax in enumerate(axes):
    normal_flat = normal_windows[:, :, i].flatten()
    anomaly_flat = anomaly_windows[:, :, i].flatten()
    # 서브샘플링하여 플롯
    sns.kdeplot(normal_flat[::100], ax=ax, label="Normal", color="#2196F3")
    sns.kdeplot(anomaly_flat[::100], ax=ax, label="Anomaly", color="#F44336")
    ax.set_title(f"{axis_names[i]} 분포")
    ax.legend()
plt.tight_layout()
plt.savefig("lecture/h1_eda_distribution.png", dpi=150)
plt.close()
print("  → 저장: lecture/h1_eda_distribution.png")


# ============================================================
# 3. 전처리: 다운샘플링
# ============================================================
print("\n" + "=" * 60)
print("3. 전처리 — 다운샘플링")
print("=" * 60)

def downsample(data, factor):
    """윈도우 데이터를 factor만큼 다운샘플링 (평균 풀링)."""
    # [리뷰] 다운샘플링이 필요한 이유:
    #        원본 20,000포인트 → 그대로 사용하면 특징 추출/모델 학습에 과도한 연산
    #        factor=10으로 평균 풀링 → 2,000포인트로 축소
    #        단순 샘플링(간격 띄우기)이 아닌 평균 풀링을 사용해 정보 손실 최소화
    n_windows = data.shape[0]
    new_len = data.shape[1] // factor
    n_axes = data.shape[2]
    result = np.zeros((n_windows, new_len, n_axes))
    for w in range(n_windows):
        for a in range(n_axes):
            # reshape으로 factor개씩 묶은 뒤 평균 → 평균 풀링
            reshaped = data[w, :new_len * factor, a].reshape(new_len, factor)
            result[w, :, a] = reshaped.mean(axis=1)
    return result

normal_ds   = downsample(normal_windows, DOWNSAMPLE_FACTOR)
anomaly_ds  = downsample(anomaly_windows, DOWNSAMPLE_FACTOR)
mixed_ds    = downsample(mixed_windows, DOWNSAMPLE_FACTOR)

print(f"  다운샘플링 후: {normal_ds.shape[1]} 포인트/윈도우 (2000 → {20000 // DOWNSAMPLE_FACTOR})")
print(f"  정상: {normal_ds.shape}, 이상: {anomaly_ds.shape}, 혼합: {mixed_ds.shape}")


# ============================================================
# 4. 특징 추출 (Feature Extraction)
# ============================================================
print("\n" + "=" * 60)
print("4. 특징 추출 — 시간 & 주파수 도메인")
print("=" * 60)

def extract_time_features(window):
    """단일 윈도우(샘플수, 3축)에서 시간 도메인 특징 추출."""
    # [리뷰] 진동 신호에서 의미 있는 9개 시간 도메인 특징을 추출
    #        3축 × 9특징 = 27개 시간 특징이 생성됨
    #        - mean/std: 중심경향과 산포도
    #        - skew/kurtosis: 분포의 비대칭성과 첨도 (이상 탐지에 중요)
    #        - RMS: 실효값, 진동 세기의 대표 지표
    #        - peak-to-peak: 진폭의 최대 범위
    #        - MAD: 이상치에 강건한 산포 측도
    features = []
    for axis in range(window.shape[1]):
        sig = window[:, axis]
        features.extend([
            np.mean(sig),                                       # 평균
            np.std(sig),                                        # 표준편차
            np.min(sig),                                        # 최솟값
            np.max(sig),                                        # 최댓값
            stats.skew(sig),                                    # 왜도
            stats.kurtosis(sig),                                # 첨도
            np.sqrt(np.mean(sig ** 2)),                         # RMS (실효값)
            np.max(np.abs(sig)) - np.min(np.abs(sig)),         # peak-to-peak
            np.median(np.abs(sig) - np.median(sig)),           # MAD (중앙값 절대 편차)
        ])
    return features

def extract_freq_features(window, fs=200):
    """단일 윈도우에서 주파수 도메인 특징 추출 (Welch PSD)."""
    # [리뷰] Welch 방법으로 파워 스펙트럼 밀도(PSD)를 추정
    #        fs=200: 다운샘플링 후 샘플링 주파수 (2000/10=200Hz)
    #        3축 × 5특징 = 15개 주파수 특징이 생성됨
    #        - dominant frequency: 가장 강한 진동 주파수
    #        - spectral centroid: 스펙트럼의 무게중심 주파수
    #        - total power: 전체 진동 에너지
    features = []
    for axis in range(window.shape[1]):
        sig = window[:, axis]
        freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
        features.extend([
            np.max(psd),                                 # 최대 파워
            np.mean(psd),                                # 평균 파워
            freqs[np.argmax(psd)],                       # 지배 주파수 (dominant freq)
            np.sum(psd),                                 # 총 파워 (에너지)
            np.sum(psd * freqs) / np.sum(psd),           # 스펙트럼 중심 (centroid)
        ])
    return features

def extract_features(data):
    """전체 데이터셋 특징 추출."""
    all_features = []
    for w in range(data.shape[0]):
        t_feat = extract_time_features(data[w])
        f_feat = extract_freq_features(data[w])
        all_features.append(t_feat + f_feat)
    return np.array(all_features)

# 특징 이름 생성
time_feat_names = []
freq_feat_names = []
for axis_name in axis_names:
    time_feat_names.extend([
        f"{axis_name}_mean", f"{axis_name}_std", f"{axis_name}_min",
        f"{axis_name}_max", f"{axis_name}_skew", f"{axis_name}_kurtosis",
        f"{axis_name}_rms", f"{axis_name}_ptp", f"{axis_name}_mad",
    ])
    freq_feat_names.extend([
        f"{axis_name}_psd_max", f"{axis_name}_psd_mean",
        f"{axis_name}_dom_freq", f"{axis_name}_total_power",
        f"{axis_name}_spectral_centroid",
    ])
feature_names = time_feat_names + freq_feat_names

print(f"  추출 특징 수: {len(feature_names)} (시간: {len(time_feat_names)}, 주파수: {len(freq_feat_names)})")

normal_feat   = extract_features(normal_ds)
anomaly_feat  = extract_features(anomaly_ds)
mixed_feat    = extract_features(mixed_ds)

print(f"  정상 특징 행렬: {normal_feat.shape}")
print(f"  이상 특징 행렬: {anomaly_feat.shape}")

# 특징 데이터프레임 생성 (시각화용)
feat_columns = feature_names
normal_df = pd.DataFrame(normal_feat, columns=feat_columns)
anomaly_df = pd.DataFrame(anomaly_feat, columns=feat_columns)
normal_df["label"] = 0  # 정상
anomaly_df["label"] = 1  # 이상

# 주요 특징 분포 비교
top_features = ["acc_z_std", "acc_z_rms", "acc_x_kurtosis", "acc_y_ptp",
                "acc_z_psd_max", "acc_z_total_power"]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for idx, feat in enumerate(top_features):
    if feat not in feat_columns:
        continue
    ax = axes[idx // 3, idx % 3]
    sns.kdeplot(normal_df[feat], ax=ax, label="Normal", color="#2196F3")
    sns.kdeplot(anomaly_df[feat], ax=ax, label="Anomaly", color="#F44336")
    ax.set_title(feat)
    ax.legend()
plt.tight_layout()
plt.savefig("lecture/h1_feature_distribution.png", dpi=150)
plt.close()
print("  → 저장: lecture/h1_feature_distribution.png")


# ============================================================
# 5. 전처리: 정규화 및 학습/검증 분할
# ============================================================
print("\n" + "=" * 60)
print("5. 정규화 & 데이터 분할")
print("=" * 60)

# 특징 기반 접근용
X_all = np.vstack([normal_feat, anomaly_feat])
y_all = np.concatenate([np.zeros(len(normal_feat)), np.ones(len(anomaly_feat))])

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
)

scaler_feat = MinMaxScaler()
X_train_scaled = scaler_feat.fit_transform(X_train)
X_test_scaled  = scaler_feat.transform(X_test)

print(f"  학습: {X_train_scaled.shape} (정상 {int(y_train.sum())}, 이상 {int((1-y_train).sum())})")
print(f"  테스트: {X_test_scaled.shape} (정상 {int(y_test.sum())}, 이상 {int((1-y_test).sum())})")

# Autoencoder용 — 평탄화된 다운샘플링 데이터
# [리뷰] (윈도우수, 2000, 3) → (윈도우수, 6000)으로 평탄화
#        Autoencoder는 1D 벡터를 입출력으로 사용하므로 평탄화 필요
normal_flat = normal_ds.reshape(normal_ds.shape[0], -1)   # (n, 2000*3)
anomaly_flat = anomaly_ds.reshape(anomaly_ds.shape[0], -1)
mixed_flat = mixed_ds.reshape(mixed_ds.shape[0], -1)

# Min-Max 정규화 (샘플별)
# [리뷰] 중요: 샘플별(per-sample) 정규화를 수행
#        전역 정규화가 아닌 각 윈도우를 독립적으로 [0,1] 범위로 변환
#        이유: 윈도우마다 진동 범위가 다를 수 있어, 개별 정규화가 더 안정적
def normalize_per_sample(data):
    min_v = data.min(axis=1, keepdims=True)
    max_v = data.max(axis=1, keepdims=True)
    return (data - min_v) / (max_v - min_v + 1e-11)  # 1e-11로 division-by-zero 방지

normal_norm   = normalize_per_sample(normal_flat)
anomaly_norm  = normalize_per_sample(anomaly_flat)
mixed_norm    = normalize_per_sample(mixed_flat)

# [리뷰] Autoencoder의 핵심 원칙: 정상 데이터로만 학습!
#        정상 패턴만 학습 → 이상 데이터는 재구성을 잘 못함 → 오차가 크게 발생
#        이 오차 차이를 이용해 이상을 탐지하는 것이 Autoencoder 기반 이상탐지의 원리
X_ae_train, X_ae_val = train_test_split(normal_norm, test_size=0.2, random_state=RANDOM_STATE)
print(f"  AE 학습: {X_ae_train.shape}, AE 검증: {X_ae_val.shape}")


# ============================================================
# 6. 이상탐지 — Isolation Forest (특징 기반)
# ============================================================
print("\n" + "=" * 60)
print("6. 이상탐지 — Isolation Forest")
print("=" * 60)

# [리뷰] Isolation Forest: 결정 트리 기반 이상탐지 알고리즘
#        원리: 이상치는 특성 공간에서 고립되기 쉽다 → 적은 분할로 격리 가능
#        - n_estimators=200: 트리 개수 (많을수록 안정적)
#        - contamination=0.3: 데이터의 30%가 이상일 것으로 가정
#        - n_jobs=-1: 모든 CPU 코어 사용
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

# [리뷰] Isolation Forest 출력: 1=정상, -1=이상
#        우리 라벨링: 0=정상, 1=이상 → 변환 필요
#        predict()==-1 이면 True(1=이상), 아니면 False(0=정상)
y_pred_iso = (iso_forest.predict(X_test_scaled) == -1).astype(int)

print("\n[Isolation Forest 결과]")
print(classification_report(y_test, y_pred_iso, target_names=["Normal", "Anomaly"]))


# ============================================================
# 7. 이상탐지 — Autoencoder (재구성 오차 기반)
# ============================================================
print("\n" + "=" * 60)
print("7. 이상탐지 — Autoencoder")
print("=" * 60)

input_dim = X_ae_train.shape[1]  # 6000 (2000포인트 × 3축)

# [리뷰] Autoencoder 아키텍처 설계
#        인코더: 6000 → 512 → 128 → 64 → 32 (정보 압축)
#        디코더: 32 → 64 → 128 → 512 → 6000 (정보 복원)
#        잠재공간(latent)=32: 원본 6000차원을 32차원으로 압축
#        - Dropout(0.2): 과적합 방지, 20% 뉴런을 랜덤하게 비활성화
#        - 최종 레이어 sigmoid: 출력을 [0,1] 범위로 제한 (정규화된 입력과 매칭)
#        - loss="mae": MSE보다 이상치에 덜 민감하여 안정적 학습
input_layer = Input(shape=(input_dim,))
encoded = Dense(512, activation="relu")(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(128, activation="relu")(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)  # ← 잠재공간 (bottleneck)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(512, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss="mae")
autoencoder.summary()

# 정상 데이터로만 학습
EPOCHS = 50
BATCH_SIZE = 32

history = autoencoder.fit(
    X_ae_train, X_ae_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_ae_val, X_ae_val),
    shuffle=True,
    verbose=1
)

# 학습 곡선
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("Autoencoder Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig("lecture/h1_ae_training.png", dpi=150)
plt.close()
print("  → 저장: lecture/h1_ae_training.png")


# ============================================================
# 8. 임계값 설정 & 이상 판정
# ============================================================
print("\n" + "=" * 60)
print("8. 임계값 설정 & 이상 판정")
print("=" * 60)

# [리뷰] 임계값 설정 — IQR(사분위수 범위) 방법
#        핵심 질문: "재구성 오차가 얼마나 커야 이상으로 판정할까?"
#        접근: 정상 데이터의 오차 분포를 분석 → 통계적 임계값 산출
#        - Q1(25%), Q3(75%): 정상 오차의 중간 50% 범위
#        - IQR = Q3 - Q1: 데이터의 산포도
#        - upper_threshold = Q3 + 1.5*IQR: 박스플롯에서 이상치 기준과 동일
#        참고: 캐글 메달 노트북에서 널리 사용되는 방식
val_pred = autoencoder.predict(X_ae_val, verbose=0)
val_loss = tf.keras.losses.mae(val_pred, X_ae_val).numpy()

Q1 = np.quantile(val_loss, 0.25)
Q3 = np.quantile(val_loss, 0.75)
IQR = Q3 - Q1
upper_threshold = Q3 + 1.5 * IQR

print(f"  Q1={Q1:.6f}, Q3={Q3:.6f}, IQR={IQR:.6f}")
print(f"  이상탐지 임계값 (상한): {upper_threshold:.6f}")

# 임계값 분포 시각화
plt.figure(figsize=(10, 4))
sns.histplot(val_loss, bins=30, alpha=0.8, label="Normal Val Loss")
plt.axvline(x=upper_threshold, color="red", linestyle="--", label=f"Threshold={upper_threshold:.4f}")
plt.title("재구성 오차 분포 & 임계값")
plt.legend()
plt.tight_layout()
plt.savefig("lecture/h1_threshold.png", dpi=150)
plt.close()
print("  → 저장: lecture/h1_threshold.png")

# [리뷰] 테스트 구성: 앞쪽은 정상, 뒤쪽은 이상 데이터를 순차 배치
#        이렇게 구성하면 시각화(9단계)에서 정상/이상 경계를 명확히 볼 수 있음
test_all_norm = np.vstack([normalize_per_sample(normal_flat), anomaly_norm])
test_labels = np.concatenate([np.zeros(len(normal_flat)), np.ones(len(anomaly_flat))])

test_pred = autoencoder.predict(test_all_norm, verbose=0)
test_loss = tf.keras.losses.mae(test_pred, test_all_norm).numpy()

# [리뷰] 핵심 판정 로직: 재구성 오차 > 임계값 → 이상!
#        정상 데이터는 학습했으므로 잘 재구성 → 오차 작음
#        이상 데이터는 처음 보는 패턴 → 재구성 실패 → 오차 큼
ae_pred = (test_loss > upper_threshold).astype(int)

print("\n[Autoencoder 결과]")
print(classification_report(test_labels, ae_pred, target_names=["Normal", "Anomaly"]))

# 혼동 행렬
cm = confusion_matrix(test_labels, ae_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Autoencoder Confusion Matrix")
plt.tight_layout()
plt.savefig("lecture/h1_confusion_matrix.png", dpi=150)
plt.close()
print("  → 저장: lecture/h1_confusion_matrix.png")


# ============================================================
# 9. 재구성 오차 시각화
# ============================================================
print("\n" + "=" * 60)
print("9. 재구성 오차 시각화")
print("=" * 60)

plt.figure(figsize=(14, 5))
plt.plot(test_loss, alpha=0.7, linewidth=0.5)
plt.axhline(y=upper_threshold, color="red", linestyle="--", label=f"Threshold={upper_threshold:.4f}")
plt.axvline(x=len(normal_flat), color="green", linestyle=":", alpha=0.7, label="Normal | Anomaly 경계")
plt.xlabel("샘플 인덱스")
plt.ylabel("재구성 오차 (MAE)")
plt.title("샘플별 재구성 오차 — 정상 vs 이상")
plt.legend()
plt.tight_layout()
plt.savefig("lecture/h1_reconstruction_error.png", dpi=150)
plt.close()
print("  → 저장: lecture/h1_reconstruction_error.png")


# ============================================================
# 10. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("10. 결과 요약")
print("=" * 60)

iso_f1 = f1_score(y_test, y_pred_iso, average="weighted")
ae_f1  = f1_score(test_labels, ae_pred, average="weighted")

print(f"  Isolation Forest F1 (weighted): {iso_f1:.4f}")
print(f"  Autoencoder     F1 (weighted): {ae_f1:.4f}")
print(f"\n  생성된 시각화 파일:")
for f in sorted(glob("lecture/h1_*.png")):
    print(f"    - {f}")
print("\n완료.")

# 세션 2 실습 — Claude Code 시연 & 실습 모음


> 세션 2(딥러닝 이상 탐지)의 각 강의 섹션에 흩어져 있던 **Claude Code 시연**과 **실습** 부분을 한곳에 모았습니다.
> 각 항목의 이론·배경 설명은 해당 강의 섹션을 참고하세요.


---

```{admonition} 출처: 섹션 1 · Autoencoder
:class: note dropdown
아래 시연/실습은 원래 **섹션 1 · Autoencoder** 섹션에 있던 내용입니다.
```

## 1-3. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: Autoencoder 구조 설계보다 **재구성 오차가 실제로 이상 신호를 잡는지** 확인하는 과정에 집중하세요.
```

**Claude에게 던질 프롬프트 예시**:
```
제조 진동 데이터 이상탐지용 Autoencoder를 만들어줘.
- 정상 데이터: 사인파 기반 합성 신호 (1000개)
- 이상 데이터: 노이즈가 추가된 신호 (50개, 학습에는 사용 안 함)
- PyTorch로 Autoencoder 구현 (input_dim=64, latent_dim=8)
- 정상 데이터로만 학습 (epoch=50)
- 결과: 정상/이상 재구성 오차 분포를 겹쳐서 시각화
- 95th percentile 임계값 표시
```

**시연 흐름**:
1. 정상/이상 합성 데이터 생성
2. Autoencoder 학습 (정상만)
3. 정상/이상 각각 재구성 오차 계산
4. 분포 히스토그램 비교 + 임계값 표시
5. **Claude에게 추가 질문**: *"latent_dim을 8에서 2로 줄이면 분리도가 어떻게 달라져?"*

---

## 1-4. 실습

### 과제

`latent_dim` (잠재 공간 차원)을 바꿔가며 정상/이상 분리도를 비교하세요.

| 실험 | latent_dim | 관찰 포인트 |
|------|-----------|------------|
| A | 32 | 압축이 거의 없음 → 이상도 잘 복원? |
| B | 16 | 기준선 |
| C | 8 | 강의 시연과 동일 |
| D | 4 | 강한 압축 → 분리도 변화? |

**분리도 측정 방법**:
```python
# 정상과 이상의 재구성 오차 분포가 얼마나 겹치는지 확인
# 간단한 방법: 임계값(95th pct) 기준으로 이상 탐지율(Recall) 계산
threshold = np.percentile(normal_errors, 95)
recall = (anomaly_errors > threshold).mean()
print(f"latent_dim={latent_dim}, 이상 탐지율: {recall:.2%}")
```

**제출 항목**:
- 4가지 latent_dim의 재구성 오차 분포 그래프
- 각 설정의 이상 탐지율(Recall) 비교 표
- "압축이 너무 강하거나 너무 약할 때 각각 어떤 문제가 생기는가?" 한 문단

### 실습 시작 코드

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(42)
t = np.linspace(0, 1, 64)

# 정상 데이터: 사인파 (약간의 노이즈 포함)
normal_data = np.array([
    np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(64)
    for _ in range(1000)
], dtype=np.float32)

# 이상 데이터: 스파이크 또는 위상 변이
anomaly_data = np.array([
    np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(64)  # 강한 노이즈
    for _ in range(50)
], dtype=np.float32)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_and_evaluate(latent_dim, normal_data, anomaly_data, epochs=50):
    # TODO: 모델 생성 → 학습 → 오차 계산 → 분리도 평가
    pass

for latent_dim in [32, 16, 8, 4]:
    train_and_evaluate(latent_dim, normal_data, anomaly_data)
```

---

```{admonition} 출처: 섹션 2 · RUL 예측
:class: note dropdown
아래 시연/실습은 원래 **섹션 2 · RUL 예측** 섹션에 있던 내용입니다.
```

## 2-3. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: 모델 구현보다 **학습 곡선과 예측 결과 해석**에 집중하세요.
```

**Claude에게 던질 프롬프트 예시**:
```
NASA Turbofan FD001 데이터로 LSTM 기반 RUL 예측 모델을 만들어줘.
- 데이터 로드 및 RUL 라벨 생성 (clip 125)
- window_size=30, 센서 14개 사용
- LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
- Early Stopping (patience=10)
- 결과 시각화 2개:
  1. 학습/검증 손실 곡선
  2. 실제 RUL vs 예측 RUL 산점도 (대각선 기준선 포함)
```

**시연 후 Claude에게 추가 질문**:
> *"RMSE는 낮은데 실제로 조기경보가 잘 안 될 수 있는 이유가 뭐야?  
> Scoring Function이라는 비대칭 평가 지표는 왜 필요한 거야?"*

**시연 흐름**:
1. 데이터 로드 + RUL 라벨 생성 확인
2. 슬라이딩 윈도우 시퀀스 구성
3. LSTM 모델 정의 및 학습
4. 학습 곡선 (Early Stopping 발동 시점 확인)
5. 예측 vs 실제 비교 산점도

---

## 2-4. 실습

### 과제

LSTM 레이어 수와 Dropout 비율 조합을 바꿔보며 과적합과 성능의 트레이드오프를 분석하세요.

| 실험 | LSTM 레이어 | Dropout | 예상 결과 |
|------|-----------|---------|----------|
| A | 1개 | 0.0 | 기준선 (과적합 가능) |
| B | 1개 | 0.2 | Dropout 효과 확인 |
| C | 2개 | 0.0 | 복잡한 모델 (과적합↑) |
| D | 2개 | 0.2 | 복잡 + 정규화 |

**제출 항목**:
- 4가지 조합의 학습/검증 손실 곡선 (subplot)
- 각 조합의 RMSE 비교 표
- "Dropout이 없을 때와 있을 때 학습 곡선이 어떻게 달랐는가?" 한 문단

### 실습 시작 코드

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# NASA Turbofan FD001 데이터 로드
# 다운로드: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD001.txt', sep=' ', header=None, names=cols)
train_df = train_df.dropna(axis=1)

# RUL 라벨 생성 (클리핑 포함)
max_cycle = train_df.groupby('unit')['cycle'].max()
train_df = train_df.merge(max_cycle.rename('max_cycle'), on='unit')
train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)

# 센서 선택 (변동이 있는 14개)
sensor_cols = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']

def create_sequences(df, sensor_cols, window_size=30):
    X, y = [], []
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit][sensor_cols].values
        unit_rul = df[df['unit'] == unit]['RUL'].values
        for i in range(len(unit_data) - window_size):
            X.append(unit_data[i:i+window_size])
            y.append(unit_rul[i+window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = create_sequences(train_df, sensor_cols)

# 정규화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_reshaped = X.reshape(-1, len(sensor_cols))
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# 학습/검증 분리
split = int(len(X_scaled) * 0.8)
X_train, X_val = X_scaled[:split], X_scaled[split:]
y_train, y_val = y[:split], y[split:]

results = {}
for n_layers, dropout_rate in [(1, 0.0), (1, 0.2), (2, 0.0), (2, 0.2)]:
    # TODO: 모델 구성 → 학습 → RMSE 계산
    pass

# 결과 비교 시각화
# TODO
```

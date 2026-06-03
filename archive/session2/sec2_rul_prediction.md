# 섹션 2 | RUL 예측 — 얼마나 버틸 수 있는가

---

## 2-1. 문제 제기

**RUL(Remaining Useful Life)**: 지금부터 설비가 고장날 때까지 남은 사이클(또는 시간)

```{mermaid}
flowchart LR
    A["① 사후보전\n고장 → 수리\n비용: 최대\n위험: 최대"] --> B["② 예방보전\n일정 주기마다 교체\n비용: 중간\n위험: 낮음"]
    B --> C["③ 예지보전 ← 목표\nRUL 예측 기반 교체\n비용: 최소\n위험: 최저"]
```

**RUL의 정의**:
```
엔진이 총 300 사이클 후 고장난다면:

사이클  1: RUL = 299
사이클 100: RUL = 200
사이클 250: RUL = 50   ← 이 시점에 교체 결정
사이클 300: RUL = 0    ← 고장
```

→ **핵심 질문**: 현재 센서 데이터만 보고 RUL을 예측할 수 있을까?

---

## 2-2. 이론

### ① 왜 일반 회귀가 아닌 LSTM인가

RUL 예측은 결국 숫자를 맞추는 회귀(Regression) 문제입니다.  
그런데 **왜 선형 회귀나 일반 신경망(MLP)으로는 부족할까요?**

```
[문제: 순서가 중요한 데이터]

센서 값만 보면:
  사이클 50:  온도=520, 압력=14.6, 진동=0.02  → RUL = ?
  사이클 250: 온도=521, 압력=14.7, 진동=0.02  → RUL = ?

두 샘플의 순간 값이 거의 같아도 RUL은 전혀 다름
→ "지금 값"이 아니라 "값이 변해온 패턴"이 중요

MLP: 각 샘플을 독립적으로 처리 → 순서 정보 무시
LSTM: 이전 상태를 기억하며 처리 → 변화 패턴 학습
```

---

### ② LSTM의 직관

LSTM(Long Short-Term Memory)은 **"무엇을 기억하고, 무엇을 버릴지"를 학습하는** 순환 신경망입니다.

```
[LSTM 셀의 직관적 이해]

일반 RNN의 문제:
  먼 과거 정보가 현재에 영향을 못 줌 (기울기 소실)

LSTM의 해결책: 게이트 3개
  ┌─────────────────────────────────────────┐
  │  입력 게이트: "새 정보를 얼마나 받아들일까?" │
  │  망각 게이트: "과거 정보를 얼마나 잊을까?"   │
  │  출력 게이트: "지금 무엇을 내보낼까?"        │
  └─────────────────────────────────────────┘

엔진 마모 예시:
  초반 사이클: 망각 게이트 → 정상 패턴 유지 (최근 값 중심)
  마모 진행중: 입력 게이트 → 온도 상승 패턴 축적
  말기 사이클: 출력 게이트 → "RUL이 낮다"는 신호 출력
```

**시퀀스 입력 구조**:
```
슬라이딩 윈도우 (window_size=30 사이클):

사이클 1~30   → [센서값 30×14] → LSTM → RUL 예측값
사이클 2~31   → [센서값 30×14] → LSTM → RUL 예측값
사이클 3~32   → [센서값 30×14] → LSTM → RUL 예측값
...
```

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(30, 14), return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # RUL 예측값 (회귀)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

---

### ③ 과적합 방지: Dropout과 Early Stopping

제조 센서 데이터는 **노이즈가 많고 샘플 수가 제한적**이라 과적합 위험이 높습니다.

```
[과적합이 발생하는 신호]

학습 손실:    ↓↓↓↓↓↓↓↓↓  (계속 감소)
검증 손실:    ↓↓↓↓↑↑↑↑↑  (어느 순간 다시 상승)
                    ↑
               이 시점이 최적
```

**Dropout**: 학습 중 뉴런의 일부를 랜덤하게 끄는 기법

```python
# Dropout(0.2): 20%의 뉴런을 랜덤하게 비활성화
tf.keras.layers.Dropout(0.2)

# 효과:
# - 특정 뉴런에 과의존하지 않도록 강제
# - 앙상블 효과: 매 배치마다 다른 네트워크를 학습하는 것과 유사
# - 추론 시에는 자동으로 비활성화됨
```

**Early Stopping**: 검증 손실이 개선되지 않으면 학습을 자동으로 멈추는 기법

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',    # 검증 손실을 모니터링
    patience=10,           # 10 epoch 동안 개선 없으면 중단
    restore_best_weights=True  # 가장 좋았던 시점의 가중치 복원
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,            # 충분히 크게 설정
    callbacks=[early_stopping]  # 알아서 멈춤
)
```

---

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

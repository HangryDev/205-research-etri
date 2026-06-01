# 심화 학습 자료

:::{admonition} 심화 학습 자료
:class: note
강의에서 시간 관계상 다루지 못한 심화 내용입니다. 강의 후 궁금증이 생겼을 때 아래 순서로 탐색하시길 권장합니다.
:::

---

## 📘 섹션 1 심화: Autoencoder 변형과 Deep SVDD

:::{admonition} 참고 교재
:class: note
📖 **교재**: *Hands-On Unsupervised Learning Using Python* (Ankur A. Patel, O'Reilly)  
**심화 챕터 로드맵**:
- **Ch.1 Unsupervised Learning Using Python** — 비지도 학습 전체 지형도
- **Ch.3 Autoencoders** — Vanilla AE, Sparse AE, Denoising AE 비교
- **Ch.7 Anomaly Detection with Autoencoders** — 재구성 오차 분포 분석, 임계값 최적화, Deep SVDD 연결
:::

### Autoencoder 변형 비교

| 종류 | 핵심 아이디어 | 이상탐지에서의 장점 |
|------|------------|-----------------|
| Vanilla AE | 기본 압축-복원 | 구현 단순, 기준선 |
| Sparse AE | 잠재 표현에 희소성 제약 | 더 선명한 특징 분리 |
| Denoising AE | 노이즈 입력 → 원본 복원 학습 | 노이즈에 강건한 정상 표현 |
| VAE | 잠재 공간을 확률 분포로 학습 | 이상 점수에 불확실성 포함 |
| LSTM-AE | 시계열 입력의 Autoencoder | 시간 패턴 이상 탐지에 최적 |

```python
# Variational Autoencoder (VAE) 핵심 구조
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 128)
        self.mu = nn.Linear(128, latent_dim)      # 평균
        self.log_var = nn.Linear(128, latent_dim)  # 로그 분산

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std  # 미분 가능한 샘플링

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        mu, log_var = self.mu(h), self.log_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

# VAE 손실: 재구성 오차 + KL 발산
# Loss = MSE(x, x̂) + KL(q(z|x) || p(z))
```

### Deep SVDD (Support Vector Data Description)

:::{admonition} 참고 교재
:class: note
📖 *Hands-On Unsupervised Learning* **Ch.7** — One-Class 분류와 SVDD 원리
:::

Autoencoder가 "재구성 오차"로 이상을 판단한다면,  
Deep SVDD는 **"정상 데이터를 하나의 구에 가둔다"** 는 아이디어입니다.

```{mermaid}
flowchart LR
    subgraph "학습 전"
        A1["● ● ●"] 
    end
    subgraph "학습 후"
        B1["┌───────┐"]
        B2["│ ● ● ● │  c ← 중심"]
        B3["└───────┘"]
    end
    A1 -->|"학습"| B1
```

```
이상 탐지:  중심 c와의 거리 > 반지름 R → 이상
```

```python
# Deep SVDD 핵심 손실 함수
def svdd_loss(z, center, radius, nu=0.1):
    """
    z: 잠재 표현
    center: 구의 중심 (학습 전 초기화 후 고정)
    nu: 이상값 허용 비율
    """
    dist = torch.sum((z - center) ** 2, dim=1)
    loss = radius ** 2 + (1 / nu) * torch.mean(
        torch.clamp(dist - radius ** 2, min=0)
    )
    return loss
```

---

## 📘 섹션 2 심화: RNN 계열 모델 비교와 Keras 심화

:::{admonition} 참고 교재
:class: note
📖 **교재**: *Deep Learning with Python, 2nd Ed.* (François Chollet, Manning)  
**심화 챕터 로드맵**:
- **Ch.5 Fundamentals of Machine Learning** — 과적합 원인, 정규화 전략 (L1/L2, Dropout, Batch Norm)
- **Ch.7 Working with Keras: A Deep Dive** — Functional API, 커스텀 레이어, 콜백
- **Ch.10 Deep Learning for Timeseries** — RNN, LSTM, GRU, Conv1D 비교, 양방향 LSTM
:::

### RNN 계열 모델 비교

| 모델 | 파라미터 수 | 학습 속도 | 장기 의존성 | RUL 예측 적합도 |
|------|-----------|---------|-----------|--------------|
| Simple RNN | 적음 | 빠름 | 약함 | 짧은 시퀀스만 |
| LSTM | 많음 | 느림 | 강함 | 표준 선택 |
| GRU | 중간 | 중간 | 강함 | LSTM 대안 (더 단순) |
| Bidirectional LSTM | 2배 | 느림 | 양방향 | 배치 예측에 효과적 |
| Conv1D + LSTM | 중간 | 빠름 | 강함 | 지역+전역 패턴 동시 |

```python
# GRU: LSTM보다 파라미터가 적고 성능은 유사
gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(30, 14)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Conv1D + LSTM: 지역 패턴(Conv1D) + 전역 패턴(LSTM) 조합
hybrid_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu',
                            input_shape=(30, 14)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])
```

### Keras Functional API: 복잡한 모델 설계

```python
# Sequential API의 한계: 단일 입력/출력만 가능
# Functional API: 다중 입력, 분기, 잔차 연결 가능

inputs = tf.keras.Input(shape=(30, 14))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.LSTM(32)(x)

# 다중 출력: RUL 예측 + 이상 점수 동시 출력
rul_output = tf.keras.layers.Dense(1, name='rul')(x)
anomaly_output = tf.keras.layers.Dense(1, activation='sigmoid',
                                        name='anomaly')(x)

model = tf.keras.Model(inputs=inputs,
                       outputs=[rul_output, anomaly_output])
model.compile(
    optimizer='adam',
    loss={'rul': 'mse', 'anomaly': 'binary_crossentropy'},
    loss_weights={'rul': 1.0, 'anomaly': 0.5}
)
```

### 불확실성 정량화: 예측에 신뢰 구간 붙이기

```python
# MC Dropout: 추론 시에도 Dropout을 켜서 여러 번 예측 → 분포 추정
def predict_with_uncertainty(model, X, n_iter=100):
    predictions = np.array([
        model(X, training=True).numpy()  # training=True: Dropout 활성화
        for _ in range(n_iter)
    ])
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    return mean_pred, std_pred

mean, std = predict_with_uncertainty(model, X_test)
# RUL 예측값 ± 2σ 범위로 신뢰 구간 시각화 가능
```

---

## 📘 섹션 3 심화: 시계열 이상탐지 고급 기법

:::{admonition} 참고 교재
:class: note
📖 **교재**: *Machine Learning for Time-Series with Python* (Ben Auffarth, Packt)  
**심화 챕터 로드맵**:
- **Ch.2 Exploratory Data Analysis** — 시계열 EDA, 정상성 검정, 자기상관 분석
- **Ch.3 Preprocessing Time-Series Data** — 정규화, 결측치, 슬라이딩 윈도우 설계
- **Ch.4 Feature Engineering** — 시계열 특징 추출, 비대칭 손실 함수 설계
- **Ch.7 Machine Learning for Time-Series Classification** — LSTM 분류/회귀 심화
- **Ch.11 Probabilistic Models** — 베이지안 구조변화 탐지, 온라인 이상탐지
:::

### 도메인 적응 (Domain Adaptation): FD001 → FD002

```python
# 문제: FD001(단일 조건)에서 학습 → FD002(6가지 조건) 적용 시 성능 저하
# 원인: 운전 조건(op1, op2, op3)에 따라 센서 범위가 달라짐

# 해결책 1: 운전 조건별 정규화
for op_condition in df['op_condition'].unique():
    mask = df['op_condition'] == op_condition
    df.loc[mask, sensor_cols] = scaler.fit_transform(
        df.loc[mask, sensor_cols]
    )

# 해결책 2: 운전 조건을 모델 입력에 포함
# Functional API로 센서 입력 + 운전조건 입력을 결합
sensor_input = tf.keras.Input(shape=(30, 14))
op_input = tf.keras.Input(shape=(3,))  # 운전조건 3개

lstm_out = tf.keras.layers.LSTM(64)(sensor_input)
combined = tf.keras.layers.Concatenate()([lstm_out, op_input])
output = tf.keras.layers.Dense(1)(combined)
```

### 온라인/스트리밍 RUL 예측

```{mermaid}
flowchart LR
    A["새 사이클\n센서값"] --> B["윈도우 업데이트"]
    B --> C["LSTM 예측\nRUL 산출"]
    C --> D{"RUL < 30?"}
    D -->|"Yes"| E["경고 알림\n잔여 수명 n 사이클"]
    D -->|"No"| F["정상 운영\n다음 사이클 대기"]
```

```python
# 실시간 예측: 새 사이클이 들어올 때마다 윈도우를 업데이트하며 예측
from collections import deque

class OnlineRULPredictor:
    def __init__(self, model, window_size=30, n_sensors=14):
        self.model = model
        self.window = deque(maxlen=window_size)
        self.window_size = window_size

    def update(self, new_sensor_values):
        """새로운 사이클의 센서값 추가"""
        self.window.append(new_sensor_values)

    def predict(self):
        """현재 윈도우로 RUL 예측"""
        if len(self.window) < self.window_size:
            return None  # 윈도우가 아직 채워지지 않음
        X = np.array(self.window)[np.newaxis, ...]
        return self.model.predict(X)[0, 0]

# 사용 예시
predictor = OnlineRULPredictor(model)
for new_data in streaming_sensor_data:
    predictor.update(new_data)
    rul = predictor.predict()
    if rul is not None and rul < 30:
        send_alert(f"경고: 잔여 수명 {rul:.1f} 사이클")
```

---

## 📚 추천 학습 경로

### 비지도 이상탐지 (1~2주) — *Hands-On Unsupervised Learning Using Python*

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.1** | 비지도 학습 전체 지형도 | 어떤 방법이 언제 적합한지 파악 |
| 2 | **Ch.3** | Autoencoder 구조와 변형 | 강의 내용 깊이 이해 |
| 3 | **Ch.7** | 이상탐지 응용, Deep SVDD | One-Class 분류 심화 |

### 딥러닝 시계열 (2주) — *Deep Learning with Python, 2nd Ed.*

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.1** | 딥러닝 개요, 표현 학습 | 전체 맥락 파악 |
| 2 | **Ch.5** | 과적합 원인과 정규화 전략 | Dropout, L2, Early Stopping 원리 |
| 3 | **Ch.10** | RNN, LSTM, GRU, Conv1D | 시계열 모델 선택 기준 |
| 4 | **Ch.7** | Keras Functional API | 복잡한 모델 설계 능력 |

### 통합 실습 심화 (2주) — *ML for Time-Series with Python*

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.2** | 시계열 EDA | 데이터 탐색 능력 |
| 2 | **Ch.3** | 전처리, 슬라이딩 윈도우 | 파이프라인 설계 |
| 3 | **Ch.4** | 특징 공학, 비대칭 손실 | 실무 평가 지표 설계 |
| 4 | **Ch.7** | LSTM 분류/회귀 심화 | 강의 실습 확장 |
| 5 | **Ch.11** | 확률적 모델, 온라인 탐지 | 실시간 시스템 구현 |

### 실무 적용 프로젝트 아이디어

- [ ] **CWRU Bearing Dataset**: Autoencoder로 베어링 결함 탐지 (섹션 1 심화)
- [ ] **NASA Turbofan FD002~FD004**: 운전 조건 변화 시 도메인 적응 실험 (섹션 3 심화)
- [ ] **PHM08 Challenge Dataset**: LSTM-Autoencoder 통합 파이프라인으로 RUL 예측 대회 재현
- [ ] **실시간 시뮬레이터**: Streamlit으로 온라인 RUL 예측 대시보드 구현

---

*이 문서는 세션 2 강의 자료로 제작되었습니다.*  
*질문 및 피드백: Claude Code를 활용해 실습 중 막히는 부분을 언제든지 질문하세요.*  
*세션 1 자료와 함께 참고하세요: `session1_manufacturing_data_analysis.md`*

# 출처
## 섹션 1. 텍스트북 : [Hands-On Unsupervised Learning Using Python](https://www.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/)
## 섹션 2. 텍스트북 : [Deep Learning with Python (2nd Ed.)](https://www.oreilly.com/library/view/deep-learning-with/9781617296422/)
## 섹션 3. 텍스트북 : [Machine Learning for Time-Series with Python](https://www.oreilly.com/library/view/machine-learning-for/9781801819626/)

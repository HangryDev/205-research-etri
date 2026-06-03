# 섹션 3 | NASA Turbofan 통합 실습

```{admonition} 참고 교재
:class: note

**참고 교재**: *Machine Learning for Time-Series with Python* (Ben Auffarth, Packt)  
```

---

## 3-1. 데이터 확인 + RUL 라벨 설계

### NASA Turbofan Engine Degradation 데이터셋 소개

```{admonition} 참고 교재
:class: note

📖 *ML for Time-Series with Python* **Ch.2 Exploratory Data Analysis** — 시계열 데이터 탐색, 분포 분석, 센서 상관관계 파악
```

```
[데이터셋 구성]

FD001: 단일 운전 조건, 단일 고장 모드  ← 이번 실습 주 데이터
FD002: 6가지 운전 조건, 단일 고장 모드
FD003: 단일 운전 조건, 2가지 고장 모드
FD004: 6가지 운전 조건, 2가지 고장 모드

각 파일 구조:
  열: unit번호 | 사이클 | 운전조건 3개 | 센서값 21개
  행: 각 유닛의 사이클별 측정값
  train: 고장 시점까지의 전체 이력
  test: 고장 전 일부 구간 (RUL을 맞춰야 함)
```

**Claude로 EDA 요청**:
```
NASA Turbofan FD001 train 데이터를 탐색해줘.
1. 기본 정보: shape, 결측치, 유닛 수, 사이클 범위
2. 센서별 시계열 플롯 (유닛 1개 선택, 센서 21개를 4×6 subplot)
3. 변동이 거의 없는 센서 식별 (std < 0.01인 센서)
4. 유닛별 수명(max cycle) 분포 히스토그램
```

### RUL 클리핑: 왜 125로 자르는가

```{admonition} 참고 교재
:class: note

📖 *ML for Time-Series with Python* **Ch.3 Preprocessing Time-Series Data** — RUL 라벨 설계, 클리핑 전략, 비선형 RUL 정의
```

```
[클리핑이 필요한 이유]

실제 RUL 분포 (클리핑 전):
엔진 초반 구간에서 RUL = 250, 300, 350...
이 구간의 센서 데이터는 사실상 모두 "정상"으로 동일
→ 모델이 초반 구간을 구분하는 데 불필요한 에너지 낭비

클리핑 후 (max=125):
RUL > 125인 구간은 모두 125로 고정
→ "남은 수명이 125 이하인 구간"만 정밀하게 학습
→ 실무적으로 의미 있는 예측 범위에 집중
```

```python
# RUL 라벨 생성 + 클리핑
max_cycle = train_df.groupby('unit')['cycle'].max().reset_index()
max_cycle.columns = ['unit', 'max_cycle']
train_df = train_df.merge(max_cycle, on='unit')
train_df['RUL'] = (train_df['max_cycle'] - train_df['cycle']).clip(upper=125)

# Claude로 시각화 요청
# "RUL 클리핑 전후 분포를 나란히 비교하는 히스토그램 그려줘"
```

---

## 3-2. 모델링 + 평가

### LSTM 모델 적용

```{admonition} 참고 교재
:class: note

📖 *ML for Time-Series with Python* **Ch.7 Machine Learning for Time-Series Classification** — LSTM 기반 회귀, 슬라이딩 윈도우 입력 구성
```

섹션 2에서 만든 LSTM 구조를 Turbofan 데이터에 그대로 적용합니다.

```python
# 섹션 2 실습 결과 중 가장 좋은 조합 사용 (예: 2층 LSTM + Dropout 0.2)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(30, 14), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)
```

### 평가 지표: RMSE만으로는 부족한 이유

```{admonition} 참고 교재
:class: note

📖 *ML for Time-Series with Python* **Ch.4 Feature Engineering** — 평가 지표 설계, 비대칭 손실 함수
```

```
[RMSE의 한계]

RMSE = √(Σ(실제 RUL - 예측 RUL)² / N)
→ 조기 예측 오차와 지연 예측 오차를 동일하게 취급

실무에서의 문제:
  실제 RUL=10, 예측 RUL=20 (조기 경보: 10 사이클 일찍 교체)
  → 비용: 교체 비용만 발생

  실제 RUL=10, 예측 RUL=0  (지연 경보: 10 사이클 늦게 알림)
  → 비용: 설비 고장 + 생산 중단 + 안전사고
```

**NASA Scoring Function** (비대칭 평가):

```python
import numpy as np

def nasa_score(y_true, y_pred):
    """
    조기 예측(예측 > 실제): 덜 엄격한 페널티
    지연 예측(예측 < 실제): 더 엄격한 페널티
    """
    d = y_pred - y_true  # 양수: 조기 예측, 음수: 지연 예측
    score = np.where(
        d < 0,
        np.exp(-d / 13) - 1,   # 지연 예측: 급격히 증가
        np.exp(d / 10) - 1     # 조기 예측: 완만하게 증가
    )
    return np.sum(score)

y_pred = model.predict(X_val).flatten()
rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
score = nasa_score(y_val, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"NASA Score: {score:.2f}  (낮을수록 좋음)")
```

**Claude에게 추가 질문**:
> *"RMSE가 같아도 NASA Score가 크게 다를 수 있는 시나리오를 예시로 보여줘"*

---

## 3-3. 자유 실습

본인의 수준에 맞는 과제를 선택하세요.

### ★ 기본 — FD001 단일 학습 및 평가
```
목표: 전체 파이프라인을 처음부터 끝까지 직접 돌린다

1. FD001 데이터 로드 + RUL 라벨 생성 (클리핑 125)
2. 센서 14개 선택 + 슬라이딩 윈도우(30) 시퀀스 생성
3. LSTM 학습 (Early Stopping)
4. RMSE + NASA Score 계산
5. 실제 vs 예측 RUL 산점도 출력

제출: RMSE와 NASA Score 숫자 + 산점도 이미지
```

### ★★ 중급 — FD001 vs FD002 비교 (운전 조건 일반화)
```
목표: 운전 조건이 달라지면 모델 성능이 얼마나 달라지는지 확인

1. FD001로 학습한 모델을 FD002 데이터에 바로 적용
   → RMSE, NASA Score 기록
2. FD002로 재학습한 모델과 성능 비교
   → "운전 조건이 다른 데이터에 바로 적용하면 어떻게 되는가?"
3. Claude에게 질문:
   "FD001과 FD002의 센서 분포가 다른 이유가 뭐야?
    도메인 적응(Domain Adaptation)은 어떤 방식으로 해결해?"

제출: FD001 학습 → FD001 평가 vs FD002 평가 비교 표
```

### ★★★ 심화 — Autoencoder + LSTM 통합 파이프라인
```
목표: 섹션 1(이상탐지)과 섹션 2(RUL 예측)을 하나의 파이프라인으로 연결
```

```{mermaid}
flowchart TB
    A["센서 데이터"] --> B["Autoencoder → 재구성 오차 계산"]
    B --> C{"재구성 오차 > 임계값?"}
    C -->|"Yes"| D["이상 경보 발생"]
    C -->|"No"| E["LSTM → RUL 예측"]
    E --> F{"RUL < 30?"}
    F -->|"Yes"| G["교체 권고"]
    F -->|"No"| H["정상 운영"]
```

```
구현 순서:
  1. 섹션 1 Autoencoder로 각 윈도우의 이상 점수 계산
  2. 이상 점수를 LSTM 입력의 추가 특징으로 연결
  3. 통합 파이프라인 실행 → 이상 경보 + RUL 예측 동시 출력

제출: 파이프라인 흐름도 + 주요 구간에서의 이상 점수 + RUL 예측값
```

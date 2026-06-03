# 세션 1 실습 — Claude Code 시연 & 실습 모음


> 세션 1(제조 데이터 심층 분석)의 각 강의 섹션에 흩어져 있던 **Claude Code 시연**과 **실습** 부분을 한곳에 모았습니다.
> 각 항목의 이론·배경 설명은 해당 강의 섹션을 참고하세요.


---

```{admonition} 출처: 섹션 1 · 노이즈 제거
:class: note dropdown
아래 시연/실습은 원래 **섹션 1 · 노이즈 제거** 섹션에 있던 내용입니다.
```

## 1-3. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: 코드 자체보다 **Claude에게 어떻게 질문하는가**를 주목하세요.
```

**Claude에게 던질 프롬프트 예시**:
```
CNC 밀링머신 진동 데이터에서 채터링 노이즈를 제거하려고 해.
pywt 라이브러리를 사용해서:
1. db4 wavelet으로 5레벨 분해
2. 상위 2개 레벨(고주파)에 soft threshold 적용
3. 재합성 후 원본과 결과를 subplot으로 비교
샘플 데이터는 numpy로 직접 생성해줘.
```

**시연 흐름**:
1. Raw signal 생성 (절삭 신호 + 합성 노이즈)
2. `wavedec` 로 분해 → 각 레벨 계수 시각화
3. Threshold 적용 → 계수 변화 확인
4. `waverec` 로 재합성 → before/after 비교
5. **Claude에게 추가 질문**: *"db4 대신 sym5를 쓰면 어떤 차이가 있어?"*

---

## 1-4. 실습

### 과제

아래 파라미터 조합을 바꿔보고 결과를 비교하세요.

| 조합 | Wavelet | Level | 관찰 포인트 |
|------|---------|-------|------------|
| A | `db4` | 3 | 기준선 |
| B | `db4` | 5 | 레벨이 늘면? |
| C | `sym5` | 3 | wavelet 종류 변경 시? |
| D | `sym5` | 5 | 둘 다 변경 시? |

**제출 항목**:
- 4가지 조합의 before/after 그래프
- "어떤 조합이 채터링 제거에 가장 효과적이었는가?" 한 줄 답변 + 이유

### 실습 시작 코드

```python
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 합성 신호 생성
np.random.seed(42)
t = np.linspace(0, 1, 1024)
cutting_signal = np.sin(2 * np.pi * 50 * t)          # 50Hz: 절삭 신호
spindle_noise  = 0.3 * np.sin(2 * np.pi * 200 * t)   # 200Hz: 스핀들 노이즈
chatter        = 0.2 * np.random.randn(len(t))         # 광대역 채터링
signal = cutting_signal + spindle_noise + chatter

def denoise_wavelet(signal, wavelet='db4', level=5, threshold_mode='soft'):
    # TODO: 여기를 채워보세요
    # 1. pywt.wavedec 로 분해
    # 2. 고주파 계수에 threshold 적용
    # 3. pywt.waverec 로 재합성
    pass

# 결과 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(t, signal, label='원본 신호', alpha=0.7)
# TODO: 노이즈 제거 결과 추가
plt.tight_layout()
plt.show()
```

---

```{admonition} 출처: 섹션 2 · 데이터 불균형
:class: note dropdown
아래 시연/실습은 원래 **섹션 2 · 데이터 불균형** 섹션에 있던 내용입니다.
```

## 2-3. Claude Code 시연

**Claude에게 던질 프롬프트 예시**:
```
불량률 1%인 제조 데이터를 시뮬레이션해줘.
1. 기본 RandomForest 모델 학습
2. SMOTE 적용 후 재학습
3. class_weight='balanced' 적용 후 재학습
세 모델의 Confusion Matrix와 F1-score를 나란히 비교하는 시각화를 만들어줘.
```

**시연 후 Claude에게 추가 질문**:
> *"SMOTE가 항상 최선의 방법인가? 어떤 상황에서는 안 쓰는 게 나을까?"*

---

## 2-4. 실습

### 과제

SMOTE의 `k_neighbors` 파라미터가 결과에 어떤 영향을 미치는지 분석하세요.

| 실험 | k_neighbors | 관찰 지표 |
|------|-------------|----------|
| 기본 | 5 (기본값) | F1-Score, Recall |
| 실험 A | 3 | 위와 동일 |
| 실험 B | 7 | 위와 동일 |
| 실험 C | 10 | 위와 동일 |

**제출 항목**:
- k_neighbors별 F1-Score와 Recall 비교 표 또는 그래프
- "k_neighbors가 커질수록 어떤 경향이 있었는가?" 한 문단 분석

### 실습 시작 코드

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE

# 불균형 데이터 생성 (불량 1%)
X, y = make_classification(
    n_samples=10000,
    weights=[0.99, 0.01],  # 정상 99%, 불량 1%
    n_features=10,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for k in [3, 5, 7, 10]:
    # TODO: 각 k_neighbors로 SMOTE 적용 후 모델 학습 및 평가
    smote = SMOTE(k_neighbors=k, random_state=42)
    # ...
    pass

# 결과 비교 시각화
# TODO
```

---

```{admonition} 출처: 섹션 3 · CNC 이상탐지 파이프라인
:class: note dropdown
아래 시연/실습은 원래 **섹션 3 · CNC 이상탐지 파이프라인** 섹션에 있던 내용입니다.
```

## 3-3. Claude Code 시연

**Claude에게 던질 프롬프트 예시**:
```
CNC 밀링머신의 공구 마모 탐지 파이프라인을 만들어줘.
- 시뮬레이션 데이터: 정상 구간 + 마모 시작 구간 포함
- 단계: Wavelet 노이즈 제거 → 슬라이딩 윈도우 특징 추출 → Isolation Forest
- 각 단계의 결과를 subplot으로 시각화 (원본, 노이즈 제거 후, 이상 점수 시계열)
- contamination=0.05로 설정
```

**시연 후 Claude에게 추가 질문**:
> *"탐지 민감도를 높이려면 어떤 파라미터를 조정해야 하고, 그때 트레이드오프는 뭐야?"*

**시연 흐름**:
1. 합성 진동 데이터 생성 (정상 → 마모 시작 → 심각한 마모)
2. Wavelet 노이즈 제거 적용
3. 슬라이딩 윈도우로 특징 벡터 계산
4. Isolation Forest로 이상 점수 계산
5. 이상 점수 시계열 플롯 → 마모 시작 구간이 탐지되는지 확인

---

## 3-4. 실습

### 과제

`contamination` 파라미터를 조정하며 탐지 민감도의 트레이드오프를 분석하세요.

| 실험 | contamination | 예상 결과 |
|------|--------------|----------|
| A | 0.01 (1%) | 매우 보수적 탐지 |
| B | 0.05 (5%) | 기본값 |
| C | 0.10 (10%) | 민감한 탐지 |

**제출 항목**:
- 3가지 설정의 이상 점수 시계열 그래프 (subplot)
- "너무 민감한 것과 너무 둔감한 것의 트레이드오프를 실무 관점에서 설명하라" (한 문단)

### 실습 시작 코드

```python
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 시뮬레이션 데이터 생성
np.random.seed(42)
t = np.linspace(0, 10, 10000)  # 10초, 1000Hz 샘플링

# 정상 구간 (0~5초) + 마모 시작 (5~7초) + 심각한 마모 (7~10초)
signal = np.sin(2 * np.pi * 50 * t)                          # 기본 절삭 신호
signal[5000:7000] += 0.3 * np.random.randn(2000)              # 마모 시작: 노이즈 증가
signal[7000:] += 0.8 * np.random.randn(3000)                  # 심각한 마모: 큰 노이즈

# Step 1: Wavelet 노이즈 제거
def denoise(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * np.std(coeffs[-1])
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

denoised = denoise(signal)

# Step 2: 슬라이딩 윈도우 특징 추출
def extract_features_windowed(signal, window_size=100, step=50):
    features = []
    for i in range(0, len(signal) - window_size, step):
        window = signal[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        peak = np.max(np.abs(window))
        std = np.std(window)
        kurt = np.mean(((window - np.mean(window)) / (std + 1e-8))**4)
        features.append([rms, peak, kurt])
    return np.array(features)

features = extract_features_windowed(denoised)

# Step 3: Isolation Forest (contamination 파라미터를 바꿔가며 실험)
for contamination in [0.01, 0.05, 0.10]:
    # TODO: Isolation Forest 학습 및 이상 점수 계산
    # TODO: 결과 시각화
    pass
```

# 섹션 3 | CNC 이상탐지 파이프라인 — 공구 마모 시작점 잡기

```{admonition} 참고 교재
:class: note

**참고 교재**: *Machine Learning for Time-Series with Python* (Ben Auffarth)  
```

---

## 3-1. 문제 제기 + 데이터 확인

**목표**: 공구 마모가 **시작되는 순간**을 실시간으로 포착할 수 있을까?

```{mermaid}
flowchart LR
    A["정상 가공\n진동 안정적\nRMS 낮음"] --> B["초기 마모 시작\n진동 약간 증가\nRMS 서서히 상승"]
    B --> C["심각한 마모\n진동 급격히 증가\nRMS 급등, 불규칙"]
```

**이 순간을 잡지 못하면**:
- 마모된 공구로 계속 가공 → 제품 불량
- 공구 교체 시점을 놓침 → 더 심한 손상, 비용 증가

**섹션 1에서 정제된 데이터 재확인**:
- 노이즈 제거 전: 공구 마모 신호가 다른 노이즈에 묻혀 있음
- 노이즈 제거 후: 절삭 신호의 변화 패턴이 명확하게 드러남

---

## 3-2. 이론: 파이프라인 설계 논리

### ① 왜 Raw Signal을 그대로 쓰면 안 되는가: 특징 추출

```{admonition} 참고 교재
:class: note

📖 *ML for Time-Series with Python* **Ch.3 Preprocessing Time-Series Data** — 시계열 데이터 전처리와 특징 공학 / **Ch.6 Unsupervised Methods for Anomaly Detection** — 비지도 이상탐지를 위한 특징 설계
```

진동 신호를 그대로 ML 모델에 넣으면 두 가지 문제가 생깁니다:
1. **차원의 저주**: 1초에 10,000개 샘플 × 10시간 = 3억 6천만 개의 입력값
2. **패턴 모호성**: 모델이 노이즈와 신호를 구분하기 어려움

**해결책**: 물리적으로 의미 있는 특징을 추출합니다.

```
[핵심 특징 3가지]

1. RMS (Root Mean Square) — 신호의 에너지
   RMS = √(Σx² / N)
   → 마모가 진행될수록 RMS 상승

2. Peak Value — 순간 최대 진폭
   → 채터링, 충격 발생 시 급등

3. Kurtosis — 신호의 첨도 (뾰족한 정도)
   Kurtosis = E[(X-μ)⁴] / σ⁴
   → 정상 신호: ~3 (정규분포)
   → 결함 발생: 3 이상으로 상승 (충격성 이벤트 증가)
```

```python
import numpy as np

def extract_features(signal_window):
    """진동 신호 윈도우에서 특징 추출"""
    rms = np.sqrt(np.mean(signal_window**2))
    peak = np.max(np.abs(signal_window))
    mean = np.mean(signal_window)
    std = np.std(signal_window)
    kurtosis = np.mean(((signal_window - mean) / std)**4) if std > 0 else 0
    return {'rms': rms, 'peak': peak, 'kurtosis': kurtosis}
```

---

### ② 이상탐지 방식 선택: 레이블 유무에 따른 전략

```{admonition} 참고 교재
:class: note

📖 *ML for Time-Series with Python* **Ch.6 Unsupervised Methods for Anomaly Detection** — Isolation Forest, LOF 등 비지도 탐지 / **Ch.7 Machine Learning for Time-Series Classification** — 레이블 있는 경우의 지도학습 분류
```

```{mermaid}
flowchart TD
    A{"레이블 유무?"} -->|"레이블 있음\n(정상/불량 알고 있음)"| B["지도학습 분류기\nRandomForest, SVM\n섹션 2의 불균형 처리 기법 활용"]
    A -->|"레이블 없음\n(정상만 알고 있음)"| C["비지도 이상탐지\nIsolation Forest, One-Class SVM\n실제 현장에서 더 흔한 상황"]
```

**Isolation Forest 원리**:

```
핵심 아이디어: 이상값은 "고립시키기 쉽다"

랜덤하게 데이터를 분할할 때:
- 정상 데이터: 많은 분할이 필요 (데이터가 밀집)
- 이상 데이터: 적은 분할로 고립됨 (데이터가 희박)

→ 고립에 필요한 평균 분할 수 = 이상 점수
→ 분할 수가 적을수록 이상 점수가 높음
```

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination=0.05,  # 전체 데이터 중 이상값 예상 비율 (조정 필요)
    random_state=42
)
model.fit(X_normal)  # 정상 데이터로만 학습
predictions = model.predict(X_test)  # 1: 정상, -1: 이상
```

---

### ③ 전체 파이프라인 설계

```{admonition} 참고 교재
:class: note

📖 *ML for Time-Series with Python* **Ch.3 Preprocessing** — 슬라이딩 윈도우와 특징 추출 파이프라인 구성 / **Ch.6 Anomaly Detection** — 엔드투엔드 이상탐지 시스템 구조
```

```{mermaid}
flowchart TB
    A["1. 데이터 수집\n센서 데이터 (진동, 소리, 전류) 실시간 스트림"] --> B["2. 전처리 (섹션 1)\nWavelet Transform → 노이즈 제거"]
    B --> C["3. 윈도우 분할\n슬라이딩 윈도우 (예: 1초 단위, 0.5초 겹침)"]
    C --> D["4. 특징 추출\n각 윈도우 → RMS, Peak, Kurtosis 계산"]
    D --> E["5. 이상탐지\nIsolation Forest → 이상 점수 계산"]
    E --> F["6. 알림\n이상 점수 > 임계값 → 경보 발생"]
```

---

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

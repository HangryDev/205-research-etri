# 심화 학습 자료

```{admonition} 심화 학습 자료
:class: note

강의에서 시간 관계상 다루지 못한 심화 내용입니다. 강의 후 궁금증이 생겼을 때 아래 순서로 탐색하시길 권장합니다.
```

---

## 📘 섹션 1 심화: 신호 처리

### Wavelet 종류별 특성

| Wavelet 계열 | 특성 | 적합한 용도 |
|-------------|------|------------|
| Haar | 가장 단순, 불연속 | 단계 함수 신호, 빠른 계산 |
| Daubechies (db) | 비대칭, 지지 소형 | 일반적 제조 진동 신호 |
| Symlets (sym) | db보다 대칭에 가까움 | 위상 왜곡이 민감한 경우 |
| Coiflets (coif) | 높은 소멸 모멘트 | 부드러운 신호 |
| Morlet | 연속 Wavelet, 복소수 | 주파수 분석 중심 |

### 최적 Decomposition Level 결정

```python
# 경험적 규칙: level = log2(N) 이하
# N: 신호 길이
import numpy as np

N = 1024
max_level = int(np.floor(np.log2(N)))
print(f"권장 최대 레벨: {max_level}")  # 10

# 실무에서는 3~6 레벨이 가장 흔히 사용됨
# 레벨이 너무 높으면: 계수 수가 너무 적어 재합성 오차 증가
# 레벨이 너무 낮으면: 주파수 분리 불충분
```

### Wavelet Packet Transform (WPT)

기본 DWT는 저주파(Approximation)만 계속 분해합니다.  
WPT는 고주파(Detail)도 재귀적으로 분해하여 더 세밀한 주파수 분해가 가능합니다.

```python
# WPT 예시
wp = pywt.WaveletPacket(data=signal, wavelet='db4', mode='symmetric')
# 전체 트리에서 원하는 노드 선택 가능
nodes = wp.get_level(4, 'natural')
```

**참고**: *Think DSP* **Ch.8 Filtering and Convolution** — 필터 구현 실습 / **Ch.9 Differentiation and Integration** — 차분 방정식을 이용한 재귀적 필터

---

### Short-Time Fourier Transform (STFT): Wavelet과의 비교

```
STFT                          Wavelet Transform
─────────────────────         ─────────────────────
고정된 윈도우 크기            가변 윈도우 크기
주파수마다 동일한 시간 해상도  저주파: 긴 윈도우, 고주파: 짧은 윈도우
구현 단순                     구현 복잡
Heisenberg 불확정성 제약 큼   제약 상대적으로 적음
```

```python
from scipy.signal import stft
import matplotlib.pyplot as plt

f, t_stft, Zxx = stft(signal, fs=1000, nperseg=64)
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Spectrogram')
plt.colorbar()
```

---

## 📘 섹션 2 심화: 불균형 데이터

### SMOTE 변형 알고리즘 비교

| 알고리즘 | 특징 | 주의사항 |
|---------|------|---------|
| SMOTE | 기본, k-NN 기반 보간 | 경계 근처 노이즈 취약 |
| ADASYN | 학습하기 어려운 샘플 주변을 더 많이 생성 | 극도의 불균형에 불안정 |
| Borderline-SMOTE | 경계 근처 샘플만 합성 | 위험 구간 집중 학습 |
| SMOTE-ENN | SMOTE 후 노이즈 제거 | 더 깨끗하지만 느림 |
| SMOTE-Tomek | SMOTE + Tomek link 제거 | 경계 선명화 |

```python
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

# 각 방법 비교
methods = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42),
}
```

### 앙상블 기반 불균형 처리

**BalancedRandomForest**: 각 트리를 학습할 때 소수 클래스를 오버샘플링

```python
from imblearn.ensemble import BalancedRandomForestClassifier

model = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='auto',
    random_state=42
)
```

**EasyEnsemble**: 다수 클래스를 여러 번 언더샘플링해서 여러 모델 학습

```python
from imblearn.ensemble import EasyEnsembleClassifier

model = EasyEnsembleClassifier(n_estimators=10, random_state=42)
```

### Model Calibration: 확률값 보정

불균형 데이터로 학습한 모델은 확률값 자체가 왜곡될 수 있습니다.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Platt Scaling (sigmoid) 또는 Isotonic Regression으로 보정
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# Calibration Curve (신뢰도 다이어그램)
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, calibrated_model.predict_proba(X_test)[:, 1], n_bins=10
)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="보정 후")
plt.plot([0, 1], [0, 1], "k:", label="완벽한 보정")
```

**참고**: *ML for Imbalanced Data* **Ch.10 Threshold Adjustment and Model Calibration** — Platt Scaling, Isotonic Regression 구현 및 calibration curve 해석

---

## 📘 섹션 3 심화: 이상탐지

### 이상탐지 알고리즘 비교

| 알고리즘 | 방식 | 장점 | 단점 |
|---------|------|------|------|
| Isolation Forest | 트리 기반 분리 | 빠름, 고차원 적합 | contamination 설정 필요 |
| One-Class SVM | 경계 학습 | 이론적 근거 확실 | 대용량에 느림, 파라미터 민감 |
| LOF | 밀도 기반 | 지역적 이상 탐지 | n_neighbors 설정 필요 |
| AutoEncoder | 딥러닝 기반 | 복잡한 패턴 학습 | 데이터 많이 필요, 학습 비용 |
| LSTMD | 시계열 특화 딥러닝 | 시간 패턴 학습 | 구현 복잡, 해석 어려움 |

### 시계열 특화 특징 추출

강의에서 다룬 RMS, Peak, Kurtosis 외에 실무에서 자주 쓰이는 특징들:

```python
import numpy as np
from scipy import stats

def extract_full_features(window):
    """확장된 특징 추출"""
    return {
        # 시간 영역
        'rms': np.sqrt(np.mean(window**2)),
        'peak': np.max(np.abs(window)),
        'crest_factor': np.max(np.abs(window)) / (np.sqrt(np.mean(window**2)) + 1e-8),
        'kurtosis': stats.kurtosis(window),
        'skewness': stats.skew(window),

        # 주파수 영역 (FFT 기반)
        'spectral_centroid': np.sum(
            np.abs(np.fft.rfft(window)) * np.fft.rfftfreq(len(window))
        ) / (np.sum(np.abs(np.fft.rfft(window))) + 1e-8),

        # 엔트로피 (신호의 복잡도)
        'sample_entropy': _sample_entropy(window),
    }

def _sample_entropy(signal, m=2, r_ratio=0.2):
    """Sample Entropy: 신호의 불규칙성 측정"""
    r = r_ratio * np.std(signal)
    N = len(signal)
    # ... (구현 생략, scipy.stats 또는 nolds 라이브러리 활용 권장)
```

### 온라인/실시간 이상탐지

배치 처리가 아닌 스트리밍 데이터에서의 이상탐지:

```python
# Half-Space Trees: 실시간 이상탐지에 적합
# pip install river
from river import anomaly

model = anomaly.HalfSpaceTrees(
    n_trees=25,
    height=15,
    window_size=250,
)

# 스트리밍 데이터 처리
for x in data_stream:
    score = model.score_one(x)
    model.learn_one(x)
    if score > threshold:
        trigger_alert()
```

**참고**: *ML for Time-Series with Python* **Ch.6 Unsupervised Methods for Anomaly Detection** — 스트리밍 환경 이상탐지 전략 / **Ch.11 Probabilistic Models** — 온라인 변화점 탐지

---

## 📚 추천 학습 경로

### 기초 확립 (1~2주) — *Think DSP*
| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.1** Sounds and Signals | 신호, 주파수, 진폭의 정의 | 모든 내용의 언어를 배우는 단계 |
| 2 | **Ch.3** Non-periodic Signals | FFT의 한계, 비정상 신호 | 왜 Wavelet이 필요한지 이해 |
| 3 | **Ch.4** Noise | 노이즈의 스펙트럼과 종류 | 제거 대상을 정의하는 단계 |
| 4 | **Ch.8** Filtering and Convolution | 필터 원리, 저역/고역 통과 | Wavelet threshold의 이론적 토대 |
| 5 | **Ch.9** Differentiation and Integration | 차분 방정식, 재귀 필터 | 심화 필터 설계를 위한 배경 |

### 불균형 데이터 심화 (1주) — *ML for Imbalanced Data*
| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.1** Overcoming the Challenge | 정확도의 역설, 문제 정의 | 문제를 올바르게 보는 시각 |
| 2 | **Ch.3** Metrics | Precision/Recall/F1/AUC-ROC | 무엇을 기준으로 판단할지 |
| 3 | **Ch.2** Oversampling Methods | SMOTE 전체 변형 비교 | 핵심 해결 기법 깊이 이해 |
| 4 | **Ch.4** Ensemble Methods | BalancedRF, EasyEnsemble | 실무에서 가장 강력한 방법 |
| 5 | **Ch.10** Threshold & Calibration | 임계값 최적화, 확률 보정 | 모델을 실제로 배포할 때 필수 |

### 시계열 이상탐지 심화 (2주) — *ML for Time-Series with Python*
| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.3** Preprocessing | 슬라이딩 윈도우, 정상화 | 파이프라인의 입력을 설계 |
| 2 | **Ch.6** Anomaly Detection | Isolation Forest, LOF, OC-SVM | 비지도 이상탐지 알고리즘 비교 |
| 3 | **Ch.7** Classification | 지도학습 분류기, 피처 중요도 | 레이블이 있을 때의 전략 |
| 4 | **Ch.11** Probabilistic Models | 베이지안 변화점 탐지 | 실시간 스트리밍 환경 심화 |

### 실무 적용 프로젝트 아이디어
- [ ] UCI ML Repository의 [CNC Mill Tool Wear Dataset](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill) 적용
- [ ] CWRU Bearing Dataset으로 베어링 결함 탐지
- [ ] NASA Turbofan Engine Degradation 데이터로 잔여 수명 예측(RUL)

---

*이 문서는 세션 1 강의 자료로 제작되었습니다.*  
*질문 및 피드백: Claude Code를 활용해 실습 중 막히는 부분을 언제든지 질문하세요.*

# 출처
## 섹션 1. 텍스트 북 : [Think DSP: Digital Signal Processing in Python](https://www.oreilly.com/library/view/think-dsp/9781491938508/)
## 섹션 2. 텍스트 북 : [Machine Learning for Imbalanced Data](https://www.oreilly.com/library/view/machine-learning-for/9781801070836/)
## 섹션 3. 텍스트 북 : [Machine Learning for Time-Series with Python](https://www.oreilly.com/library/view/machine-learning-for/9781801819626/)

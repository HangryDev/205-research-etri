# [실습 시 프롬프트 제안] 세션 1 — Claude Code 시연 & 실습 모음


> 세션 1(제조 데이터 심층 분석)의 각 강의 섹션에 있던 **Claude Code 시연**과 **실습** 부분을 한곳에 모았습니다.
> 각 항목의 이론·배경 설명은 해당 강의 섹션을 참고하세요.


---

```{admonition} 출처: 노이즈 제거
:class: note dropdown
아래 시연/실습은 원래 노이즈 제거** 섹션에 있던 내용입니다.
```

## 1-3. Claude Code 시연

```{admonition} 시연 포인트
:class: tip

코드 자체보다 **Claude에게 어떻게 질문하는가**를 주목하세요. 좋은 프롬프트의 핵심 세 가지:
1. **목적**을 명확히 말하고
2. 사용할 **라이브러리/방법**을 지정하고
3. 원하는 **출력 형태**를 구체적으로 요청
```

**시연 프롬프트**:

```
CNC 밀링머신 진동 데이터에서 채터링 노이즈를 제거하려고 해.
pywt 라이브러리를 사용해서:
1. db4 wavelet으로 5레벨 분해
2. 상위 2개 레벨(고주파)에 soft threshold 적용
3. 재합성 후 원본과 결과를 subplot으로 비교
샘플 데이터는 numpy로 직접 생성해줘.
```

**시연 흐름**:
- Raw signal 생성 → `wavedec` 분해 → Threshold 적용 → `waverec` 재합성 → before/after 비교
- 결과: 윗그래프는 울퉁불퉁한 원본, 아랫그래프는 깔끔하게 정리된 50Hz 사인파

**추가 실험**: "db4 대신 sym5를 쓰면 어떤 차이가 있어?"
- `db4`(Daubechies): 비대칭 → 일반적인 제조 진동에 적합
- `sym5`(Symlets): db보다 대칭적 → 위상 왜곡에 민감한 경우에 유리
- 파라미터를 바꿔가며 실험하는 것이 Claude Code의 핵심 활용법

---

## 1-4. 실습

### 실험 매트릭스

| 조합 | Wavelet | Level | 관찰 포인트 |
|------|---------|-------|-------------|
| A | `db4` | 3 | 기준선 |
| B | `db4` | 5 | 레벨이 늘면? |
| C | `sym5` | 3 | wavelet 종류 변경 시? |
| D | `sym5` | 5 | 둘 다 변경 시? |

### STEP 1: 가상 신호 만들기

```python
import numpy as np, pywt, matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 1, 1024)
cutting_signal = np.sin(2 * np.pi * 50 * t)        # 50Hz 절삭 신호
spindle_noise  = 0.3 * np.sin(2 * np.pi * 200 * t) # 200Hz 스핀들 노이즈
chatter        = 0.2 * np.random.randn(len(t))       # 채터링
signal = cutting_signal + spindle_noise + chatter
```

### STEP 2: 노이즈 제거 함수 — 직접 채워보세요

```python
def denoise_wavelet(signal, wavelet='db4', level=5, threshold_mode='soft'):
    # TODO: 3줄을 채워 넣으세요
    # 1. pywt.wavedec 로 분해
    # 2. 고주파 계수에 threshold 적용
    # 3. pywt.waverec 로 재합성
    pass
```

힌트:
```python
T = np.sqrt(2*np.log(len(signal))) * np.std(coeffs[-1])
# coeffs[-1] = cD1 (가장 고주파 계수)에서 σ 추정
# coeffs[0] (근사 계수)는 건드리지 말 것
# coeffs[1:] 슬라이싱으로 디테일 계수만 처리
```

### STEP 3: 결과 시각화

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(t, signal, label='원본 신호', alpha=0.7)
# TODO: axes[1]에 denoised 결과 그리기
plt.tight_layout(); plt.show()
```

```{admonition} 성공 체크리스트
:class: tip

- 윗그래프는 **울퉁불퉁**, 아랫그래프는 **50Hz 사인파**에 가까워야 합니다
- 아랫그래프가 여전히 울퉁불퉁 → threshold가 너무 낮음
- 아랫그래프가 너무 매끄러움 → 신호까지 같이 제거됨
- 레벨을 높이면 노이즈 제거가 더 정교해지지만, 너무 높이면 **신호까지 손상**될 수 있음
```

---

```{admonition} 출처: 데이터 불균형
:class: note dropdown
아래 시연/실습은 원래 데이터 불균형** 섹션에 있던 내용입니다.
```

## 2-3. Claude Code 시연

**시연 프롬프트**:

```
불량률 1%인 제조 데이터를 시뮬레이션해줘.
1. 기본 RandomForest 모델 학습
2. SMOTE 적용 후 재학습
3. class_weight='balanced' 적용 후 재학습
세 모델의 Confusion Matrix와 F1-score를 나란히 비교하는 시각화를 만들어줘.
```

**시연 결과**:
- **기본 모델**: 불량을 거의 잡지 못함, FN이 엄청나게 많음, Recall ≈ 0. 정확도는 99%
- **SMOTE 적용**: Recall 크게 개선, FN이 확 줄음. Precision은 약간 하락(오탐 증가)
- **class_weight**: SMOTE와 비슷한 수준으로 Recall 개선. 데이터는 건드리지 않음

**추가 질문**: "SMOTE가 항상 최선인가? 안 쓰는 게 나을 때는?"
- 데이터에 노이즈가 많거나 소수 클래스 샘플 자체가 잘못 라벨링된 경우 → SMOTE가 오히려 해가 됨
- 이럴 때는 class_weight가 더 안전
- 데이터가 충분히 많으면 불균형 자체가 큰 문제가 아닐 수도 있음

---

## 2-4. 실습

### STEP 1: 데이터 준비

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE

X, y = make_classification(n_samples=10000, weights=[0.99, 0.01],
                           n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### STEP 2: k_neighbors별 실험

```python
results = {}
for k in [3, 5, 7, 10]:
    smote = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(random_state=42).fit(X_res, y_res)
    pred = model.predict(X_test)
    results[k] = {'f1': f1_score(y_test, pred), 'recall': recall_score(y_test, pred)}
    print(f"k={k}: {sum(y_res == 1)} minority samples")
```

### STEP 3: 결과 비교

```python
import matplotlib.pyplot as plt
ks = list(results.keys())
plt.plot(ks, [results[k]['f1'] for k in ks], marker='o', label='F1')
plt.plot(ks, [results[k]['recall'] for k in ks], marker='s', label='Recall')
plt.xlabel('k_neighbors'); plt.ylabel('score'); plt.legend(); plt.show()
```

```{admonition} 실험 관찰 포인트
:class: tip

- **k=3**: 이웃이 너무 가까워 합성 샘플의 다양성이 떨어짐 (비슷한 점만 생성)
- **k=5~7**: 보통 F1이 가장 높게 나옴 (적절한 균형)
- **k=10**: 멀리 있는 샘플까지 포함 → 노이즈가 섞여 경계가 흐려짐
- 최적의 k는 **문제와 데이터에 따라 다름** — 이게 실무 모델 튜닝의 기본 과정
```

---

```{admonition} 출처: CNC 이상탐지 파이프라인
:class: note dropdown
아래 시연/실습은 원래 CNC 이상탐지 파이프라인** 섹션에 있던 내용입니다.
```

## 3-3. Claude Code 시연

**시연 프롬프트**:

```
CNC 밀링머신 공구 마모 탐지 파이프라인을 만들어줘.
- 시뮬레이션 데이터: 정상 구간 + 마모 시작 구간 포함
- 단계: Wavelet 노이즈 제거 → 슬라이딩 윈도우 특징 추출 → Isolation Forest
- 각 단계 결과를 subplot으로 시각화
- contamination=0.05
```

**시연 흐름 단계별 결과**:

**Step 1: 데이터 생성**
- 10초짜리 신호: 0~5초 정상, 5~7초 마모 시작, 7~10초 심각한 마모
- 노이즈 크기가 0.3 → 0.8로 증가하며 마모 진행 시뮬레이션

**Step 2: 노이즈 제거**
- db4 wavelet, level=5, soft threshold 적용
- 노이즈는 깔끔히 제거되지만 마모 구간의 특징은 그대로 유지
- Universal Threshold가 "보수적으로" 설계된 이유가 여기에 있음

**Step 3: 특징 추출**
- 정상 구간: RMS 낮고 일정 (~0.7), Kurtosis ~3
- 마모 시작 구간: RMS 서서히 상승 (0.7→0.9), Kurtosis 3 이상
- 세 특징 모두 마모 시작점(5초)에서 변화 시작

**Step 4: Isolation Forest**
- 이상 점수를 시계열로 시각화
- 정상 구간: 낮고 안정적인 점수
- 마모 시작 구간(5초): 점수가 확 올라감
- 심각한 마모 구간: 점수가 더 높아짐

**추가 질문**: "탐지 민감도를 높이려면 어떤 파라미터를 조정해야 하고, 트레이드오프는?"
- `contamination`을 높이면 더 민감해지지만 **오탐도 증가** (0.05 → 0.10)
- 윈도우 크기를 작게 하면 빠른 변화 감지 가능하지만 **노이즈에 취약**
- 배운 Recall-Precision 트레이드오프와 동일한 원리
- 현장에서는 "오탐 10번 vs 불량 하나 놓침"을 결정하고 contamination 조절

---

## 3-4. 실습

### STEP 1: 가상 마모 신호

```python
np.random.seed(42)
t = np.linspace(0, 10, 10000)
signal = np.sin(2 * np.pi * 50 * t)
signal[5000:7000] += 0.3 * np.random.randn(2000)  # 마모 시작
signal[7000:] += 0.8 * np.random.randn(3000)       # 심각한 마모
```

### STEP 2: Wavelet 노이즈 제거

```python
def denoise(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * np.std(coeffs[-1])
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)
```

### STEP 3: 슬라이딩 윈도우 특징 추출

```python
def extract_features_windowed(signal, window_size=100, step=50):
    features = []
    for i in range(0, len(signal) - window_size, step):
        w = signal[i:i+window_size]
        rms = np.sqrt(np.mean(w**2))
        peak = np.max(np.abs(w))
        std = np.std(w)
        kurt = np.mean(((w - np.mean(w)) / (std + 1e-8))**4)
        features.append([rms, peak, kurt])
    return np.array(features)
```

### STEP 4: Isolation Forest — contamination별 실험

```python
for contamination in [0.01, 0.05, 0.10]:
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(features)
    scores = -model.score_samples(features)  # 클수록 이상
    # TODO: 시간 축에 맞춰 scores 시계열로 시각화
```

```{admonition} 성공 체크리스트
:class: tip

- 마모 시작 **5초 부근**에서 이상 점수가 확 올라가야 성공
- **contamination=0.01**: 너무 보수적 — 마모 구간 일부만 이상으로 탐지
- **contamination=0.05**: 적절한 균형
- **contamination=0.10**: 너무 민감 — 정상 구간에서도 오탐 발생
- STEP 1과 2는 코드를 복사해서 바로 실행하고, **STEP 3과 4에 집중**하세요
```

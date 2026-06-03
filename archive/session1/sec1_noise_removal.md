# 섹션 1 | 노이즈 제거 — 신호에서 의미를 꺼내는 법

---

## 1-1. 문제 제기

CNC 밀링머신의 진동 센서가 수집하는 데이터에는 여러 신호가 **뒤섞여** 있습니다.

```
[실제 CNC 진동 데이터에 섞인 성분들]

절삭 신호     ──── 우리가 원하는 것 (공구-소재 접촉 정보)
스핀들 노이즈 ──── 회전체의 기계적 진동 (고주파, 일정한 패턴)
채터링       ──── 공구 떨림 (불규칙 고주파, 가공 불량의 전조)
전기 간섭    ──── 모터/인버터에서 오는 전자기 노이즈 (특정 주파수에 집중)
```

Raw plot을 보면 이 성분들을 **육안으로 구분하기 불가능**합니다.  
→ **핵심 질문**: 우리가 원하는 신호만 꺼낼 수 있을까?

---

## 1-2. 이론

### ① FFT로 먼저 보면 어떨까?

**FFT(Fast Fourier Transform)** 는 신호를 주파수 성분으로 분해합니다.

```{mermaid}
flowchart LR
    A["시간 영역 신호 x(t)"] --> B["FFT"]
    B --> C["주파수 스펙트럼 X(f)"]
```

```
       진폭                         진폭
        │  /\/\                       │ │  │    │
        │ /    \  /\/                 │ │  │    │  │
        └──────────── 시간            └─┴──┴────┴──┴── 주파수
```

- **장점**: "어떤 주파수 성분이 얼마나 있는가"를 한눈에 파악
- **단점**: **"그 주파수가 언제 발생했는가"를 알 수 없다**

```
예시) 채터링이 가공 시작 후 30초에 발생했는가, 50초에 발생했는가?
     FFT로는 알 수 없음 → 시간-주파수 동시 분석이 필요
```

```{admonition} 핵심
:class: important

**핵심 한계**: FFT는 신호 전체를 하나의 스펙트럼으로 압축하기 때문에,  
시간에 따라 변하는(비정상 신호, non-stationary) 데이터에는 적합하지 않습니다.
```

---

### ② Wavelet Transform의 직관

Wavelet Transform은 **"돋보기를 스케일 바꿔가며 신호를 훑는 것"** 입니다.

```
큰 스케일 (저주파 돋보기)  → 신호의 전체적인 추세(절삭 패턴)를 봄
중간 스케일               → 중간 주파수 성분 (스핀들 진동)을 봄
작은 스케일 (고주파 돋보기) → 빠른 변화(채터링, 전기 노이즈)를 봄
```

```{mermaid}
flowchart TB
    subgraph "Wavelet 분석 결과"
        L1["레벨 1 (고주파) — 전기 간섭 + 채터링 → 제거 대상"]
        L2["레벨 2 (중주파) — 스핀들 노이즈 → 제거 대상"]
        L3["레벨 3 (저주파) — 절삭 신호 → 보존 대상"]
        L4["Approximation — 장기 추세 → 보존 대상"]
    end
```

FFT와의 결정적 차이: **주파수 정보 + 시간 위치 정보를 동시에** 갖습니다.

---

### ③ Hard/Soft Threshold: 어떤 성분을 버릴 것인가

Wavelet 분해 후, 각 레벨의 계수(coefficient)에 **임계값(threshold)** 을 적용해서 노이즈를 제거합니다.

```
Hard Threshold:  |계수| < T  →  0으로 설정 (잘라내기)
                 |계수| ≥ T  →  그대로 유지

Soft Threshold:  |계수| < T  →  0으로 설정
                 |계수| ≥ T  →  크기를 T만큼 줄임 (부드럽게 처리)
```

```
어떤 것을 선택해야 할까?
- Hard: 강한 신호 보존, 경계에서 불연속 발생 가능
- Soft: 신호가 약간 왜곡되지만 더 부드러운 결과
- 제조 데이터 실무: 보통 Soft Threshold가 더 안정적
```

```{admonition} 핵심
:class: important

**임계값 T는 어떻게 정하나?**  
가장 널리 쓰이는 방법: `T = σ × √(2 × log(N))`  
σ: 노이즈 표준편차 추정값, N: 신호 샘플 수 (Universal Threshold, Donoho & Johnstone, 1994)
```

---

### ④ pywt 라이브러리 핵심 함수 3개

```python
import pywt
import numpy as np

# 1. 사용 가능한 wavelet 종류 확인
print(pywt.wavelist(kind='discrete'))  # db, sym, haar 등

# 2. 다중 레벨 분해 (DWT)
coeffs = pywt.wavedec(signal, wavelet='db4', level=5)
# coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]
#            근사  디테일5 디테일4 ... 디테일1(최고주파)

# 3. 재합성 (역 DWT)
reconstructed = pywt.waverec(coeffs_modified, wavelet='db4')
```

```{mermaid}
flowchart TB
    A["원본 신호"] --> B["pywt.wavedec() — 분해"]
    B --> C["계수 배열 cA, cD5, cD4, cD3, cD2, cD1"]
    C --> D["threshold 적용 — 노이즈 계수를 0에 가깝게"]
    D --> E["수정된 계수 배열"]
    E --> F["pywt.waverec() — 재합성"]
    F --> G["노이즈 제거된 신호"]
```

---

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

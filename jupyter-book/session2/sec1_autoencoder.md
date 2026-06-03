# 섹션 1 | Autoencoder — 정상을 학습해서 이상을 찾는다

> 참고 교재: *Hands-On Unsupervised Learning Using Python* (Ankur A. Patel, O'Reilly) — Ch.1 비지도 학습 개요, Ch.3 차원 축소, Ch.4 이상 탐지
>
> 소요 시간: 문제 제기 3분 + 이론 13분 + 시연 10분 + 실습 19분

---

## 1-1. 문제 제기

### 라벨 문제: 현장의 현실

- 공장에서 **수만 건의 정상**이 생산되는 동안 **결함은 수십 건**에 불과
- 그 수십 건에 **정확한 라벨**이 붙어 있을 가능성은 극히 낮음
  - "이건 분명 결함인데 어떤 종류인지 모르겠다"
  - "정상인가 결함인가 경계가 애매하다"
  - "검사원마다 판정이 다르다"

```
[현실적인 제조 데이터의 모습]

정상 데이터: ████████████████████████████  수만 건 (라벨 있음)
이상 데이터: █                             수십 건 (라벨 없음 or 불확실)
```

### 지도 학습의 한계

- **지도 학습은 라벨을 먹고 자란다** — RandomForest든 신경망이든 "이건 정상, 저건 결함"이라는 정답이 있어야 학습 가능
- 라벨이 없으면 출발 자체가 안 됨
- 새로운 결함 유형이 나타날 때마다 사람이 일일이 라벨링하는 것은 비현실적
- 신제품 라인이 가동되면 라벨이 쌓이기 전에 모델이 필요

```{admonition} 교재 인용
:class: tip

*Hands-On Unsupervised Learning Using Python* 1장: **"라벨이 충분히 있으면 지도 학습이 우월하다. 하지만 대부분의 실제 문제에서는 라벨이 부족하다."**

Yann LeCun의 비유: **"지능이 케이크라면, 비지도 학습은 케이크 자체이고, 지도 학습은 그 위의 아이싱이다."**
```

### 핵심 질문과 아이디어

- **핵심 질문**: "라벨 없이, 정상 데이터만 가지고 이상을 탐지할 수 있는가?"
- **핵심 아이디어**: **"정상만 학습한 모델은 이상을 재현하지 못한다."**
- 정상 패턴을 압축→복원하는 능력은 키웠지만, 처음 보는 이상 패턴은 복원이 안 됨
- 복원 실패의 크기, 즉 **재구성 오차(reconstruction error)** 가 이상 점수가 됨

![PCA 산점도: 정상과 이상의 분포 차이](../../lecture/images/s2_1_img04.png)

---

## 1-2. 이론

### ① 비지도 학습의 프레임워크

- **지도 학습**: 입력과 정답 라벨이 쌍으로 존재, 모델이 정답을 맞추도록 학습
- **비지도 학습**: 라벨 없이 데이터 자체의 구조를 학습

```{mermaid}
flowchart LR
    subgraph "지도학습"
        A1["입력"] --> B1["모델"]
        B1 --> C1["예측값\n(정답 라벨로 학습)"]
    end
    subgraph "비지도학습"
        A2["입력"] --> B2["모델"]
        B2 --> C2["구조/표현\n(데이터 자체로 학습)"]
    end
```

비지도 학습이 답하는 두 가지 질문:

1. "이 데이터는 어떤 구조를 가지고 있는가?"
2. "이 샘플은 다른 샘플들과 얼마나 다른가?"

이상 탐지에서의 비지도 학습 전략:

```{mermaid}
flowchart LR
    A["① 정상 데이터만으로\n'정상이 어떤 모습인지' 학습"] --> B["② 새 데이터가 들어오면\n'정상과 얼마나 다른지' 측정"]
    B --> C{"③ 차이가 임계값 초과?"}
    C -->|"Yes"| D["이상으로 판단"]
    C -->|"No"| E["정상으로 판단"]
```

![비지도 학습 프레임워크 개요](../../lecture/images/s2_1_img06.png)

**PCA와 Autoencoder의 관계**:

- Session 1에서 배운 PCA도 비지도 학습(차원 축소)의 일종 — 고차원 데이터를 저차원으로 압축
- 하지만 PCA는 **선형 변환**이라 복잡한 패턴을 잡기 어려움
- Autoencoder는 **비선형 차원 축소**를 수행 — 더 복잡한 패턴 학습 가능
- PCA는 선형 Autoencoder의 특수한 형태

---

### ② Autoencoder 구조와 재구성 오차

Autoencoder는 **인코더(Encoder)** 와 **디코더(Decoder)** 두 부분으로 구성됩니다.

```{mermaid}
flowchart LR
    A["입력 x\n[64차원]"] --> B["Encoder\n(압축)"]
    B --> C["잠재 표현 z\n[8차원]\n(latent space)"]
    C --> D["Decoder\n(복원)"]
    D --> E["재구성 x̂\n[64차원]"]
```

- **인코더**: 입력을 압축 — 64차원 → 32차원 → 8차원. 중간의 8차원이 **잠재 공간(latent space)**
- **디코더**: 압축된 표현을 다시 복원 — 8차원 → 32차원 → 64차원
- **손실 함수**: MSE(평균 제곱 오차) — 원본 입력과 복원 출력의 차이 측정

```
학습 목표: x와 x̂의 차이(재구성 오차)를 최소화
Loss = ||x - x̂||²  (MSE)
```

**표현 학습(Representation Learning)**:

- *Deep Learning with Python* 1장: 각 층을 거치면서 중요한 정보만 남고 불필요한 정보는 걸러지는 **"다단계 정보 증류 과정"**
- 데이터를 다른 관점에서 바라보고, 더 유용한 형태로 변환하는 것

**이상 탐지 원리**:

```{mermaid}
flowchart TB
    subgraph "정상 데이터 (학습한 패턴)"
        A1["정상 입력"] --> B1["Encoder"]
        B1 --> C1["정상 패턴의 잠재 표현"]
        C1 --> D1["Decoder"]
        D1 --> E1["잘 복원됨\n재구성 오차 낮음"]
    end
    subgraph "이상 데이터 (처음 보는 패턴)"
        A2["이상 입력"] --> B2["Encoder"]
        B2 --> C2["정상 패턴으로 억지로 표현"]
        C2 --> D2["Decoder"]
        D2 --> E2["복원 실패\n재구성 오차 높음"]
    end
```

```
결론: 재구성 오차 = 이상 점수
```

![Autoencoder 재구성 오차 분포: 정상 vs 이상](../../lecture/images/s2_1_img07.png)

**신용카드 사기 탐지 사례** (*Hands-On Unsupervised Learning* 4장):

- 28만 건 거래 중 492건만 사기인 데이터셋
- PCA로 차원 축소 후 재구성 오차 계산
- 상위 350개 의심 거래 중 **264개가 실제 사기** — 정밀도 75%
- 라벨 없이도 강력하게 작동

**PyTorch 구현**:

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

- `nn.Sequential`: 층들을 순서대로 묶어주는 컨테이너
- `latent_dim=8`: 잠재 공간의 차원 — 이 숫자가 작을수록 압축이 강해지고 이상 탐지에 유리

**재구성 오차 계산**:

```python
def reconstruction_error(model, x):
    x_hat = model(x)
    return ((x - x_hat) ** 2).mean(dim=1)  # 샘플별 MSE
```

**latent_dim — 핵심 하이퍼파라미터**:

- **너무 크면**: 이상도 잘 재구성되어 이상 탐지 불가 (교재 4장에서 PCA 성분 30개로 설정 시 사기 탐지 전혀 안 됨)
- **너무 작으면**: 정상도 재구성하지 못함
- **적절한 크기를 찾는 것이 관건** — 이것이 실습 과제

---

### ③ 임계값 설정: 어디서부터 이상인가

모델 학습만큼이나 **임계값 설정**이 중요합니다. 실무에서는 이 단계에서 가장 많은 시간을 씁니다.

```
[정상 데이터의 재구성 오차 분포]

   빈도
    │     ████
    │    ██████
    │   █████████
    │  ███████████
    │ ██████████████░░░░
    └──────────────┬──────── 재구성 오차
                  ↑
              임계값 T

T보다 낮은 오차 → 정상
T보다 높은 오차 → 이상
```

**세 가지 임계값 설정 방법**:

```
방법 1. 정상 오차의 95번째 백분위수 (가장 흔함)
  T = np.percentile(normal_errors, 95)
  → "정상 샘플의 5%는 이상으로 오탐되는 것을 허용"

방법 2. 평균 + 3시그마
  T = μ + 3σ  (3-sigma rule)
  → 정상 데이터의 99.7%가 평균에서 3시그마 이내

방법 3. 도메인 전문가 판단
  "재구성 오차 0.05 이상이면 작업자에게 알림"
  → 실무에서 최종 조정은 항상 사람이 함
```

![임계값에 따른 이상 탐지율 변화](../../lecture/images/s2_1_img08.png)
![ROC 커브: 재구성 오차 기반 이상 탐지](../../lecture/images/s2_1_img09.png)

**트레이드오프**:

| 임계값 설정 | 민감도 | 오탐율 | 적용 사례 |
|:-----------|:------|:------|:---------|
| T 낮게 | 이상을 더 많이 잡음 | 정상을 이상으로 오탐 (현장 피로도 증가) | 안전 직결 설비 |
| T 높게 | 실제 이상 놓칠 수 있음 | 알림이 울리면 진짜 이상 | 단순 품질 검사 |

```{admonition} 주의 — 알림 피로의 위험
:class: warning

임계값을 낮추면 오탐이 늘어나 현장 작업자가 "늘 알림이 울리는데 다 괜찮은 거네"라며 알림을 무시하게 됨.
오히려 위험한 상황이 발생할 수 있음.
```

**비지도 학습 평가의 어려움** (교재 4장):

- 알려진 사기 패턴은 잡을 수 있지만, 아직 발견되지 않은 새로운 패턴은 평가 불가
- 비지도 학습의 진정한 가치: **알려지지 않은 새로운 이상 패턴을 발견할 가능성**

---

### PCA vs Autoencoder 비교

![PCA vs Autoencoder 차원 축소 비교](../../lecture/images/s2_1_img10.png)
![PCA 주성분 분석 결과](../../lecture/images/s2_1_img11.png)
![Autoencoder 잠재 공간 시각화](../../lecture/images/s2_1_img12.png)

- PCA는 데이터의 **분산을 최대화**하는 축을 찾음 (선형)
- Autoencoder는 **재구성 오차를 최소화**하는 잠재 표현을 학습 (비선형)
- 결과적으로 유사한 목적이지만 Autoencoder가 더 높은 표현력을 가짐

---

## 1-3. Claude Code 시연

```{admonition} 시연 포인트
:class: tip

모델 구현 자체가 아니라, **"재구성 오차가 실제로 이상 신호를 포착하는지"** 를 데이터로 확인하는 흐름에 집중.
Autoencoder는 PyTorch 몇 줄이면 만들 수 있음. 진짜 어려운 건 결과의 분포를 보고 의미를 읽어내는 것.
```

**Claude 프롬프트**:

```text
제조 진동 데이터 이상탐지용 Autoencoder를 만들어줘.
- 정상 데이터: 사인파 기반 합성 신호 (1000개)
- 이상 데이터: 노이즈가 추가된 신호 (50개, 학습에는 사용 안 함)
- PyTorch로 Autoencoder 구현 (input_dim=64, latent_dim=8)
- 정상 데이터로만 학습 (epoch=50)
- 결과: 정상/이상 재구성 오차 분포를 겹쳐서 시각화
- 95th percentile 임계값 표시
```

### 시연 흐름 5단계

**1단계 — 합성 데이터 생성**

- 정상: 깨끗한 사인파 + 작은 가우시안 노이즈 (1000개)
- 이상: 같은 사인파 + 강한 노이즈 + 임의의 스파이크 (50개)
- **이상 50개는 학습에 절대 사용하지 않음** — 평가용으로만

**2단계 — Autoencoder 정의**

```python
# 인코더: Linear(64→32) → ReLU → Linear(32→8)
# 디코더: Linear(8→32) → ReLU → Linear(32→64)
# 잠재 공간 8차원이 정보의 병목 역할
```

**3단계 — 정상 데이터로만 학습**

- MSE 손실로 정상 입력과 복원 출력 사이의 거리를 최소화
- 50 epoch 학습
- **이상 데이터는 학습에 들어가지 않음**

**4단계 — 재구성 오차 계산**

- 정상 1000개와 이상 50개 각각에 대해 MSE 계산
- 가설: 정상은 학습한 패턴이라 오차가 낮고, 이상은 처음 보는 패턴이라 오차가 높을 것

**5단계 — 분포 비교 시각화**

- 정상 분포가 왼쪽에 몰려 있고, 이상 분포가 오른쪽에 위치하면 성공
- 정상 오차의 95번째 백분위수에 빨간 세로선 = 임계값

### 시연 결과

- **정상 재구성 오차**: 평균 0.01 근처에 몰려 있음
- **이상 재구성 오차**: 0.05~0.15 사이에 퍼져 있음
- 두 분포가 거의 겹치지 않으며, 임계값이 두 분포 사이에 깔끔하게 위치
- 이상 샘플 중 약 90% 이상이 임계값 초과

### 시연 후 질문

> **"latent_dim을 8에서 2로 줄이면 분리가 어떻게 변할까? 반대로 32나 64로 키우면?"**

- **latent_dim 축소**: 압축이 강해져 정상의 핵심 패턴만 살아남고 이상은 더더욱 복원 안 됨 — **분리가 더 선명해질 수 있음**. 하지만 너무 작으면 정상도 제대로 복원이 안 돼서 임계값이 모호해짐
- **latent_dim 확대**: 잠재 공간이 입력 차원과 같아지면 사실상 복사가 가능해짐 — 이상조차도 거의 완벽히 복원 — **두 분포가 겹쳐서 분리 안 됨**

---

## 1-4. 실습

### 과제: latent_dim 변경하며 정상/이상 분리도 비교

| 실험 | latent_dim | 관찰 포인트 |
|:-----|:-----------|:-----------|
| A | 32 | 압축이 거의 없음 — 이상도 잘 복원? |
| B | 16 | 기준선 |
| C | 8 | 강의 시연과 동일 |
| D | 4 | 강한 압축 — 분리도 변화? |

### 분리도 측정

```python
threshold = np.percentile(normal_errors, 95)
recall = (anomaly_errors > threshold).mean()
print(f"latent_dim={latent_dim}, 이상 탐지율: {recall:.2%}")
```

### 실습 시작 코드

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(42)
t = np.linspace(0, 1, 64)

# 정상 데이터: 사인파 (약간의 노이즈 포함)
normal_data = np.array([
    np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(64)
    for _ in range(1000)
], dtype=np.float32)

# 이상 데이터: 강한 노이즈
anomaly_data = np.array([
    np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(64)
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
    # TODO: 모델 생성 -> 학습 -> 오차 계산 -> 분리도 평가
    pass

for latent_dim in [32, 16, 8, 4]:
    train_and_evaluate(latent_dim, normal_data, anomaly_data)
```

### 실습 포인트

1. 각 `latent_dim`에서 정상과 이상의 재구성 오차 분포 변화를 그래프로 확인
2. 탐지율이 가장 높은 `latent_dim`을 찾기
3. 압축이 너무 약할 때와 너무 강할 때 각각 어떤 문제가 생기는지 직접 경험

### 제출물

- 4개 분포 그래프
- Recall 비교 표
- "너무 강하거나 약한 압축이 초래하는 문제는 무엇인가?"에 대한 한 문단

---

## 참고 문헌

- *Hands-On Unsupervised Learning Using Python* (Ankur A. Patel, O'Reilly)
  - Ch.1: 비지도 학습 개요
  - Ch.3: 차원 축소 (PCA 등)
  - Ch.4: 이상 탐지
- *Deep Learning with Python* 2판 (Francois Chollet, Manning)
  - Ch.1: 표현 학습

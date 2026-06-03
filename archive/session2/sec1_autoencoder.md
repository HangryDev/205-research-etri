# 섹션 1 | Autoencoder — 정상을 학습해서 이상을 찾는다

---

## 1-1. 문제 제기

**상황**: 공장에서 수만 개의 정상 제품이 생산되는 동안 불량은 몇 개뿐입니다.  
이 불량에는 라벨이 붙어 있지 않습니다. 작업자가 미처 발견하지 못했거나, 아직 고장이 나지 않은 상태이기 때문입니다.

```
[현실적인 제조 데이터의 모습]

정상 데이터: ████████████████████████████  수만 건 (라벨 있음)
이상 데이터: █                             수십 건 (라벨 없음 or 불확실)
```

**지도학습으로 해결하려면**:
- 불량 샘플이 충분히 필요 → 현실에서 불가능
- 라벨링 작업 필요 → 비용과 시간 소모
- 새로운 유형의 불량 → 기존 모델이 탐지 못함

**핵심 질문**: 라벨 없이, 정상 데이터만 가지고 이상을 찾을 수 있을까?  
→ **핵심 아이디어**: 정상만 보고 학습한 모델은 이상을 제대로 재현하지 못한다

---

## 1-2. 이론

### ① 비지도 학습의 프레임워크

라벨 없이 데이터 자체의 구조를 학습합니다.

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

이상탐지에서의 비지도 학습 전략:

```{mermaid}
flowchart LR
    A["정상 데이터만으로\n'정상이 어떤 모습인지' 학습"] --> B["새로운 데이터가 들어오면\n'정상과 얼마나 다른지' 측정"]
    B --> C{"차이가 임계값 초과?"}
    C -->|"Yes"| D["이상으로 판단"]
    C -->|"No"| E["정상으로 판단"]
```

---

### ② Autoencoder 구조와 재구성 오차

Autoencoder는 **압축(Encoder) → 복원(Decoder)** 을 학습하는 신경망입니다.

```{mermaid}
flowchart LR
    A["입력 x\n(원본 데이터)\n[64차원]"] --> B["Encoder\n(압축)"]
    B --> C["잠재 표현 z\n(압축된 표현)\n[8차원]"]
    C --> D["Decoder\n(복원)"]
    D --> E["재구성 x̂\n(복원된 데이터)\n[64차원]"]
```

```
학습 목표: x와 x̂의 차이(재구성 오차)를 최소화
Loss = ||x - x̂||²  (MSE)
```

**이상탐지에 활용하는 원리**:

```{mermaid}
flowchart TB
    subgraph "정상 데이터로만 학습"
        A1["정상 입력"] --> B1["Encoder"]
        B1 --> C1["정상 패턴의 잠재 표현"]
        C1 --> D1["Decoder"]
        D1 --> E1["잘 복원됨\n재구성 오차 ↓ (낮음)"]
    end
    subgraph "이상 데이터 투입"
        A2["이상 입력"] --> B2["Encoder"]
        B2 --> C2["정상 패턴으로 억지로 표현"]
        C2 --> D2["Decoder"]
        D2 --> E2["복원 실패\n재구성 오차 ↑ (높음)"]
    end
```

```
결론: 재구성 오차 = 이상 점수
```

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

# 재구성 오차 계산
def reconstruction_error(model, x):
    x_hat = model(x)
    return ((x - x_hat) ** 2).mean(dim=1)  # 샘플별 MSE
```

---

### ③ 임계값 설정: 어디서부터 이상인가

모델을 학습했다고 끝이 아닙니다. **임계값을 어떻게 정하느냐**가 실무의 핵심입니다.

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

**임계값 설정 방법 3가지**:

```
방법 1. 백분위수 기반 (가장 흔함)
  T = 정상 데이터 재구성 오차의 95th percentile
  → "정상 샘플의 5%는 이상으로 오탐되는 것을 허용"

방법 2. 평균 + k×표준편차
  T = μ + 3σ  (3-sigma rule)
  → 정규분포를 가정할 때 유효

방법 3. 도메인 전문가 설정
  "재구성 오차 0.05 이상이면 작업자에게 알림"
  → 실무에서 최종 조정은 항상 사람이 함
```

**트레이드오프 정리**:

```{admonition} 주의
:class: warning

**임계값 T ↓ (낮게 설정)**
  → 민감도 ↑: 이상을 더 많이 잡음
  → 오탐 ↑: 정상을 이상으로 잘못 판단 (현장 피로도 증가)

**임계값 T ↑ (높게 설정)**
  → 민감도 ↓: 실제 이상을 놓칠 수 있음
  → 오탐 ↓: 알림이 울리면 진짜 이상

**제조업 판단 기준**:
  안전과 직결된 설비 → T를 낮게 (놓치는 것이 더 위험)
  단순 품질 검사 → T를 높게 (오탐이 더 비쌈)
```

---

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

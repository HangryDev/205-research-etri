# 섹션 1 | Diffusion Model — 노이즈에서 결함 이미지를 꺼내는 법

:::{admonition} 참고 교재
:class: note
**참고 교재**: *Hands-On Generative AI with Transformers and Diffusion Models* (Omar Sanseviero et al., O'Reilly)
**소요 시간**: 45분 (문제 제기 3 + 이론 13 + 시연 10 + 실습 19)
:::

---

## 1-1. 문제 제기 (3분)

**상황**: 제조 결함 탐지 모델을 학습시키려는데 데이터가 부족합니다.

```
[제조 현장의 데이터 현실]

정상 제품 이미지:  ████████████████████  수만 장 (쉽게 수집 가능)
스크래치 결함:     ██                    수백 장
기포 결함:         █                     수십 장
크랙 결함:         ░                     수 장   ← 학습 불가 수준
```

세션 1에서 SMOTE로 해결했지만, 이미지 데이터에 SMOTE를 적용하면 어떤 문제가 생길까요?

```
[이미지에 SMOTE를 적용하면]

결함 이미지 A: [픽셀값 배열]
결함 이미지 B: [픽셀값 배열]
두 이미지의 중간값: 흐릿하고 의미없는 혼합 이미지

→ 픽셀 보간은 의미 있는 새 결함을 만들지 못함
→ 필요한 것: "진짜처럼 보이는" 새 결함 이미지
```

**핵심 질문**: 없는 결함 이미지를 진짜처럼 만들어낼 수 있을까?
→ 세션 1·2의 데이터 부족 문제를 **데이터 생성**으로 해결하는 접근

---

## 1-2. 이론 (13분)

### ① Diffusion의 직관: 노이즈 추가 → 노이즈 제거 (5분)

:::{admonition} 참고 교재
:class: note
📖 *Hands-On Generative AI* **Ch.3 Diving Into Diffusion Models** — Diffusion 모델의 Forward/Reverse Process 원리 / **Ch.4 Transformer-Based Diffusion Models** — U-Net 구조와 노이즈 예측 메커니즘
:::

Diffusion 모델이 배우는 것은 딱 하나입니다: **"이 이미지에서 노이즈를 빼면 어떤 이미지가 나오는가"**

```{mermaid}
flowchart TD
    A["원본 결함 이미지 x₀"] --> B["노이즈 조금 추가"]
    B --> C["x₁ (약간 흐려짐)"]
    C --> D["노이즈 추가"]
    D --> E["x₂ (더 흐려짐)"]
    E --> F["... (T번 반복)"]
    F --> G["xₜ (완전한 노이즈)"]
```

```
[Forward Process: 노이즈 추가 (학습 데이터 생성)]

원본 결함 이미지 x₀
      ↓ 노이즈 조금 추가
      x₁  (약간 흐려짐)
      ↓ 노이즈 추가
      x₂  (더 흐려짐)
      ↓ ... (T번 반복)
      xₜ  (완전한 노이즈)

→ 모델은 (xₜ, t) 쌍을 보고 "얼마나 노이즈가 있는지" 학습
```

```{mermaid}
flowchart TD
    A["순수 노이즈 xₜ"] --> B["노이즈 조금 제거 (모델 예측)"]
    B --> C["xₜ₋₁"]
    C --> D["반복..."]
    D --> E["x₀ (새로운 결함 이미지 생성!)"]
```

```
[Reverse Process: 노이즈 제거 (이미지 생성)]

순수 노이즈 xₜ
      ↓ 노이즈 조금 제거 (모델 예측)
      xₜ₋₁
      ↓ 반복...
      x₀  (새로운 결함 이미지 생성!)
```

:::{admonition} 핵심
:class: important
**핵심 직관**: 모델은 "노이즈 → 이미지" 방향의 역과정을 학습합니다.
한 번도 본 적 없는 새 이미지를 순수 노이즈에서 생성할 수 있습니다.
:::

---

### ② Fine-tuning 전략: 적은 데이터로 특정 스타일 학습 (5분)

:::{admonition} 참고 교재
:class: note
📖 *Hands-On Generative AI* **Ch.6 Fine-Tuning Language Models** — Fine-tuning의 원리와 LoRA / DreamBooth 전략 비교
:::

Stable Diffusion 같은 대형 모델을 처음부터 학습하는 것은 불가능합니다.
**Fine-tuning**: 이미 학습된 모델을 우리 결함 이미지에 맞게 **미세조정**합니다.

```
[Fine-tuning 방법 비교]

방법           필요 이미지 수   학습 시간   파라미터 변경
──────────────────────────────────────────────────────
DreamBooth     3~20장          수십 분     전체 모델 업데이트
LoRA           10~100장        수 분       일부 파라미터만 추가
Textual Inversion  5~10장     수 분       텍스트 임베딩만 학습
```

**LoRA (Low-Rank Adaptation) 직관**:

```{mermaid}
flowchart LR
    A["원래 모델 가중치 W<br/>거대한 행렬 (변경 안 함)"] --> B["+ LoRA 추가<br/>작은 행렬 두 개의 곱 (ΔW = A × B)"]
```

```
[원래 모델 가중치 W]         [LoRA 추가]
거대한 행렬 (변경 안 함)  +  작은 행렬 두 개의 곱 (ΔW = A × B)

학습하는 파라미터 수: 전체의 0.1~1% 수준
→ 결함 이미지 10장으로도 특정 결함 패턴 학습 가능
→ 학습 완료 후 원본 모델에 병합 가능
```

```python
from diffusers import StableDiffusionPipeline
import torch

# 사전 fine-tuning된 결함 생성 모델 로드
# (실습에서는 학습 실행 대신 사전 준비된 가중치 사용)
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# LoRA 가중치 로드 (fine-tuning 결과)
pipeline.load_lora_weights("./lora_defect_weights")
pipeline = pipeline.to("cuda")
```

:::{admonition} 주의
:class: warning
**실습 주의사항**: Fine-tuning 학습 자체는 GPU에서 수십 분~수 시간이 필요합니다.
이번 실습에서는 **사전 준비된 가중치**를 로드해서 생성 결과만 실험합니다.
:::

---

### ③ 생성 품질 평가: 무엇이 좋은 이미지인가 (3분)

:::{admonition} 참고 교재
:class: note
📖 *Hands-On Generative AI* **Ch.3 Diving Into Diffusion Models** — 생성 이미지 품질 평가 지표와 실무 활용
:::

```
[생성 이미지 평가 지표 3가지]

① 육안 검사 (가장 기본)
   → 실제 결함처럼 보이는가?
   → 결함 위치·형태가 자연스러운가?

② SSIM (Structural Similarity Index)
   → 구조적 유사도: 0(완전 다름) ~ 1(완전 동일)
   → 생성 이미지와 실제 결함의 구조적 패턴 비교
   → 실무 기준: SSIM > 0.7이면 유사하다고 판단

③ FID (Fréchet Inception Distance)
   → 생성된 이미지 집합 전체의 분포와
     실제 이미지 분포 사이의 거리
   → 낮을수록 좋음 (0에 가까울수록 실제와 유사)
   → 단점: 계산에 최소 수백 장의 이미지 필요
```

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(real_img, generated_img):
    """두 이미지의 SSIM 계산"""
    # 그레이스케일로 변환 후 비교
    real_gray = np.mean(real_img, axis=2)
    gen_gray = np.mean(generated_img, axis=2)
    score = ssim(real_gray, gen_gray, data_range=1.0)
    return score

# 사용 예시
score = calculate_ssim(real_defect, generated_defect)
print(f"SSIM: {score:.3f}")  # 0.7 이상이면 품질 양호
```

---

## 1-3. Claude Code 시연 (10분)

:::{admonition} 팁
:class: tip
**시연 포인트**: Fine-tuning 학습 과정이 아닌 **생성 결과의 품질을 평가하는 방법**에 집중하세요.
:::

**Claude에게 던질 프롬프트 예시**:
```
Stable Diffusion으로 제조 결함 이미지를 생성하고 품질을 평가해줘.
- diffusers 라이브러리로 파이프라인 구성
- 결함 유형 3가지 프롬프트:
  "scratch defect on metal surface, industrial quality inspection"
  "bubble defect on painted surface, factory inspection photo"
  "crack defect on ceramic component, quality control"
- num_inference_steps=50으로 각 유형당 3장 생성
- 결과: 3×3 grid 시각화
- 생성 이미지와 합성 실제 결함 이미지(numpy 생성)의 SSIM 계산
- GPU 없을 때를 위해 CPU fallback 처리 포함
```

**시연 흐름**:
1. 사전 준비된 파이프라인 로드
2. 결함 유형별 이미지 생성 (3가지)
3. 생성 결과 grid 시각화
4. SSIM으로 실제 결함 이미지와 유사도 계산
5. **Claude에게 추가 질문**: *"생성된 이미지가 너무 비슷하게만 나오는 문제(mode collapse)는 왜 생기고, 어떻게 해결해?"*

---

## 1-4. 실습 (19분)

### 과제

`num_inference_steps`를 바꿔가며 생성 속도와 이미지 품질의 트레이드오프를 분석하세요.

| 실험 | num_inference_steps | 생성 시간(초) | SSIM | 관찰 포인트 |
|------|-------------------|------------|------|----------|
| A | 20 | 측정 | 측정 | 빠르지만 품질은? |
| B | 50 | 측정 | 측정 | 기본값 |
| C | 100 | 측정 | 측정 | 느리지만 더 좋은가? |

**제출 항목**:
- 3가지 steps 설정의 생성 이미지 비교 (각 설정당 1장씩, subplot)
- 생성 시간 + SSIM 비교 표
- "실제 제조 현장에서 실시간 생성이 필요하다면 어떤 steps를 선택할 것인가?" 한 문단

### 실습 시작 코드

```python
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from skimage.metrics import structural_similarity as ssim

# GPU 없는 환경을 위한 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 파이프라인 로드
# GPU 환경: 실제 Stable Diffusion 모델
# CPU 환경: 경량 모델로 대체
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe",  # 테스트용 경량 모델
        torch_dtype=dtype
    ).to(device)
except Exception as e:
    print(f"모델 로드 실패: {e}")
    print("합성 이미지로 대체합니다.")
    pipe = None

prompt = "scratch defect on metal surface, industrial inspection photo"
results = {}

for steps in [20, 50, 100]:
    if pipe is not None:
        start = time.time()
        image = pipe(prompt, num_inference_steps=steps).images[0]
        elapsed = time.time() - start
        img_array = np.array(image) / 255.0
    else:
        # 파이프라인 없을 때 합성 이미지로 대체
        elapsed = steps * 0.01  # 시뮬레이션
        img_array = np.random.rand(512, 512, 3)

    # 실제 결함 이미지 (합성으로 대체)
    real_defect = np.random.rand(512, 512, 3)

    # SSIM 계산
    real_gray = np.mean(real_defect, axis=2)
    gen_gray = np.mean(img_array, axis=2)
    ssim_score = ssim(real_gray, gen_gray, data_range=1.0)

    results[steps] = {
        'image': img_array,
        'time': elapsed,
        'ssim': ssim_score
    }
    print(f"steps={steps}: {elapsed:.1f}초, SSIM={ssim_score:.3f}")

# TODO: 결과 비교 시각화 (subplot 3개 + 비교 표)
```

---

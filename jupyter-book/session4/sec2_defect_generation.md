# 섹션 2 | Defect Generation — 프롬프트로 결함을 설계한다

:::{admonition} 참고 교재
:class: note
**참고 교재**: *Generative Deep Learning, 2nd Ed.* (David Foster, O'Reilly)
**소요 시간**: 45분 (문제 제기 3 + 이론 13 + 시연 10 + 실습 19)
:::

---

## 2-1. 문제 제기 (3분)

**섹션 1에서 배운 것**: Diffusion으로 결함 이미지를 생성할 수 있음
**이번 섹션의 질문**: 원하는 결함을 **정확하게 제어**할 수 있을까?

```
[제어의 필요성]

섹션 1: "스크래치 결함 이미지" → 랜덤한 스크래치 생성
          어디에? 얼마나 깊게? 어떤 방향으로? → 제어 불가

섹션 2: "금속 표면 왼쪽 모서리에 5mm 깊이의 수직 스크래치, 고강도"
          → 원하는 조건을 텍스트로 지정
```

**현장에서 왜 이것이 중요한가**:

```
[결함 데이터 수집의 현실]

필요한 것: 스크래치 길이 1~10mm 범위의 균일한 분포
현실:      현장에서 발생하는 스크래치는 랜덤 → 특정 크기만 부족

해결책: 부족한 조건의 결함을 프롬프트로 명세화해서 정확히 생성
```

---

## 2-2. 이론 (13분)

### ① 조건부 생성의 직관: CGAN에서 Diffusion으로 (4분)

:::{admonition} 참고 교재
:class: note
📖 *Generative Deep Learning* **Ch.4 Generative Adversarial Networks** — CGAN의 조건부 생성 원리 / **Ch.13 Multimodal Models** — 텍스트-이미지 멀티모달 생성 모델
:::

조건부 생성이라는 아이디어는 GAN 시대부터 있었습니다.

```{mermaid}
flowchart TD
    A["CGAN (Conditional GAN, 2014)"] --> B["조건 c (클래스 레이블)"]
    B --> C["Generator"]
    C --> D["조건에 맞는 이미지"]
    D --> E["한계: 클래스 레이블만 가능, 세밀한 제어 불가"]
```

```{mermaid}
flowchart TD
    A["Text-to-Image Diffusion (2022~)"] --> B["조건 c (텍스트 문장)"]
    B --> C["Diffusion"]
    C --> D["텍스트에 맞는 이미지"]
    D --> E["강점: 언어의 유연성으로 무한한 조건 표현 가능"]
```

```
[조건부 생성의 발전]

CGAN (Conditional GAN, 2014):
  조건 c (클래스 레이블) → Generator → 조건에 맞는 이미지
  예: c = "스크래치" → 스크래치 이미지 생성
  한계: 클래스 레이블만 가능, 세밀한 제어 불가

Text-to-Image Diffusion (2022~):
  조건 c (텍스트 문장) → Diffusion → 텍스트에 맞는 이미지
  예: "왼쪽에 깊은 스크래치가 있는 금속 표면" → 정확한 이미지
  강점: 언어의 유연성으로 무한한 조건 표현 가능
```

:::{admonition} 핵심
:class: important
**핵심**: 조건부 생성이라는 아이디어는 같고, 조건을 표현하는 방법이 발전했습니다.
:::

---

### ② CLIP: 텍스트와 이미지를 연결하는 다리 (5분)

:::{admonition} 참고 교재
:class: note
📖 *Generative Deep Learning* **Ch.13 Multimodal Models** — CLIP의 대조 학습 원리와 텍스트-이미지 공간 연결
:::

Stable Diffusion이 텍스트 프롬프트를 이해할 수 있는 이유는 **CLIP** 덕분입니다.

```
[CLIP의 학습 원리]

인터넷에서 수억 개의 (이미지, 텍스트 설명) 쌍으로 학습:

"a scratch on metal"  ←→  [스크래치 이미지]
"a bubble defect"     ←→  [기포 이미지]
"a crack on ceramic"  ←→  [크랙 이미지]

결과: 텍스트와 이미지를 같은 벡터 공간에서 표현
      비슷한 의미 = 가까운 벡터 위치
```

```{mermaid}
flowchart TD
    A["텍스트 프롬프트"] --> B["CLIP Text Encoder"]
    B --> C["텍스트 임베딩 벡터<br/>(의미를 숫자로 표현)"]
    C --> D["Diffusion (Cross-Attention)"]
    D --> E["프롬프트 의미에 맞는 이미지 생성"]
```

```
[Stable Diffusion에서 CLIP의 역할]

텍스트 프롬프트
      ↓ CLIP Text Encoder
텍스트 임베딩 벡터 (의미를 숫자로 표현)
      ↓ Diffusion (Cross-Attention)
프롬프트 의미에 맞는 이미지 생성
```

```python
from transformers import CLIPTextModel, CLIPTokenizer

# CLIP 텍스트 인코더로 프롬프트 임베딩 확인
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

prompts = [
    "scratch defect on metal surface",
    "deep crack on ceramic material",
    "bubble defect on painted surface"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    embeddings = text_encoder(**inputs).last_hidden_state
    print(f"'{prompt[:30]}...' → 임베딩 shape: {embeddings.shape}")
    # shape: [1, 토큰 수, 512] → 이 벡터가 Diffusion의 조건이 됨
```

---

### ③ 제조 결함 프롬프트 설계 원칙 (4분)

:::{admonition} 참고 교재
:class: note
📖 *Generative Deep Learning* **Ch.13 Multimodal Models** — 효과적인 텍스트 프롬프트 설계와 네거티브 프롬프트 활용
:::

좋은 프롬프트는 구조화된 정보를 담습니다.

```
[결함 프롬프트 설계 공식]

[결함 유형] + [위치] + [심각도] + [소재/배경] + [촬영 스타일]

예시:
"a deep vertical scratch defect on the left edge of a stainless steel surface,
 industrial quality inspection photo, high resolution, studio lighting"

분해:
  결함 유형: "deep vertical scratch defect"
  위치:      "on the left edge"
  소재:      "stainless steel surface"
  촬영 스타일: "industrial quality inspection photo, high resolution"
```

**좋은 프롬프트 vs 나쁜 프롬프트**:

```
❌ 나쁜 프롬프트:
  "defect"  → 너무 모호, 어떤 결함인지 불분명

✅ 좋은 프롬프트:
  "hairline crack defect, 2cm length, diagonal orientation,
   on white ceramic surface, macro photography, sharp focus"

❌ 나쁜 프롬프트:
  "very very bad defect extremely damaged"  → 강조 반복은 효과 없음

✅ 좋은 프롬프트:
  "severe deep crack defect, multiple branching lines"  → 구체적 묘사
```

**네거티브 프롬프트**: 원하지 않는 요소를 제거합니다.

```python
# 네거티브 프롬프트: 생성에서 제외할 요소 지정
positive_prompt = "scratch defect on metal surface, inspection photo"
negative_prompt = "blurry, low quality, cartoon, illustration, text, watermark"

image = pipeline(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

---

## 2-3. Claude Code 시연 (10분)

:::{admonition} 팁
:class: tip
**시연 포인트**: 프롬프트 설계가 생성 결과에 미치는 영향을 **직접 비교**하는 과정에 집중하세요.
:::

**Claude에게 던질 프롬프트 예시**:
```
제조 결함 유형별 프롬프트 템플릿을 설계하고 이미지를 생성해줘.

결함 유형별 프롬프트 (각각 3가지 강도):
1. 스크래치: 경미 / 중간 / 심각
2. 기포: 소형 / 중형 / 대형
3. 크랙: 표면 / 깊은 / 관통

- diffusers 파이프라인으로 각 프롬프트 생성
- 결과: 3×3 grid (행=결함유형, 열=심각도)
- guidance_scale=7.5 고정
- guidance_scale을 3 / 7.5 / 15로 바꾼 비교도 추가
```

**시연 후 Claude에게 추가 질문**:
> *"같은 프롬프트인데 결과가 매번 다른 이유는 뭐야?
> 재현 가능한 결과를 원한다면 어떻게 해야 해?"*

**시연 흐름**:
1. Claude와 함께 결함 프롬프트 템플릿 설계
2. 유형별 × 강도별 이미지 생성
3. 3×3 grid 시각화
4. `guidance_scale` 변화 효과 확인
5. **seed 고정**으로 재현성 확보 방법 시연

---

## 2-4. 실습 (19분)

### 과제

`guidance_scale`을 바꿔가며 프롬프트 충실도와 이미지 다양성의 트레이드오프를 분석하세요.

| 실험 | guidance_scale | 특성 | 관찰 포인트 |
|------|--------------|------|----------|
| A | 3 | 프롬프트 약하게 반영 | 다양하지만 프롬프트와 다를 수 있음 |
| B | 7.5 | 기본값 (권장) | 균형 |
| C | 12 | 프롬프트 강하게 반영 | 충실하지만 부자연스러울 수 있음 |
| D | 20 | 과도하게 반영 | 과포화, 아티팩트 발생 |

**제출 항목**:
- 4가지 guidance_scale의 생성 이미지 비교 (동일 seed, subplot)
- "결함 데이터 생성 목적으로 어떤 guidance_scale을 선택할 것인가?" 한 문단

### 실습 시작 코드

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# 파이프라인 로드 (섹션 1과 동일)
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    use_pipe = True
except:
    use_pipe = False
    print("파이프라인 없음: 합성 이미지로 대체")

prompt = (
    "scratch defect on stainless steel surface, "
    "industrial quality inspection photo, high resolution"
)
negative_prompt = "blurry, low quality, cartoon, text, watermark"

SEED = 42  # 재현성을 위한 seed 고정
guidance_scales = [3, 7.5, 12, 20]
generated_images = {}

for gs in guidance_scales:
    if use_pipe:
        generator = torch.Generator(device=device).manual_seed(SEED)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=gs,
            generator=generator,
            num_inference_steps=30
        ).images[0]
        generated_images[gs] = np.array(image)
    else:
        # 합성 대체
        np.random.seed(SEED)
        generated_images[gs] = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

# TODO: 4개 이미지를 subplot으로 시각화
# TODO: 각 이미지 제목에 guidance_scale 표시
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, gs in enumerate(guidance_scales):
    axes[idx].imshow(generated_images[gs])
    axes[idx].set_title(f'guidance_scale={gs}')
    axes[idx].axis('off')
plt.suptitle('Guidance Scale 비교: 동일 프롬프트, 동일 Seed')
plt.tight_layout()
plt.show()
```

---

# Defect Generation -- 프롬프트로 결함을 설계한다

---

## 2-1. 문제 제기

### 섹션 1의 한계

- 섹션 1 프롬프트: **"스크래치 결함 이미지"**
- 결과: 무작위 스크래치 생성 -- **어디에, 얼마나 깊게, 어떤 방향인지 제어 불가**
- 그냥 "스크래치가 들어 있는 이미지"가 나올 뿐

### 현장에서 필요한 것

품질 검사 모델을 학습시키려면 **특정 조건의 결함을 골고루 모아야** 합니다.

- 스크래치 길이: 1mm, 5mm, 10mm 골고루 필요
- 깊이: 표면 긁힘부터 깊은 홈까지 다양
- 방향: 수직, 수평, 대각선 골고루

**현실**: 현장 데이터는 편향되어 있음

- 라인 특성상 수직 스크래치가 많고, 깊은 스크래치는 거의 없음
- 모델이 편향된 데이터로만 학습하면 본 적 없는 형태에는 약함

### 해결책: 부족한 조건의 결함을 정확한 명세로 생성

```
"스테인리스 스틸 표면 왼쪽 가장자리에
 깊이 5mm의 수직 스크래치 결함,
 산업용 품질 검사 사진, 고해상도"
```

텍스트 한 줄로 결함의 모든 속성을 지정하고, 그대로 이미지가 만들어진다면 **데이터셋의 빈 구멍을 정확히 채울 수 있음**.

### 핵심 질문

> 생성된 결함을 우리가 원하는 대로 정밀하게 제어할 수 있는가?
> 그 답이 **프롬프트 기반 조건부 생성**입니다.

---

## 2-2. 이론

### STEP 01 -- 조건부 생성의 직관: CGAN에서 Diffusion으로

> *Generative Deep Learning* 2판 Ch.4, Ch.13 -- 조건부 생성의 발전 흐름

**CGAN(Conditional GAN, 2014)** 은 GAN에 **조건 $c$**(클래스 레이블)를 추가한 것입니다. Generator 입력에 노이즈 $z$와 레이블 $c$가 함께 들어가, 예컨대 $c$를 "스크래치"로 주면 스크래치 이미지를 생성합니다. 다만 조건이 클래스 레이블 수준이라 "왼쪽에 5mm 깊이의 수직 스크래치" 같은 세밀한 표현은 **불가능** 하다는 한계가 있습니다.

```{mermaid}
flowchart LR
    A["노이즈 z +<br/>클래스 레이블 c"] --> B["Generator"]
    B --> C["조건에 맞는 이미지"]
    C --> D["한계: 클래스 레이블만<br/>세밀한 제어 불가"]
```

**Text-to-Image Diffusion(2022~)** 은 조건을 클래스 레이블에서 **자연어 문장 전체** 로 확장했습니다. Stable Diffusion, DALL-E 2, Imagen 등이 이 흐름의 산물이며, 자연어의 무한한 표현력 덕분에 사실상 무한한 제어가 가능해졌습니다.

```{mermaid}
flowchart LR
    A["노이즈 z +<br/>텍스트 문장 c"] --> B["Diffusion"]
    B --> C["텍스트에 맞는 이미지"]
    C --> D["강점: 언어의 유연성으로<br/>무한한 조건 표현 가능"]
```

```{admonition} 핵심
:class: important

조건부 생성이라는 아이디어는 같고, **조건을 표현하는 방법이 클래스에서 자연어로 진화**한 것.
핵심 아이디어는 변하지 않음.
```

GAN에는 **Mode Collapse** 라는 고질적 한계가 있습니다(*Generative Deep Learning* 4장). Generator가 Discriminator를 속이기 가장 쉬운 한두 가지 패턴에만 빠져 같은 이미지만 반복 생성하는 현상입니다. Diffusion은 학습 방식이 달라 mode collapse가 훨씬 덜 일어나며, 이것이 Diffusion이 GAN을 대체하는 흐름이 강해진 이유입니다.

```{figure} ../../lecture/images/s4_2_img02.png
:alt: Figure 4-2. Inputs and outputs of the two networks in a GAN
:width: 67%

Figure 4-2. Inputs and outputs of the two networks in a GAN
```

- GAN의 두 네트워크(**Generator·Discriminator**)의 입출력 관계를 보여줌
- 생성자는 가짜를 만들고 판별자는 진짜/가짜를 가름 → 적대적 학습
- 조건부 생성(CGAN)의 출발이 되는 기본 구조

```{figure} ../../lecture/images/s4_2_img06.png
:alt: Figure 4-5. Training the DCGAN -- 회색 박스는 동결된 가중치
:width: 67%

Figure 4-5. Training the DCGAN -- 회색 박스는 동결된 가중치
```

- DCGAN 학습 과정 — 회색 박스는 그 단계에서 **동결된(업데이트 안 하는) 가중치**
- 생성자와 판별자를 번갈아 학습시키는 GAN 훈련 방식
- GAN 학습이 까다롭고 불안정할 수 있음을 시사

```{figure} ../../lecture/images/s4_2_img16.png
:alt: Figure 4-16. Inputs and outputs of the generator and critic in a CGAN
:width: 67%

Figure 4-16. Inputs and outputs of the generator and critic in a CGAN
```

- **조건(레이블)** 을 입력에 추가한 CGAN의 생성자·비평자 입출력
- 노이즈 + 클래스 레이블 → 해당 클래스의 이미지 생성
- "조건부 생성"의 초기 형태

```{figure} ../../lecture/images/s4_2_img17.png
:alt: Figure 4-17. Output from the CGAN -- 조건 레이블로 생성 제어
:width: 67%

Figure 4-17. Output from the CGAN -- 조건 레이블로 생성 제어
```

- CGAN이 **조건 레이블로 생성을 제어**한 출력 예시
- 같은 모델이 레이블에 따라 다른 종류의 이미지를 냄
- 다만 제어 수준이 클래스 단위에 머무는 한계를 보여줌

---

### STEP 02 -- CLIP: 텍스트와 이미지를 연결하는 다리

> *Generative Deep Learning* Ch.13 / *Hands-On Generative AI* Ch.2

**CLIP(Contrastive Language-Image Pre-training, OpenAI 2021)** 은 인터넷에서 수집한 수억 개의 (이미지, 텍스트 캡션) 쌍으로 학습합니다. 학습 방식은 **대조 학습(Contrastive Learning)** 으로, 한 배치에 들어온 $N$쌍에 대해 $N \times N$개의 모든 조합의 유사도를 계산한 뒤 **정답 쌍의 유사도는 높이고 나머지는 낮추는 방향** 으로 학습합니다. 그 결과 **텍스트와 이미지를 같은 벡터 공간에 놓는 능력** 을 얻습니다. 그래서 "a scratch on metal", "a bubble defect" 같은 문구가 각각 해당 이미지의 벡터와 가까운 위치에 놓이게 되어, 비슷한 의미일수록 벡터 공간에서도 가까이 자리합니다.

```{mermaid}
flowchart TD
    A["텍스트 프롬프트"] --> B["CLIP Text Encoder"]
    B --> C["텍스트 임베딩 벡터<br/>(의미를 숫자로 표현)"]
    C --> D["Diffusion<br/>(Cross-Attention)"]
    D --> E["프롬프트 의미에 맞는 이미지 생성"]
```

Stable Diffusion에서 CLIP은 텍스트와 이미지를 잇는 다리 역할을 합니다. 프롬프트가 입력되면 CLIP이 벡터로 변환하고, 이 벡터가 Diffusion의 노이즈 제거 과정에 **조건으로 주입** 됩니다. 즉 노이즈를 제거할 때마다 "이 텍스트 벡터에 가까워지는 방향으로 가라"는 가이드가 작동하는 셈입니다.

```{figure} ../../lecture/images/s4_2_img20.png
:alt: Figure 13-4. The CLIP training process -- 대조 학습으로 텍스트-이미지 연결
:width: 67%

Figure 13-4. The CLIP training process -- 대조 학습으로 텍스트-이미지 연결
```

- 수억 쌍의 (이미지, 텍스트)로 **대조 학습**하는 CLIP 훈련 과정
- 정답 쌍은 가깝게, 오답 쌍은 멀게 → 텍스트·이미지를 같은 공간에 정렬
- 텍스트로 이미지를 제어할 수 있게 하는 핵심 다리

```{figure} ../../lecture/images/s4_2_img36.png
:alt: Figure 13-20. The Stable Diffusion architecture -- 잠재 확산 모델 구조
:width: 67%

Figure 13-20. The Stable Diffusion architecture -- 잠재 확산 모델 구조
```

- **잠재 확산(latent diffusion)** 기반 Stable Diffusion의 전체 구조
- 텍스트(CLIP) 조건이 U-Net의 노이즈 제거에 **cross-attention**으로 주입됨
- 프롬프트 → 이미지가 실제로 어떻게 연결되는지 보여줌

```{figure} ../../lecture/images/s4_2_img13.png
:alt: Figure 13-1. An example of text-to-image generation by DALL-E 2
:width: 67%

Figure 13-1. An example of text-to-image generation by DALL-E 2
```

- DALL-E 2의 **텍스트→이미지 생성** 결과 예시
- 자연어 문장만으로 새 이미지를 만들어내는 능력을 보여줌
- 조건이 클래스에서 자연어로 진화했음을 실증

```{figure} ../../lecture/images/s4_2_img26.png
:alt: Figure 13-11. The GLIDE diffusion process -- 텍스트 임베딩으로 U-Net 가이드
:width: 67%

Figure 13-11. The GLIDE diffusion process -- 텍스트 임베딩으로 U-Net 가이드
```

- GLIDE에서 **텍스트 임베딩으로 U-Net을 가이드**하는 확산 과정
- 노이즈 제거 매 단계마다 프롬프트 의미 쪽으로 끌어당김
- 텍스트 조건이 생성에 주입되는 메커니즘의 구체 사례

**CLIP 임베딩 확인 코드**

```python
from transformers import CLIPTextModel, CLIPTokenizer

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
    print(f"'{prompt[:30]}...' -> shape: {embeddings.shape}")
    # 결과: [1, 토큰 수, 512] -> 각 토큰별 512차원 벡터
```

임베딩 shape이 `[1, 토큰 수, 512]` 라는 것은 각 토큰마다 512차원 벡터가 만들어진다는 뜻입니다. 예를 들어 "scratch defect on metal surface"가 7~8개 토큰으로 쪼개지면 토큰별로 의미 벡터가 생성되고, 이 벡터들이 Diffusion의 **cross-attention 메커니즘** 을 통해 이미지 생성에 영향을 줍니다.

CLIP은 학습 데이터에 없는 단어 조합도 잘 처리하는 **zero-shot 일반화** 능력을 갖습니다. "스테인리스 스틸 표면의 깊은 수직 스크래치" 같은 조합이 학습 데이터에 그대로 있을 리 없지만, CLIP은 각 단어의 의미를 조합해 합리적인 벡터를 만들어 냅니다. 이것이 자유로운 프롬프트로 결함을 설계할 수 있는 근거입니다.

---

### STEP 03 -- 제조 결함 프롬프트 설계 원칙

> *Generative Deep Learning* Ch.13 -- 효과적인 텍스트 프롬프트 설계

### 프롬프트 설계 공식

효과적인 프롬프트는 보통 **[결함 유형] + [위치] + [심각도] + [재질/배경] + [촬영 스타일]** 의 구성을 따릅니다.

**예시 분해**

```
"a deep vertical scratch defect on the left edge
 of a stainless steel surface,
 industrial quality inspection photo,
 high resolution, studio lighting"
```

위 예시를 뜯어 보면 결함 유형은 "deep vertical scratch defect", 위치는 "on the left edge", 재질은 "stainless steel surface", 촬영 스타일은 "industrial quality inspection photo, high resolution, studio lighting"에 해당합니다.

### 좋은 프롬프트 vs 나쁜 프롬프트

| 나쁜 프롬프트 | 좋은 프롬프트 |
|------------|-------------|
| `"defect"` (모호, 결과가 들쭉날쭉) | `"hairline crack defect, 2cm length, diagonal orientation, on white ceramic surface, macro photography"` (구체적) |
| `"very very bad defect extremely damaged"` (강조 반복은 효과 없음) | `"severe deep crack defect, multiple branching lines"` (구체적 묘사) |

**첫 번째 함정**: "defect" 한 단어는 CLIP이 무수히 많은 가능성 중 무작위로 하나를 골라야 해서 결과가 불안정

**두 번째 함정**: 강조를 반복한다고 효과가 커지지 않음. "very very bad extremely damaged"는 CLIP 입장에서 비슷한 의미의 중복 노이즈. 구체적인 단어가 훨씬 효과적

### 네거티브 프롬프트

원하지 않는 요소를 명시하는 강력한 기법:

```python
positive_prompt = "scratch defect on metal surface, inspection photo"
negative_prompt = "blurry, low quality, cartoon, illustration, text, watermark"

image = pipeline(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5    # 프롬프트 충실도
).images[0]
```

네거티브 프롬프트는 "이런 요소는 빼 달라"고 지정하는 기법으로, 흐릿함·만화풍·일러스트·텍스트·워터마크 등을 제외할 수 있습니다. 다만 너무 많은 단어를 넣으면 오히려 모델에 혼란을 주므로 **5~7개 키워드가 적당** 합니다(*Generative Deep Learning* 13장).

### guidance_scale 매개변수

guidance_scale은 "프롬프트를 얼마나 강하게 반영할지"를 정하는 다이얼입니다. **낮으면** 자유롭고 다양하지만 프롬프트와 달라질 수 있고, **높으면** 충실하지만 부자연스러워질 수 있습니다. 권장 기본값은 **7.5** 입니다.

```{admonition} guidance_scale 요약
:class: tip

- 낮으면 자유로움, 높으면 충실함
- 7.5가 기본 권장값
- 실습에서 직접 바꿔 보면서 효과 체감
```

---

# 세션 4 실습 — Claude Code 시연 & 실습 모음


> 세션 4(Generative AI 합성 데이터)의 각 강의 섹션에 있던 **Claude Code 시연**과 **실습** 부분을 한곳에 모았습니다.
> 각 항목의 이론·배경 설명은 해당 강의 섹션을 참고하세요.


---

```{admonition} 출처: 섹션 1 · Diffusion Model
:class: note dropdown
아래 시연/실습은 원래 **섹션 1 · Diffusion Model** 섹션에 있던 내용입니다.
```

## 1-3. Claude Code 시연

### 시연 목표

코드 작성보다 **생성된 이미지의 품질을 평가하는 흐름**을 보여주는 것에 집중합니다.

**Claude에게 던질 프롬프트**

```
Stable Diffusion으로 제조 결함 이미지를 생성하고 품질을 평가해줘.
- diffusers 라이브러리로 파이프라인 구성
- 결함 유형 3가지 프롬프트:
  "scratch defect on metal surface, industrial quality inspection"
  "bubble defect on painted surface, factory inspection photo"
  "crack defect on ceramic component, quality control"
- num_inference_steps=50, 유형당 3장
- 결과: 3x3 그리드 시각화
- 생성 이미지와 합성 실제 결함의 SSIM 계산
- GPU 없으면 CPU fallback
```

### 시연 흐름

1. **파이프라인 로드**: `StableDiffusionPipeline.from_pretrained`로 모델 로드. `torch_dtype=torch.float16`은 메모리 절약용 반정밀도 옵션 (GPU에서만 의미)
2. **결함 유형 3가지 생성**: 3개 프롬프트 x 3장 = 총 9장. `num_inference_steps=50`은 역방향 노이즈 제거를 50번 반복 (DDIM으로 50번이면 충분)
3. **3x3 그리드 시각화**: 같은 행 = 같은 결함 유형, 같은 열 = 같은 생성 인덱스. 시드가 다르면 결과가 미묘하게 다름
4. **SSIM 평가**: 시연에서는 합성 노이즈 이미지를 "실제 결함"으로 가정하고 비교

### 시연 후 토론: Mode Collapse

- **정의**: 모델이 학습 데이터의 일부 패턴에만 빠져서 다양성을 잃어버리는 현상 (*Generative Deep Learning* 4장)
- Diffusion에서도 fine-tuning 데이터가 너무 좁으면 비슷한 문제 발생 가능

**원인**

- 학습 데이터 다양성 부족
- 학습률 설정 실수
- 모델 용량 부족

**해결책**

- 데이터 증강 충분히 적용
- 프롬프트 다양화로 무작위성 주입
- LoRA 가중치 조정

```{admonition} 한계 인지
:class: warning

생성 이미지가 100% 완벽하지는 않음. 텍스트가 흐릿하거나 결함 형태가 미묘하게 부자연스러울 수 있음.
이런 한계는 섹션 2의 **프롬프트 제어**와 섹션 3의 **합성 데이터 검증**에서 보완.
```

---

## 1-4. 실습

### 과제: 추론 스텝 수가 생성 속도와 품질에 미치는 영향

| 실험 | num_inference_steps | 관찰 포인트 |
|------|-------------------|------------|
| A | 20 | 빠른 샘플링. 품질은? |
| B | 50 | 기본값 |
| C | 100 | 느린 샘플링. 더 좋아지는가? |

**성공 체크리스트**

- 100스텝이 20스텝보다 3~5배 느리면 시간 측정이 정상
- 시각적으로 100스텝이 가장 깔끔하면 추론 스텝 효과가 정상

**추가 도전 과제**

1. **DDIM 스케줄러**로 교체해서 같은 50스텝에서 품질 변화 확인
2. **시드 고정**으로 같은 스텝에서 항상 같은 이미지가 나오는지 확인 (재현성 측면에서 중요)

**실습 시작 코드**

```python
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from skimage.metrics import structural_similarity as ssim

# GPU/CPU 자동 감지
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 파이프라인 로드
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe",
        torch_dtype=dtype
    ).to(device)
except Exception as e:
    print(f"모델 로드 실패: {e}")
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
        elapsed = steps * 0.01
        img_array = np.random.rand(512, 512, 3)

    real_defect = np.random.rand(512, 512, 3)
    real_gray = np.mean(real_defect, axis=2)
    gen_gray = np.mean(img_array, axis=2)
    ssim_score = ssim(real_gray, gen_gray, data_range=1.0)

    results[steps] = {'image': img_array, 'time': elapsed, 'ssim': ssim_score}
    print(f"steps={steps}: {elapsed:.1f}초, SSIM={ssim_score:.3f}")

# TODO: 결과 비교 시각화 (subplot 3개 + 비교 표)
```

**제출 항목**

- 3가지 steps 설정의 생성 이미지 비교 (각 설정당 1장씩, subplot)
- 생성 시간 + SSIM 비교 표
- 실시간 생성 환경에서의 steps 선택 근거 (한 문단)

---

```{admonition} 출처: 섹션 2 · Defect Generation
:class: note dropdown
아래 시연/실습은 원래 **섹션 2 · Defect Generation** 섹션에 있던 내용입니다.
```

## 2-3. Claude Code 시연

### 시연 목표

**프롬프트 설계가 생성 결과에 미치는 영향을 비교**하는 흐름에 집중

**Claude에게 던질 프롬프트**

```
제조 결함 유형별 프롬프트 템플릿을 설계하고 이미지를 생성해줘.

결함 유형별 프롬프트 (각각 3가지 강도):
1. 스크래치: 경미 / 중간 / 심각
2. 기포: 소형 / 중형 / 대형
3. 크랙: 표면 / 깊은 / 관통

- diffusers 파이프라인으로 각 프롬프트 생성
- 결과: 3x3 그리드 (행=결함유형, 열=심각도)
- guidance_scale=7.5 고정
- guidance_scale을 3 / 7.5 / 15로 바꾼 비교도 추가
```

### 시연 흐름

1. **프롬프트 템플릿 설계**: 결함 유형 x 심각도를 조합해서 자동으로 프롬프트를 만드는 함수

```python
def build_prompt(defect_type, severity, material="metal"):
    severity_map = {
        "light": "minor surface",
        "medium": "moderate",
        "heavy": "severe deep"
    }
    return (
        f"{severity_map[severity]} {defect_type} defect "
        f"on {material} surface, "
        f"industrial quality inspection photo, "
        f"high resolution, neutral background"
    )
```

2. **이미지 생성 루프**: 9가지 조합 각각에 대해 한 장씩 생성. 시드를 고정해서 차이를 명확히 비교

3. **3x3 그리드 시각화**: 같은 행 = 같은 결함 유형(심각도 변화), 다른 행 = 결함 유형 자체가 다름

4. **guidance_scale 비교 실험**: 같은 프롬프트, 같은 시드, 다른 guidance_scale 3가지 (3, 7.5, 15)
   - **gs=3**: 가장 다양하지만 프롬프트와 느슨하게 연결
   - **gs=7.5**: 균형점. 프롬프트도 잘 반영하고 자연스러움
   - **gs=15**: 너무 강하게 반영해서 **과포화**나 **아티팩트** 발생

```{admonition} guidance_scale 핵심
:class: important

너무 낮아도 안 되고 너무 높아도 안 됨. 7.5 근처에 sweet spot이 있음.
```

### 시연 후 토론: 재현성

**질문**: 동일한 프롬프트인데 결과가 매번 다른 이유는? 재현 가능한 결과를 얻으려면?

- Diffusion은 **시작 시점의 노이즈가 무작위**이기 때문에 매번 다른 결과
- 시드를 고정하면 같은 노이즈에서 시작해서 같은 결과

```python
import torch
generator = torch.Generator(device="cuda").manual_seed(42)
image = pipe(prompt, generator=generator).images[0]
```

- 실무에서는 **데이터셋 재현성을 위해 시드를 항상 기록**
- 어떤 결함 이미지가 어떤 시드에서 나왔는지 메타데이터로 함께 저장하면 나중에 같은 이미지를 재생성 가능

---

## 2-4. 실습

### 과제: guidance_scale 변화에 따른 트레이드오프 체험

| 실험 | guidance_scale | 특성 | 관찰 포인트 |
|------|---------------|------|------------|
| A | 3 | 프롬프트 약하게 반영 | 다양하지만 의도와 다를 수 있음 |
| B | 7.5 | 기본값 (권장) | 균형 |
| C | 12 | 프롬프트 강하게 반영 | 충실하지만 부자연스러울 수 있음 |
| D | 20 | 과도하게 반영 | 과포화, 아티팩트 발생 |

**성공 체크리스트**

1. gs=3은 다양하지만 프롬프트와 다소 동떨어짐
2. gs=7.5가 가장 균형 잡힌 결과
3. gs=20에서 과포화나 아티팩트가 보이면 실험이 의도대로 동작한 것

**실습 시작 코드**

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    use_pipe = True
except:
    use_pipe = False
    print("파이프라인 없음: 합성 이미지로 대체")

prompt = ("scratch defect on stainless steel surface, "
          "industrial quality inspection photo, high resolution")
negative_prompt = "blurry, low quality, cartoon, text, watermark"

SEED = 42
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
        np.random.seed(SEED)
        generated_images[gs] = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, gs in enumerate(guidance_scales):
    axes[idx].imshow(generated_images[gs])
    axes[idx].set_title(f'guidance_scale={gs}')
    axes[idx].axis('off')
plt.suptitle('Guidance Scale 비교: 동일 프롬프트, 동일 Seed')
plt.tight_layout()
plt.show()
```

**추가 도전 과제**

1. **네거티브 프롬프트 효과 검증**: 비워서 생성한 결과와 5~7개 키워드를 넣어서 생성한 결과를 같은 시드로 비교
2. **프롬프트 가중치**: `(scratch:1.5)` 같은 구문으로 특정 단어를 강조하는 기법 (*Generative Deep Learning* 13장, *Hands-On Generative AI* 7장)

**제출 항목**

- 4가지 guidance_scale의 생성 이미지 비교 (동일 seed, subplot)
- **결함 데이터 생성 목적에서 어떤 guidance_scale을 선택할지, 그 이유를 한 문단으로 정리**
  - 단순히 숫자를 고르는 것이 아니라 **데이터 다양성과 프롬프트 충실도 사이의 우선순위**를 본인이 판단

---

```{admonition} 출처: 섹션 3 · 합성 데이터 고도화
:class: note dropdown
아래 시연/실습은 원래 **섹션 3 · 합성 데이터 고도화** 섹션에 있던 내용입니다.
```

## 3-2. Claude Code 시연

### 시연 목표

코드를 만드는 것 자체가 아님. **나온 결과를 보고 "왜 이런 결과가 나왔는지" Claude와 대화하는 흐름**이 진짜.

**Claude에게 던질 프롬프트**

```
합성 데이터 추가 효과를 비교하는 실험 코드를 만들어줘.

실험 설정:
- 실제 결함 이미지: 50장 (numpy로 합성)
- 합성 결함 이미지: 200장 (품질이 약간 낮은 버전으로 시뮬레이션)
- 분류 모델: ResNet18 fine-tuning
- 4가지 데이터 구성: 실제만 / 1:1 / 1:3 / 합성만

결과 시각화:
1. 4가지 모델의 Confusion Matrix (2x2 subplot)
2. F1 / Recall / Precision 비교 막대그래프
3. 합성 비율 vs 각 지표 꺾은선 그래프

GPU 없는 환경에서도 실행 가능하게 (간단한 CNN으로 대체)
```

### 시연 흐름

**1. 데이터셋 시뮬레이션**

- 실제 결함: 일정한 패턴에 약간의 노이즈
- 합성 결함: 같은 패턴이지만 조금 더 단순하고 균일
- 이 차이가 모델 학습에 어떻게 영향을 주는지가 관찰 대상

**2. 모델 학습 함수**

- ResNet18 또는 간단한 CNN을 fine-tuning
- 같은 학습 설정으로 4번 학습시키고 각각의 결과를 저장
- 결정적 요인이 **데이터 구성**이라는 점을 명확히 하기 위해 다른 변수는 모두 고정

**3. 평가**

- 각 모델을 동일한 실제 테스트셋 20장으로 평가
- Confusion Matrix, F1, Recall, Precision 계산

### 시연 결과 분석

**가상 결과 예시**

| 구성 | F1 | Recall | Precision |
|------|-----|--------|-----------|
| 실제만 | 0.72 | 0.65 | 0.81 |
| 1:1 | 0.78 | 0.82 | 0.75 |
| 1:3 | 0.74 | 0.88 | 0.64 |
| 합성만 | 0.55 | 0.71 | 0.45 |

(실제 숫자는 머신마다 다름. 핵심은 패턴)

**가장 주목할 패턴**: 합성 데이터를 더할수록 **Recall은 올라가는데 Precision은 내려감**

```{mermaid}
flowchart LR
    A["합성 데이터 증가"] --> B["Recall 상승"]
    A --> C["Precision 하락"]
    B --> D["더 많은 결함 패턴 학습"]
    C --> E["합성 특유의 부수 특징까지<br/>결함 신호로 학습"]
```

### Claude에게 결과 해석 요청

```
합성 데이터를 추가했더니 Recall은 올라갔는데 Precision은 내려갔어.
왜 이런 일이 생기는지 설명해 주고,
이 상황에서 어떤 지표를 더 믿어야 하는지 알려줘.
```

**Recall이 올라간 이유**

- 합성 데이터가 모델에게 더 많고 다양한 결함 패턴을 보여 줌
- 모델이 결함을 잘 찾아내는 능력이 올라감
- "이런 형태도 결함이다"는 폭이 넓어짐

**Precision이 내려간 이유**

- 합성 결함과 실제 결함 사이에 미묘한 차이가 있고, 모델이 그 차이까지 "결함의 신호"로 학습
- 정상 이미지에서 합성 결함 비슷한 부수 특징이 보이면 결함이라고 잘못 판정
- 오탐이 늘어남

**어떤 지표를 믿을지는 응용 분야에 따라 다름**

```{admonition} 제조업에서는 Recall이 압도적으로 중요
:class: important

- 결함을 놓치면 불량품이 출하되거나 안전 사고로 이어짐
- 반면 오탐은 인력으로 재검사하면 됨 (비용은 들지만 회복 가능)
- Session 2에서도 강조한 원칙: **제조 안전에서는 Recall 최우선**
```

### 시연 결론

- **1:1 혼합이 최적에 가까움**: F1과 Recall이 둘 다 가장 좋고, Precision도 너무 떨어지지 않음
- **1:3**: Recall은 더 좋지만 Precision이 너무 낮아져서 알람 피로 우려
- **합성만**: 명백히 안 됨

이 결론은 시드와 데이터에 따라 다를 수 있으므로 **본인이 직접 돌려서 비슷한 패턴이 보이는지 확인**하는 것이 중요.

---

## 3-3. 자유 실습

세 가지 난이도로 준비. **하나를 깊게** 파는 것을 추천.

### 기본 -- 합성 데이터 비율 최적점 찾기

"그러면 최적 비율은 정확히 얼마인가?" 시연에서는 4가지였지만, 더 촘촘히 봐야 정확한 sweet spot을 찾을 수 있음.

**과제**

1. 합성 데이터 비율을 **0%, 25%, 50%, 75%, 100%** 다섯 가지로 변화시키며 모델 학습
2. 각 비율에서 F1-Score, Recall, Precision을 기록
3. 비율 vs 각 지표의 꺾은선 그래프를 그림

**힌트**

- 비율이 올라갈수록 Recall은 단조 증가하다가 어느 시점부터 꺾임
- Precision은 단조 감소
- F1은 둘의 균형이라 **종 모양 곡선**이 나올 가능성이 큼
- 최적점은 보통 25~50% 사이에 있음

**제출**: 비율별 성능 표 + 최적 비율과 그 이유를 한 문단으로

### 중급 -- 합성 이미지 품질 필터링

모든 합성 이미지가 동일하게 가치 있지는 않음. 품질이 낮은 합성을 걸러내면 같은 양으로 더 좋은 성능을 얻을 수 있을까?

**과제**

1. 합성 이미지 200장 전체에 대해 SSIM을 계산 (실제 결함 한 장과의 유사도)
2. 세 가지 필터링을 비교:
   - 전체 사용 (필터링 없음)
   - SSIM > 0.5만 사용
   - SSIM > 0.7만 사용
3. 각각의 데이터로 학습해서 성능을 비교

**제출**: SSIM 임계값별 사용 이미지 수 + 성능 표 + 한 문단의 결론

**Claude에게 추가 질문**: "FID와 SSIM 중 어떤 지표가 학습 성능을 더 잘 예측하는가?"

- FID는 분포 수준 평가라 학습 일반화와 더 잘 연결될 수 있음
- SSIM은 개별 이미지 평가라 노이즈 필터링에 더 적합할 수 있음
- 정답이 없는 질문. Claude의 답을 본인의 실험 결과와 비교하는 것이 핵심

### 심화 -- 결함 유형별 합성 효과 차이

합성이 모든 결함 유형에 동등하게 효과적인가?

- 스크래치: 단순한 직선 패턴 -> 합성이 쉬움
- 크랙: 가지 모양 -> 합성이 더 어려움
- 가설: **합성 품질이 높은 유형에서 학습 효과도 더 좋다**

**과제**

1. 스크래치, 버블, 크랙 세 가지 결함 유형 각각에 대해 **실제만 학습 vs 실제+합성 학습**의 Recall 개선폭을 측정
2. 각 유형의 평균 SSIM(생성 품질)도 측정
3. **SSIM(x축) vs Recall 개선폭(y축) 산점도**를 그림

**제출**: 산점도 + 상관관계 해석 한 문단

- 가설이 맞다면 양의 상관관계가 나올 것
- 안 맞다면 왜 그런지 추가 분석 (결함 유형의 본질적 복잡도? 데이터 양? 다른 요인?)

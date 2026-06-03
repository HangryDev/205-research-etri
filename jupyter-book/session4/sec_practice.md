# 세션 4 실습 — Claude Code 시연 & 실습 모음


> 세션 4(Generative AI 합성 데이터)의 각 강의 섹션에 흩어져 있던 **Claude Code 시연**과 **실습** 부분을 한곳에 모았습니다.
> 각 항목의 이론·배경 설명은 해당 강의 섹션을 참고하세요.


---

```{admonition} 출처: 섹션 1 · Diffusion Model
:class: note dropdown
아래 시연/실습은 원래 **섹션 1 · Diffusion Model** 섹션에 있던 내용입니다.
```

## 1-3. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: Fine-tuning 학습 과정이 아닌 **생성 결과의 품질을 평가하는 방법**에 집중하세요.
```

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

## 1-4. 실습

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

```{admonition} 출처: 섹션 2 · Defect Generation
:class: note dropdown
아래 시연/실습은 원래 **섹션 2 · Defect Generation** 섹션에 있던 내용입니다.
```

## 2-3. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: 프롬프트 설계가 생성 결과에 미치는 영향을 **직접 비교**하는 과정에 집중하세요.
```

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

## 2-4. 실습

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

```{admonition} 출처: 섹션 3 · 합성 데이터 고도화
:class: note dropdown
아래 시연/실습은 원래 **섹션 3 · 합성 데이터 고도화** 섹션에 있던 내용입니다.
```

## 3-2. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: 코드 작성보다 **결과 수치를 보고 "왜 이런 결과가 나왔는가"를 Claude에게 질문하는 과정**에 집중하세요.
```

**Claude에게 던질 프롬프트 예시**:
```
합성 데이터 추가 효과를 비교하는 실험 코드를 만들어줘.

실험 설정:
- 실제 결함 이미지: 50장 (합성 numpy로 생성)
- 합성 결함 이미지: 200장 (품질이 약간 낮은 버전으로 시뮬레이션)
- 분류 모델: ResNet18 (fine-tuning)
- 4가지 데이터 구성: 실제만 / 1:1 / 1:3 / 합성만

결과 시각화:
1. 4가지 모델의 Confusion Matrix (2×2 subplot)
2. F1 / Recall / Precision 비교 막대그래프
3. 합성 데이터 비율 vs 각 지표 꺾은선 그래프

GPU 없는 환경에서도 실행 가능하게 (간단한 CNN으로 대체 가능)
```

**시연 후 Claude에게 결과 해석 요청**:
> *"합성 데이터를 추가했더니 Recall은 올라갔는데 Precision이 내려갔어.
> 왜 이런 일이 생기는지 설명해줘.
> 그리고 이 상황에서 어떤 지표를 더 믿어야 할까?"*

**시연 흐름**:
1. 4가지 데이터셋 구성
2. 각 구성으로 모델 학습
3. Confusion Matrix 시각화
4. 지표 비교 그래프 출력
5. Claude와 결과 해석 대화

---

## 3-3. 자유 실습

### ★ 기본 — 합성 데이터 비율 실험
```
목표: 합성 데이터 비율 변화가 F1-Score에 미치는 영향 확인

합성 데이터 비율: 0% / 25% / 50% / 75% / 100%
각 비율로 모델 학습 → F1-Score, Recall, Precision 기록

제출:
- 비율별 성능 표
- "어떤 비율에서 최적 성능이 나왔는가?" 한 문단
```

### ★★ 중급 — 합성 이미지 품질 필터링 효과
```
목표: SSIM 임계값으로 품질 낮은 합성 이미지를 걸러내면 성능이 개선되는지 확인

실험:
① 합성 이미지 전체 사용 → 성능 기록
② SSIM > 0.5인 이미지만 사용 → 성능 기록
③ SSIM > 0.7인 이미지만 사용 → 성능 기록

제출:
- SSIM 임계값별 사용 이미지 수 + 성능 표
- "품질 필터링이 실제로 효과가 있었는가?" 한 문단
- Claude에게 질문: "FID와 SSIM 중 어떤 지표가 학습 성능을 더 잘 예측하는가?"
```

### ★★★ 심화 — 결함 유형별 합성 효과 차이 분석
```
목표: 어떤 결함 유형에서 합성 데이터가 더 효과적인지 파악

실험:
① 스크래치 결함: 실제만 vs 실제 + 합성 → 유형별 Recall 비교
② 기포 결함: 동일 실험
③ 크랙 결함: 동일 실험

분석 질문:
"생성 품질(SSIM)이 높은 결함 유형에서 학습 효과도 더 좋은가?"
→ SSIM과 Recall 개선폭의 상관관계를 산점도로 시각화

제출:
- 결함 유형별 SSIM(생성 품질) + Recall 개선폭 산점도
- 상관관계 해석 한 문단
```

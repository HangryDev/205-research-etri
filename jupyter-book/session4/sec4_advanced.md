# 심화 학습 자료

> 강의에서 시간 관계상 다루지 못한 심화 내용입니다.
> 강의 후 궁금증이 생겼을 때 아래 순서로 탐색하시길 권장합니다.

---

## 📘 섹션 1 심화: Diffusion 구조와 Fine-tuning 심화

:::{admonition} 참고 교재
:class: note
📖 **교재**: *Hands-On Generative AI with Transformers and Diffusion Models* (Omar Sanseviero et al., O'Reilly)
**심화 챕터 로드맵**:

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.3** Diving Into Diffusion Models | Forward/Reverse Process 수학적 이해 | Diffusion 원리 깊이 이해 |
| 2 | **Ch.4** Transformer-Based Diffusion Models | U-Net, Cross-Attention, DDPM vs DDIM | 구조 심화 |
| 3 | **Ch.6** Fine-Tuning Language Models | LoRA, DreamBooth, Textual Inversion 비교 | Fine-tuning 전략 선택 |
:::

### Diffusion의 수학적 원리 (심화)

```
[Forward Process: 노이즈 추가 공식]

q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

βₜ: 노이즈 스케줄 (시간 t에 따라 증가)
t=0: 원본 이미지
t=T: 완전한 가우시안 노이즈

[Reverse Process: 모델이 학습하는 것]

p_θ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))

→ 모델(θ)이 노이즈를 예측하고 제거하는 방향을 학습
```

### DDIM: 더 빠른 샘플링

```python
from diffusers import DDIMScheduler

# DDPM: 1000 steps 필요
# DDIM: 50 steps로도 유사한 품질
scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    scheduler=scheduler
)

# 50 steps로 고품질 생성 (DDPM의 1/20 시간)
image = pipe(prompt, num_inference_steps=50).images[0]
```

### Fine-tuning 전략 비교 심화

| 방법 | 필요 데이터 | 학습 시간 | GPU 메모리 | 결과 품질 | 추천 상황 |
|------|-----------|---------|----------|---------|---------|
| DreamBooth | 3~20장 | 30~60분 | 24GB+ | 최고 | 특정 물체 스타일 |
| LoRA | 10~100장 | 10~30분 | 8GB+ | 높음 | 일반 결함 패턴 |
| Textual Inversion | 5~10장 | 1~2시간 | 8GB+ | 중간 | 텍스트 제어 중심 |
| Full Fine-tuning | 1000장+ | 수 시간 | 40GB+ | 최고 | 충분한 자원 있을 때 |

---

## 📘 섹션 2 심화: 고급 프롬프트 제어와 CGAN

:::{admonition} 참고 교재
:class: note
📖 **교재**: *Generative Deep Learning, 2nd Ed.* (David Foster, O'Reilly)
**심화 챕터 로드맵**:

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.4** Generative Adversarial Networks | GAN 기초, CGAN 조건부 생성 원리 | 조건부 생성의 뿌리 이해 |
| 2 | **Ch.13** Multimodal Models | CLIP, 텍스트-이미지 공간, DALL-E | 현대 생성 모델의 구조 |
:::

### ControlNet: 이미지 구조까지 제어

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import numpy as np

# ControlNet: 엣지 맵/포즈/깊이 등을 조건으로 추가 제어
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"  # 엣지 기반 제어
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# 결함 위치를 엣지 맵으로 지정
reference_image = cv2.imread("defect_reference.jpg")
edges = cv2.Canny(reference_image, 100, 200)

# 엣지 구조를 유지하면서 새 결함 생성
image = pipe(
    prompt="scratch defect on metal surface",
    image=edges,
    controlnet_conditioning_scale=0.8
).images[0]
```

### Prompt Engineering 고급 기법

```python
# 1. 프롬프트 가중치: 특정 단어 강조
prompt = "a (deep:1.5) scratch defect on (stainless steel:1.3) surface"
# (단어:가중치): 1.0 기본, 1.5는 50% 강조, 0.5는 약화

# 2. 프롬프트 혼합 (Prompt Blending)
prompt1 = "scratch defect on metal"
prompt2 = "crack defect on metal"
# 두 프롬프트를 0.5:0.5로 혼합하면 두 결함 특성이 혼합된 이미지

# 3. 구조화된 프롬프트 템플릿
def build_defect_prompt(
    defect_type: str,
    severity: str,
    location: str,
    material: str
) -> str:
    return (
        f"{severity} {defect_type} defect "
        f"located at {location} "
        f"on {material} surface, "
        f"industrial quality inspection photograph, "
        f"high resolution macro photography, sharp focus, "
        f"neutral background"
    )

prompt = build_defect_prompt(
    defect_type="scratch",
    severity="deep diagonal",
    location="upper left corner",
    material="brushed aluminum"
)
```

---

## 📘 섹션 3 심화: 합성 데이터 품질 측정

:::{admonition} 참고 교재
:class: note
📖 **교재**: *Practical Synthetic Data Generation* (Khaled El Emam et al., O'Reilly)
**심화 챕터 로드맵**:

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.1** An Introduction to Synthetic Data | 합성 데이터 활용 사례와 위험 | 전체 맥락 파악 |
| 2 | **Ch.6** Measuring the Quality of Synthetic Data | FID, SSIM, 분포 유사도 측정 | 품질 평가 능력 |
:::

### FID (Fréchet Inception Distance) 계산

```python
import torch
import numpy as np
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

def calculate_fid(real_images, fake_images, batch_size=32):
    """
    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^(1/2))
    낮을수록 좋음 (0 = 완벽하게 동일한 분포)
    """
    # Inception v3로 특징 추출
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # 분류 레이어 제거
    model.eval()

    def get_features(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = torch.tensor(images[i:i+batch_size]).float()
            with torch.no_grad():
                feat = model(batch)
            features.append(feat.numpy())
        return np.concatenate(features, axis=0)

    real_feat = get_features(real_images)
    fake_feat = get_features(fake_images)

    mu_r, sigma_r = real_feat.mean(0), np.cov(real_feat, rowvar=False)
    mu_f, sigma_f = fake_feat.mean(0), np.cov(fake_feat, rowvar=False)

    diff = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean.real)
    return fid

# FID 해석 기준
# FID < 10: 매우 좋음 (실제와 거의 구분 불가)
# FID < 50: 양호
# FID > 100: 개선 필요
```

### 합성 데이터 활용 전략 비교

| 전략 | 방법 | 효과 | 주의사항 |
|------|------|------|---------|
| 순수 보강 | 합성 이미지를 실제에 추가 | Recall 향상 | 비율 조절 필요 |
| 품질 필터링 | SSIM > 임계값인 것만 사용 | 노이즈 감소 | 임계값 튜닝 필요 |
| 도메인 적응 | 합성 스타일을 실제에 맞게 조정 | 도메인 갭 감소 | 추가 처리 필요 |
| 혼합 학습 | 배치마다 실제:합성 비율 유지 | 안정적 학습 | 비율 최적화 필요 |

---

## 📚 추천 학습 경로

### Diffusion 모델 심화 (2주) — *Hands-On Generative AI*

| 순서 | 챕터 | 핵심 내용 | 읽어야 하는 이유 |
|------|------|----------|--------------|
| 1 | **Ch.3** | Diffusion Forward/Reverse Process | 수학적 원리 이해 |
| 2 | **Ch.4** | U-Net, Cross-Attention 구조 | 모델 내부 이해 |
| 3 | **Ch.6** | LoRA, DreamBooth Fine-tuning | 실무 fine-tuning 능력 |

### 조건부 생성 심화 (1주) — *Generative Deep Learning, 2nd Ed.*

| 순서 | 챕터 | 핵심 내용 | 읽어야 하는 이유 |
|------|------|----------|--------------|
| 1 | **Ch.4** | CGAN, 조건부 생성 원리 | 조건부 생성의 뿌리 |
| 2 | **Ch.13** | CLIP, 멀티모달 모델 | 텍스트-이미지 연결 원리 |

### 합성 데이터 심화 (1주) — *Practical Synthetic Data Generation*

| 순서 | 챕터 | 핵심 내용 | 읽어야 하는 이유 |
|------|------|----------|--------------|
| 1 | **Ch.1** | 합성 데이터 사례와 위험 | 전략적 판단 능력 |
| 2 | **Ch.6** | FID, SSIM 품질 측정 | 데이터 품질 평가 |

### 실무 적용 프로젝트 아이디어

- [ ] **MVTec AD Dataset**: 실제 산업 결함 데이터셋으로 Diffusion Fine-tuning 실습 ([다운로드](https://www.mvtec.com/company/research/datasets/mvtec-ad))
- [ ] **ControlNet + 결함 엣지맵**: 결함 위치와 형태를 엣지로 지정해서 정밀 생성
- [ ] **합성 데이터 파이프라인 자동화**: 결함 유형·강도·위치 파라미터를 받아 대량 생성하는 스크립트
- [ ] **세션 3 YOLO와 연결**: 합성 결함 이미지로 YOLO 파인튜닝 → mAP 개선 측정

---

*이 문서는 세션 4 강의 자료로 제작되었습니다.*
*질문 및 피드백: Claude Code를 활용해 실습 중 막히는 부분을 언제든지 질문하세요.*
*이전 세션 자료: `session1_manufacturing_data_analysis.md` / `session2_deep_learning_anomaly_detection.md` / `session3_vision_ai_safety_monitoring.md`*

# 출처
## 섹션 1. 텍스트 북 : [Hands-On Generative AI w/ Transformers & Diffusion](https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/)
## 섹션 2. 텍스트 북 : [Generative Deep Learning 2nd](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/)
## 섹션 3. 텍스트 북 : [Practical Synthetic Data Generation](https://www.oreilly.com/library/view/practical-synthetic-data/9781492072737/)

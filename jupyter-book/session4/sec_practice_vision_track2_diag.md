# 실습 배경 -- VISION Track 2와 DIAG

```{admonition} 이 페이지의 목적
:class: note

H5 실습 노트북(`sec_practice_vision_track2_diag.ipynb`)의 **배경 설명**입니다.
어떤 데이터를 쓰는지, 어떤 대회 과제를 재현하는지(다른 트랙 포함), 그 대회 상위 솔루션은 어떻게 풀었는지,
그리고 우리가 쓰는 **DIAG**는 어디서 왔고 왜 이 과제에 적용하는지를 정리합니다.
```

---

## 1. 사용 데이터 -- KSDD2 (제조 표면 결함)

**KSDD2(Kolektor Surface-Defect Dataset 2)** 는 슬로베니아 Kolektor사와 ViCoS 연구실이 공개한 **실제 제조 검사 데이터셋**입니다. 전기 정류자(commutator) 표면을 촬영한 이미지로, 미세한 스크래치·얼룩·균열 같은 결함을 픽셀 단위 마스크로 표시해 두었습니다.

| 항목 | 내용 |
| --- | --- |
| **도메인** | 제조 표면 검사 (전기 정류자) |
| **구성** | 학습 약 2,331장(결함 246장) + 테스트 약 1,004장(결함 110장), 총 약 3,335장 |
| **라벨** | 이미지 단위 정상/결함 + **픽셀 단위 결함 마스크** |
| **특징** | 정상 多·결함 極少의 전형적 **불균형** 구조 |

이 데이터가 우리 과제에 적합한 이유는 분명합니다. **결함이 전체의 10% 안팎**으로 희소해, "결함을 합성으로 보강하면 탐지가 좋아지는가"라는 질문을 곧바로 던질 수 있습니다. 또 픽셀 마스크가 있어 결함 위치를 정확히 다룰 수 있고, 무엇보다 뒤에서 볼 VISION 챌린지의 데이터셋 모음에도 포함된 **표준 제조 벤치마크**입니다.

```{admonition} Session 1과의 연결
:class: tip

Session 1에서 다룬 **클래스 불균형**이 이미지에서 그대로 재현됩니다. SMOTE 같은 픽셀 보간이 통하지 않는 영역에서, 합성 생성이 어떻게 그 빈자리를 메우는지 확인하는 것이 이 실습의 핵심입니다.
```

---

## 2. 대회 -- VISION Challenge (Vision-based Industrial Inspection)

**VISION**은 CVPR·ECCV·ICCV에서 열리는 **산업 검사 전문 워크숍이자 챌린지**입니다(CVPR 2023 시작 → ECCV 2024 → ICCV 2025 → CVPR 2026 예정). 학계와 현장의 간극을 좁히는 것을 목표로, 실제 생산 라인의 까다로운 조건을 담은 데이터로 경연을 엽니다.

대회의 무대인 **VISION Datasets**는 Roboflow에서 수집·정제한 **14개 산업 검사 데이터셋**으로, 총 약 18,000장 이미지와 44종 결함을 담고 있으며(AITEX 직물, BeanTech, KolektorSDD/SDD2 등), 정밀 검출을 위해 폴리곤 마스크가 추가로 주석돼 있습니다.

챌린지는 두 트랙으로 나뉩니다.

| | **Track 1 — Adapt & Detect** | **Track 2 — Data Generation** |
| --- | --- | --- |
| 과제 | 제한된 라벨 + **비라벨 데이터**를 함께 활용해 다양한 산업 이미지에서 결함을 잘 학습 | **합성 데이터를 생성**해 제한된 라벨 상황에서 탐지 성능을 끌어올림 |
| 한 줄 | "적은 라벨로 잘 학습하라" | "데이터를 만들어 성능을 올려라" |
| 접근 | 준지도·자기지도 학습, 의사라벨링 등 | GAN·Diffusion 기반 결함 합성 + 증강 |
| 우리 실습 | — | **이 트랙을 재현** |

우리가 재현하는 것은 **Track 2**입니다. 즉 "제한된 라벨만 있을 때, 합성으로 탐지 성능(AP)을 올릴 수 있는가"가 채점 기준입니다. 이는 우리가 세션 내내 강조한 "합성 데이터로 불균형을 메운다"는 주제와 정확히 일치합니다.

```{admonition} 왜 "대회 과제"로 실습하는가
:class: tip

대회 과제는 **평가 기준이 명확**합니다. "예쁜 그림"이 아니라 "실제 테스트셋에서 AP가 오르는가"로 채점되므로, 합성의 진짜 가치를 객관적으로 확인할 수 있습니다.
```

---

## 3. 상위 솔루션의 방법론과 인사이트 (Track 2)

VISION 2023 Track 2에서 공개된 대표적 상위 솔루션은 **Wei et al., "Diversified and Multi-Class Controllable Industrial Defect Synthesis for Data Augmentation and Transfer" (CVPRW 2023)** 입니다. 핵심은 **DCDGANc**라는 제어형 GAN으로, 결함을 "다양하게" 그리고 "원하는 종류로" 찍어내는 데 초점을 둡니다.

```{mermaid}
flowchart LR
    A["실제 결함 이미지"] -->|"마스크로 정상 배경 제거"| B["결함 콘텐츠만 추출"]
    B --> C["DCDGANc<br/>(제어형 GAN 학습)"]
    C --> D["다양한 형태·다중 클래스<br/>결함 콘텐츠 생성"]
    D -->|"OD-SPADE로 정상 배경에 합성"| E["증강 결함 이미지"]
    E --> F["세그멘테이션 모델 학습 → 성능 향상"]
```

방법을 풀어보면, 먼저 실제 결함 이미지에서 마스크로 정상 배경을 빼내 **결함 콘텐츠만** 분리합니다. 그다음 DCDGANc가 **클래스 공간을 연속적으로 모델링**해, 같은 결함을 조금씩 다른 형태·다른 클래스로 변형하며 대량 생성합니다(학습 안정화를 위해 WGAN-GP 손실 사용). 마지막으로 생성한 결함 콘텐츠를 정상 배경 위에 자연스럽게 합성(OD-SPADE)합니다.

여기서 얻는 인사이트는 세 가지입니다. 첫째, **다양성(diversification)** 이 핵심입니다 — 같은 결함이라도 형태가 다양해야 탐지기가 처음 보는 결함에 강해집니다. 둘째, **클래스 제어 가능성** 이 중요합니다 — 부족한 특정 결함 유형을 골라 채울 수 있어야 데이터셋의 빈 구멍을 정확히 메웁니다. 셋째, 결국 평가는 **다운스트림 세그멘테이션 성능 향상**으로 증명된다는 점입니다.

```{admonition} 정확성에 대한 참고
:class: note

대회의 최종 순위는 공개 범위가 제한적이라, 위 솔루션은 "워크숍 프로시딩에 공개된 대표 상위 솔루션"으로 이해해 주세요. 방법론과 교훈을 살피기에는 충분합니다.
```

---

## 4. DIAG -- 어디서 왔고, 어떤 방식인가

우리 실습이 실제로 사용하는 것은 **DIAG**입니다. **이탈리아 베로나대학교(University of Verona)** 연구진이 발표한 방법으로(*Leveraging Latent Diffusion Models for Training-Free In-Distribution Data Augmentation for Surface Defect Detection*, **CBMI 2024**), 이름은 **D**iffusion-based **I**n-distribution **A**nomaly **G**eneration에서 왔습니다.

DCDGANc가 GAN을 **학습**시키는 무거운 방식이라면, DIAG는 정반대 철학입니다 — **생성 모델을 전혀 학습시키지 않습니다(training-free).**

```{mermaid}
flowchart LR
    A["정상 이미지"] --> D["SDXL Inpainting<br/>(사전학습, 학습 없음)"]
    B["텍스트 프롬프트<br/>(어떤 결함)"] --> D
    C["영역 마스크<br/>(어디에)"] --> D
    D --> E["in-distribution<br/>합성 결함 이미지"]
    E --> F["ResNet-50 학습 → AP 향상"]
```

작동 방식은 우리가 세션에서 배운 **"텍스트(무엇) + 마스크(어디)"** 그대로입니다. 정상 이미지를 두고, 결함을 넣을 영역을 마스크로 지정하고, 텍스트로 결함 종류를 묘사하면, 사전학습된 **SDXL inpainting**이 그 영역에만 결함을 채웁니다. 도메인 전문가가 텍스트·영역으로 가이드하는 **human-in-the-loop** 방식이라 결과를 해석하고 다듬기 쉽습니다.

가장 중요한 특징은 **in-distribution(원 분포 안)** 생성입니다. 배경을 통째로 새로 만드는 게 아니라 정상 이미지 위 지정 영역만 바꾸므로, "현실에 없는 결함"이 될 위험이 줄어듭니다. 그 결과 KSDD2에서 **양성 샘플이 있을 때 AP 약 18%, 없을 때 약 28%** 향상을 보고했고, 코드도 공개돼 있습니다(`intelligolabs/DIAG`).

| 관점 | DCDGANc (상위 솔루션) | DIAG (우리 실습) |
| --- | --- | --- |
| 생성 모델 | GAN | Diffusion(SDXL) |
| 학습 | **필요**(GAN 학습) | **불필요**(zero-shot) |
| 제어 | 클래스 레이블·연속 공간 | **텍스트 + 영역 마스크** |
| 강점 | 다양성·다중 클래스 제어 | 가볍고 재현 쉬움, in-distribution |

학습이 불필요한 방식으로 colab에서 실습이 가능합니다.

---

## 참고 자료

- **DIAG** (CBMI 2024) -- [프로젝트 페이지](https://intelligolabs.github.io/DIAG/) · [코드](https://github.com/intelligolabs/DIAG)
- **VISION Workshop & Challenge** -- [공식 사이트](https://vision-based-industrial-inspection.github.io/) · [VISION Datasets 논문](https://arxiv.org/abs/2306.07890)
- **상위 솔루션** -- Wei et al., *Diversified and Multi-Class Controllable Industrial Defect Synthesis for Data Augmentation and Transfer*, CVPRW 2023
- **KSDD2** -- [Kolektor / ViCoS 공식 페이지](https://www.vicos.si/resources/kolektorsdd2/)
- **연결**: 이미지-마스크 쌍 생성을 더 보고 싶다면 AnomalyDiffusion(`sjtuplayer/anomalydiffusion`, MVTec/VisA)

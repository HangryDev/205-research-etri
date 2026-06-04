# 행동 인식 심화 — 골격 기반 연구 동향과 데이터

---

```{admonition} 이 문서의 위치
:class: note

`sec2_action_recognition.md`가 **MediaPipe + LSTM 실습**으로 "어떻게 만드는가"를 다룬다면, 이 보충 페이지는 그 **배경이 되는 연구 동향과 데이터 현실**을 다룹니다. 실습에서 쓴 골격 기반 접근이 학계에서 어떻게 발전해 왔고(특히 **ST-GCN** 계열), 왜 위험 행동 데이터를 직접 만들어야 하는지를 정리합니다.
```

---

## 골격 기반 행동 인식 모델의 계보

`sec2`에서 본 흐름은 **골격 추출(MediaPipe) → 시퀀스 분류(LSTM)** 였습니다. 학계에서는 여기서 한 단계 더 나아가, 관절들 사이의 **연결 관계(그래프)** 까지 모델에 직접 넣는 방식이 주류가 되었습니다. 그 대표가 **ST-GCN**입니다.

```{admonition} ST-GCN이 뭔가
:class: note

*Spatial-Temporal Graph Convolutional Network(시공간 그래프 합성곱 신경망)*. 관절(keypoint)을 **점**, 뼈대 연결을 **선**으로 보는 "그래프"를 만든 뒤, 그 그래프 위에서 합성곱을 수행합니다. LSTM이 시간 순서만 보는 것과 달리, ST-GCN은 **"어느 관절이 어느 관절과 이어져 있는가(공간)"와 "그 관계가 시간에 따라 어떻게 변하는가(시간)"를 동시에** 봅니다. 그래서 "팔과 다리가 함께 움직이는" 복합 동작을 더 잘 잡습니다.
```

```{mermaid}
flowchart LR
    A["영상"] --> B["골격 추출<br/>(OpenPose / MediaPipe)"]
    B --> C["관절 그래프 구성<br/>(점=관절, 선=뼈대)"]
    C --> D["ST-GCN<br/>(시공간 그래프 합성곱)"]
    D --> E["행동 분류<br/>(위험 / 정상)"]
```

`sec2`의 LSTM 방식은 구현이 단순하고 가벼워 실습·엣지에 적합하고, ST-GCN 계열은 정확도가 높아 연구·고난도 동작에 적합합니다. 둘은 대체재가 아니라 **난이도·자원에 따른 선택지**입니다.

---

## 대표 연구 사례 — 논문 인사이트를 실무로

작업자 안전 분야에서 골격 기반 행동 인식이 어떻게 쓰였는지를 보여주는 대표 연구들입니다.

| 연구 | 방법 | 결과 / 인사이트 |
|---|---|---|
| 비계(scaffold) 위 위험행동 분류 | OpenPose keypoint + 분류 (climbing / jumping / running) | precision 89.6%, recall 90.5% |
| ST-GCN 기반 동적 위험행동 인식 | OpenPose + ST-GCN + human partitioning + non-local attention | 관절별 가중치 설계로 복잡 동작 정확도 향상 |
| **CoG-STGCN** (무게중심 인지) | ST-GCN에 무게중심(CoG) 특징 결합 | 추락위험 8종 자체 데이터셋 Top-1 95.83% (baseline ST-GCN 93.75%) |
| 추락 감지 LSTM | ResNet pose 추출 → 30프레임 단위 LSTM 예측 | 시계열 후처리(프레임 윈도우)의 전형적 예 |

```{admonition} 핵심 — 도메인 지식이 모델 설계를 이긴다
:class: important

CoG-STGCN처럼 **물리적 사전지식(무게중심)을 특징으로 주입**하는 것이, 순수하게 데이터만으로 학습하는 것보다 "추락 직전의 급격한 무게중심 이동"을 더 잘 잡아냅니다. 모델 구조를 키우기 전에 **현장 도메인 지식을 특징으로 넣을 수 있는지**를 먼저 고민하는 편이 비용 대비 효과가 큽니다.
```

---

## 데이터 부족이라는 현실

건설·제조 작업자의 **위험 행동 공개 데이터셋은 매우 부족합니다.** 그래서 위 연구들도 대부분 자체 데이터셋(예: 타워크레인 시나리오용 WeTeam22, 추락위험 8종 자체 구축셋)을 직접 만들어 사용했습니다.

```{admonition} 실무 출발점
:class: tip

"마침맞은 공개셋이 없다"가 기본 전제입니다. 따라서 현실적인 첫 단계는 **현장 영상에서 pose를 추출하고, 소량만 라벨링해 자체 데이터셋을 만드는 것**입니다. 골격 좌표는 개인 식별 정보가 없어 사내 데이터 활용·공유의 부담도 적습니다.
```

---

## 모델 지형도와 실무 선택 기준

행동 인식에는 객체 탐지의 YOLO 같은 **단일 표준 모델이 없습니다.** 대신 모델을 "정확도 순위표"가 아니라 **운영 요건**으로 고릅니다. 아래 지형도는 대표 모델들을 두 축 위에 배치한 것입니다.

```{figure} s3_2_action_model_landscape.svg
:alt: 행동 인식 모델 지형도 — 입력 양식(골격↔RGB) × 연산 부담·배포 위치(엣지↔클라우드)
:width: 90%

행동 인식 모델 지형도 — 입력 양식 × 연산 부담·배포 위치
```

- **가로축(입력 양식)**: 왼쪽은 관절 좌표만 쓰는 **골격(keypoint)** 기반, 오른쪽은 영상 픽셀 전체를 쓰는 **RGB** 기반
- **세로축(연산 부담·배포 위치)**: 아래는 **경량·엣지**, 위는 **고성능·클라우드**
- 왼쪽 아래로 갈수록 **가볍고·빠르고·사생활 보호**에 유리하고, 오른쪽 위로 갈수록 **정확도·맥락 이해**가 좋지만 무겁습니다. `sec2`의 (MediaPipe → LSTM)은 왼쪽 아래, 비디오 파운데이션 모델(VideoMAE·InternVideo)은 오른쪽 위에 위치합니다.

### 운영 요건별 선택 기준

같은 "위험 행동 인식"이라도 무엇을 위해 쓰는지에 따라 정답 계열이 달라집니다.

| 운영 요건 | 지연 허용 | 권장 계열 | 근거 |
|---|---|---|---|
| **사고 후 원인 분석** (오프라인 리뷰) | 큼(실시간 불필요) | RGB 비디오 파운데이션 모델 fine-tune, 필요 시 TAL | 지연이 허용되니 무거운 모델 사용 가능, 도구·동선 등 **맥락**까지 봐야 원인 규명 |
| **실시간 경고** (저지연 알림) | 매우 작음(수십~수백 ms) | 골격 기반(pose → ST-GCN / LSTM), 엣지 | 좌표만 처리해 가볍고 빠름, 어느 관절이 이상한지 **설명 가능** |
| **24시간 자동 감시** (상시·다수 카메라) | 작음 | 골격 기반 + 경량화, 엣지 | 픽셀을 밖으로 안 보내 **사생활 보호**, 카메라당 비용·전력↓, 장시간 안정 |
| **미세·복합 동작 / 도구 상호작용** | 요건에 따라 | RGB 또는 **멀티모달**(골격+객체) | 손동작·도구 사용은 33개 키포인트만으로 표현하기 어려움 |

```{admonition} 일반화된 판단 축
:class: tip

구체 요건이 무엇이든, 결국 다음 축들의 조합으로 계열이 결정됩니다.

1. **지연 허용치**(실시간 ↔ 사후) 2. **배포 위치·자원**(엣지 ↔ 클라우드) 3. **개인정보 민감도**(픽셀 저장 가능 여부) 4. **동작 입도**(거친 자세 ↔ 미세 손동작) 5. **맥락 의존도**(배경·객체 정보 필요성) 6. **데이터·라벨 가용성**
```

### 흔히 만나는 한계와 대응 — 추가 학습·후처리

처음 고른 모델을 그대로 현장에 올리면 거의 항상 한계에 부딪힙니다. 그때 **추가 학습(재학습)** 또는 **후처리**를 도입합니다.

| 한계 | 증상 | 대응 |
|---|---|---|
| **도메인 격차** | 공개셋과 현장의 카메라 각도·복장·조명이 달라 정확도 급락 | 자체 데이터 소량 수집 후 **fine-tune / 전이학습** |
| **클래스 불균형** | 위험 행동은 희소, 정상이 압도 | 오버샘플링·**focal loss**·이상탐지 관점 전환 |
| **오탐 폭주** | 순간 흔들림·자세를 위반으로 잘못 판정 | 후처리: **Temporal Smoothing**, N프레임 연속 규칙, 신뢰도 임계값 튜닝 |
| **가림(occlusion)·결측 키포인트** | 기계·물체에 가려 좌표가 비거나 튐 | visibility 가중, 보간, **멀티뷰**(다중 카메라) |
| **미세 동작 표현 한계** | 골격만으로 손동작·도구 사용 구분 불가 | RGB·**멀티모달** 결합, 객체(안전모·도구) 정보 추가 |
| **시점 경계 모호(TAL)** | 행동의 시작·끝 프레임이 흔들림 | 윈도우 투표, 경계 후처리(boundary refinement) |
| **분포 변화(drift)** | 계절·신규 공정·신규 설비로 성능 저하 | **주기적 재학습**, 능동학습(active learning) |

```{admonition} 현장 프로젝트가 수렴하는 형태
:class: important

거의 모든 행동 인식 현장 도입은 **"공개셋으로 사전학습된 백본 → 자체 소량 데이터로 fine-tune → 시계열 후처리"** 의 세 박자로 수렴합니다. 새 파이프라인을 맨바닥부터 만든다기보다, 같은 레시피에 **부품(백본·헤드·후처리)을 요건에 맞게 바꿔 끼우는** 작업에 가깝습니다.
```

---

## 참고할 만한 공개 데이터셋 · 벤치마크

위험 행동 전용 공개셋은 부족하지만, **일반 행동 인식 벤치마크**로 모델·파이프라인을 먼저 검증한 뒤 자체 데이터로 옮겨가는 전략이 유효합니다.

| 데이터셋 | 특징 | 비고 |
|---|---|---|
| **NTU RGB+D / 120** | 56,880클립·60클래스(RGB+D) → 120은 114,480클립·120클래스, RGB+Depth+IR+**Skeleton** 제공 | 골격 기반 행동 인식의 표준 벤치마크 |
| **Kinetics-400/700** | 대규모 in-the-wild 영상 행동 | 사전학습(pretrain)에 널리 사용 |
| **HMDB51 / UCF101** | 다양한 인간 동작(51/101 클래스) | 영상 행동 분류 입문 표준 |
| **Human Activity Recognition (Video)** | 7개 행동 클래스 | 가벼운 입문 실습용(Kaggle) |

---

## 최근 학술대회 챌린지와 방법론 동향

연구가 어디로 가고 있는지는 매년 열리는 **공개 챌린지**에서 가장 잘 드러납니다. 최근 행동 인식 챌린지는 단순 분류를 넘어, **긴 영상에서 행동의 시작·끝 시점까지 찾는 과제(TAL)** 와 **여러 시점·여러 감각을 함께 쓰는 멀티모달**로 무게가 옮겨가고 있습니다.

```{admonition} 자주 나오는 두 용어
:class: note

- **TAL(Temporal Action Localization, 시간적 행동 위치추정)**: 잘라놓은 짧은 영상의 행동을 "맞히는" 분류와 달리, 자르지 않은 긴 영상에서 "어떤 행동이 몇 초~몇 초에 일어났는가"까지 찾아내는 과제입니다. 안전 모니터링의 "사고가 언제 시작됐나"와 직결됩니다.
- **Egocentric(1인칭) 영상**: CCTV 같은 3인칭이 아니라 작업자가 쓴 카메라(헬멧캠 등)에서 찍힌 1인칭 시점 영상입니다.
```

### 학술대회 챌린지 (CVPR · ICCV 워크숍)

| 챌린지 | 주최 / 연도 | 과제 | 대표 방법론 |
|---|---|---|---|
| **BinEgo-360** | ICCV 2025 | 멀티뷰·멀티모달 TAL (360° 파노라마 + 1인칭 + 공간음향 + GPS) | Temporal Shift Module(TSM) 확장 + 멀티태스크 학습(장면 분류 + TAL 동시) |
| **ActivityNet Challenge** | CVPR 워크숍(매년) | 긴 영상의 TAL (ActivityNet-1.3) | 대형 비디오 백본(VideoSwin) 특징 + Faster-TAD + 모델 앙상블 |
| **EPIC-Kitchens Challenges** | 매년 | 1인칭 행동 인식 · 행동 예측(anticipation) · TAL | 멀티모달(RGB+모션) 융합, Transformer 기반 |
| **AI City Challenge** | CVPR/ICCV 워크숍 | 창고·다중카메라 사람 추적·안전(산업 현장) | 다중카메라 정합 + 추적 + 합성 데이터(NVIDIA Omniverse) |

```{admonition} 안전 모니터링과의 연결
:class: tip

BinEgo-360의 "1인칭 + TAL"은 **작업자 헬멧캠으로 위험 행동의 발생 시점을 잡는** 시나리오와, AI City Challenge의 "다중카메라 + 안전"은 **여러 CCTV를 묶어 현장 전체를 보는** 시나리오와 그대로 맞닿아 있습니다.
```

### Kaggle · 응용 대회

Kaggle에는 행동 인식 전용 "경진대회"는 많지 않고, **데이터셋·노트북 중심**으로 형성되어 있습니다. 다만 영상 파이프라인을 통째로 연습하기 좋은 사례가 있습니다.

- **DeepFake Detection Challenge(DFDC)**: 행동 인식은 아니지만 "영상 → 프레임 샘플링 → 검출(전처리) → 분류 → 추론 속도 제약"이라는 영상 파이프라인 전 과정을 다루는 대표 사례.
- **Human Action Recognition(Video)** 등 데이터셋: 7개 클래스 규모의 입문용으로, 포즈 추출 → 시퀀스 분류 파이프라인 검증에 적합.

### 낙상(Fall) 검출 — 안전과 가장 가까운 응용

작업자·고령자 안전과 직결되어 활발히 연구되는 주제입니다. 방법론 흐름이 본 페이지의 골격 기반 접근과 정확히 일치합니다.

- **포즈 기반**: 3D 포즈 추출 → 낙상 분류. NTU RGB+D에서 99.83% 정확도, CPU 18 FPS·GPU 63 FPS로 실시간 동작을 보고한 사례.
- **파운데이션 모델 활용(최신)**: 대형 비디오 이해 모델(ViT 기반)을 fine-tune해 untrimmed 영상에서 "Fall / Lying / 일상행동"을 TAL로 구분, HQFSD 데이터셋 F1 0.96 보고.

```{admonition} 반복되는 방법론 패턴 (대회 솔루션에서)
:class: important

1. **대형 비디오 백본 사전학습 → 특징 추출**: Kinetics·HACS로 사전학습한 VideoSwin 등으로 특징을 뽑아 재사용.
2. **Transformer 기반 TAL**: Faster-TAD·ActionFormer 계열로 행동의 경계(시작·끝)를 예측.
3. **멀티모달·멀티뷰 융합**: RGB + 모션 + 음향 + 1인칭/3인칭을 함께 사용.
4. **경량 시간 모델링**: Temporal Shift Module(TSM)처럼 추가 연산 없이 시간 정보를 섞는 기법.
5. **Zero-shot·프롬프트 학습**: 언어(VLM)와 결합해 학습하지 않은 행동도 인식(예: 골격 기반 zero-shot, CVPR 2025).
6. **모델 앙상블**: 서로 보완하는 모델을 합쳐 마지막 성능을 끌어올림.
```

---

## 핵심 요약

1. **계보**: `sec2`의 (MediaPipe → LSTM)에서 한 단계 나아간 학계 주류는 관절 그래프를 직접 다루는 **ST-GCN 계열**이다.
2. **선택 기준**: 단일 표준 모델이 없으므로 **운영 요건**(사고 후 분석·실시간 경고·24시간 감시 등)으로 계열을 고른다. 핵심 축은 지연 허용치·배포 위치·개인정보·동작 입도·맥락 의존도·데이터 가용성이며, 현장 도입은 대개 **"사전학습 백본 → 자체 데이터 fine-tune → 시계열 후처리"** 로 수렴한다.
3. **인사이트**: 무게중심 같은 **도메인 지식을 특징으로 주입**하는 것이 모델을 키우는 것보다 효과적일 때가 많다(CoG-STGCN).
4. **데이터**: 위험 행동 공개셋은 부족하다. **pose 추출 + 소량 라벨링으로 자체 데이터셋**을 만드는 것이 현실적 출발점이며, 일반 벤치마크(NTU RGB+D 등)로 먼저 검증한다.
5. **동향**: 최근 챌린지(ICCV BinEgo-360, ActivityNet 등)는 단순 분류에서 **시점 찾기(TAL)·멀티모달·1인칭**으로 이동 중이며, 대형 비디오 백본 사전학습 + Transformer TAL + 앙상블이 솔루션의 공통 패턴이다.

---

## 참고 문헌

**행동 인식 모델 (골격 · RGB)**

- Yan, Xiong, Lin, [*Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (ST-GCN)](https://arxiv.org/abs/1801.07455), AAAI 2018
- Cao 등, [*OpenPose: Realtime Multi-Person 2D Pose Estimation*](https://arxiv.org/abs/1812.08008), 2019
- [*SkelFormer* — 계층적 Transformer 골격 행동 인식](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12795391/), 2026
- Mengxue 등, [*Simba: Mamba augmented U-ShiftGCN*](https://arxiv.org/abs/2404.07645), 2024
- Tong 등, [*VideoMAE*](https://arxiv.org/abs/2203.12602), NeurIPS 2022 · [*VideoMAE V2*](https://arxiv.org/abs/2303.16727), CVPR 2023
- Wang 등, [*InternVideo*](https://arxiv.org/abs/2212.03191), 2022 · [*InternVideo2*](https://arxiv.org/abs/2403.15377), ECCV 2024 · [*InternVideo-Next*](https://arxiv.org/abs/2512.01342), 2026
- [행동 인식 종합 서베이 (과제·방법·도전)](https://www.sciencedirect.com/science/article/pii/S2405959525001869), 2025

**작업자 안전 적용 연구**

- [Identifying Unsafe Behavior of Construction Workers (ST-GCN)](https://ascelibrary.org/doi/abs/10.1061/JCEMD4.COENG-13616), J. Constr. Eng. Manage., 2023
- [Center-of-Gravity-Aware Graph Convolution (CoG-STGCN)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12431360/)
- [Construction Site Safety Management: CV & Deep Learning (fall detection LSTM)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9863726/)
- [Vision-Based Detection of Unsafe Actions — Ladder/Scaffold (OpenPose)](https://www.researchgate.net/publication/275182852_Vision-Based_Detection_of_Unsafe_Actions_of_a_Construction_Worker_Case_Study_of_Ladder_Climbing)

**공개 데이터셋 · 벤치마크**

- NTU RGB+D — Shahroudy 등, [arXiv:1604.02808](https://arxiv.org/abs/1604.02808) · [NTU RGB+D 120](https://arxiv.org/abs/1905.04757) · [github](https://github.com/shahroudy/NTURGB-D)
- Kinetics — Kay 등, [arXiv:1705.06950](https://arxiv.org/abs/1705.06950)
- HMDB51 — Kuehne 등, ICCV 2011
- [Human Activity Recognition (Video) — Kaggle](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset)

**최근 챌린지 · 방법론**

- BinEgo-360 Challenge (ICCV 2025) — [멀티뷰·멀티모달 TAL 솔루션 (TSM + 멀티태스크)](https://arxiv.org/abs/2512.11189)
- ActivityNet Challenge — [TAL 기술 보고서 (VideoSwin + Faster-TAD)](https://arxiv.org/abs/2411.00883)
- [NVIDIA AI City Challenge](https://www.aicitychallenge.org/) (CVPR/ICCV 워크숍)
- 낙상 검출(포즈 기반) — [Video Based Fall Detection Using Human Poses](https://arxiv.org/abs/2107.14633)
- 낙상 검출(파운데이션 모델·TAL) — [Cutup and Detect](https://arxiv.org/abs/2401.16280)
- 골격 기반 Zero-shot (CVPR 2025) — [Semantic-guided Cross-Modal Prompt Learning](https://cvpr.thecvf.com/virtual/2025/poster/34969)

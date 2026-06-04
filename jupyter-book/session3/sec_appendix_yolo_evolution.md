# YOLO 변천사 — v1부터 YOLO26까지

---

```{admonition} 이 문서의 위치
:class: note

객체 탐지 모델 **YOLO**가 v1(2016)부터 **YOLO26**(2026)까지 어떻게 발전해 왔는지를 살펴봅니다. 컴퓨터 비전을 전공하지 않았더라도 따라올 수 있도록 풀어 썼고, 앞선 〈고전 CV와 초기 탐지 모델〉의 마지막 장면("한 번에 보자")에서 자연스럽게 이어집니다.

관통하는 줄거리는 하나입니다 — **"정확도 경쟁"에서 "현장에 잘 배포되느냐(속도·후처리·라이선스) 경쟁"으로의 이동.** 버전마다 원 논문 출처를 함께 확인할 수 있고, 마지막에는 실무 사용 후기(블로그·커뮤니티)도 살펴봅니다.
```

---

## 먼저 풀고 가는 용어


- **객체 탐지(Object Detection)**: 사진 속에 *무엇이* 있는지(분류)뿐 아니라 *어디에* 있는지(위치, 네모 상자)까지 찾아내는 일. "이 사진에 고양이가 있다"가 분류라면, "여기 이 네모 안에 고양이가 있다"가 탐지입니다.
- **YOLO = You Only Look Once**: 직역하면 "한 번만 본다". 사진을 여러 번 쪼개 보지 않고 단 한 번의 처리로 위치와 종류를 동시에 알아내서 빠르다는 뜻. YOLO의 정체성이 이름에 담겨 있습니다.
- **CNN(합성곱 신경망)**: 이미지를 작은 필터로 훑어 가장자리·질감·모양 같은 시각 패턴을 단계적으로 뽑아내는 신경망. 오랫동안 이미지 인식의 기본 도구였습니다.
- **실시간(real-time)**: 영상이 흐르는 속도에 뒤처지지 않고 처리하는 것. 일반적인 CCTV·IP 카메라는 보통 초당 15~30장(15~30 FPS)으로 영상을 내보내므로(저장 용량 절약 시 15~20 FPS), 모델이 카메라 FPS 이상으로 처리해야 끊김 없이 따라갑니다. 30 FPS 카메라라면 한 장을 약 33ms 이내에 처리해야 한다는 뜻입니다.
- **성능 지표 3개**: 탐지 정확도 종합 점수 **mAP**(0~100%, 높을수록 정확), 초당 처리 장수 **FPS**(높을수록 빠름), 한 장 처리 시간 **Latency**(ms, 낮을수록 빠름).
- **모델의 3등분 구조**: 이미지에서 특징을 뽑는 **Backbone(몸통)**, 크고 작은 특징을 섞어 정리하는 **Neck(목)**, 최종적으로 "여기 네모, 이건 사람" 하고 답을 내놓는 **Head(머리)**.

---

## 한눈에 보는 변천사

```{mermaid}
timeline
    title YOLO 아키텍처 변천사 (핵심 혁신 중심)
    2016 : YOLOv1 (Redmon) — 단일 스테이지 grid 예측의 시작
    2018 : YOLOv3 — multi-scale 예측 · Darknet-53 backbone
    2020 : YOLOv4 (Bochkovskiy) — Mosaic 증강 · CSPDarknet : YOLOv5 (Ultralytics) — PyTorch 모듈화 · 배포 생태계
    2023 : YOLOv8 (Ultralytics) — anchor-free · decoupled head · C2f · 멀티태스크 [NMS 사용]
    2024 : YOLOv9 — PGI / GELAN : YOLOv10 (THU-MIG) — NMS-free dual assignment : YOLO11 (Ultralytics) — C3k2 backbone · C2PSA attention
    2025 : YOLOv12 (Tian 등 · NeurIPS) — attention-centric · Area Attention(A2) · R-ELAN · FlashAttention : YOLOv13 — attention 고도화 [NMS·DFL 유지]
    2026 : YOLO26 (Ultralytics) — native NMS-free · DFL 제거 · ProgLoss · STAL · MuSGD · CPU 43%↑
```

| 버전 | 연도 | 만든 곳 | 한 줄 핵심 | 원 논문/출처 |
|---|---|---|---|---|
| **YOLOv1** | 2016 | Redmon 등 | 탐지를 "한 번에" 푸는 방식의 시작 | [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) |
| **YOLOv2/9000** | 2017 | Redmon, Farhadi | anchor 도입, 더 정확하게 | [arXiv:1612.08242](https://arxiv.org/abs/1612.08242) |
| **YOLOv3** | 2018 | Redmon, Farhadi | 여러 크기 동시 탐지, 작은 물체 개선 | [arXiv:1804.02767](https://arxiv.org/abs/1804.02767) |
| **YOLOv4** | 2020 | Bochkovskiy 등 | 학습 기법 총정리(Mosaic 등) | [arXiv:2004.10934](https://arxiv.org/abs/2004.10934) |
| **YOLOv5** | 2020 | Ultralytics | PyTorch·배포 생태계 → 산업 표준화 | [github](https://github.com/ultralytics/yolov5) |
| **YOLOv6** | 2022 | Meituan | 산업용 최적화(배달로봇 실사용) | [arXiv:2209.02976](https://arxiv.org/abs/2209.02976) |
| **YOLOv7** | 2022 | Wang 등 | 추가 기능(pose 등)·정확도 | [arXiv:2207.02696](https://arxiv.org/abs/2207.02696) |
| **YOLOv8** | 2023 | Ultralytics | anchor-free + 멀티태스크 | [docs](https://docs.ultralytics.com/models/yolov8/) |
| **YOLOv9** | 2024 | Wang 등 | 정보 손실 줄이기(PGI·GELAN) | [arXiv:2402.13616](https://arxiv.org/abs/2402.13616) |
| **YOLOv10** | 2024 | 칭화대 THU-MIG | NMS 후처리 제거(NMS-free) | [arXiv:2405.14458](https://arxiv.org/abs/2405.14458) |
| **YOLO11** | 2024 | Ultralytics | 효율 모듈·공간 attention | [docs](https://docs.ultralytics.com/models/yolo11/) |
| **YOLOv12** | 2025 | Tian 등 | attention 중심 설계 전환 | [arXiv:2502.12524](https://arxiv.org/abs/2502.12524) |
| **YOLOv13** | 2025 | Lei 등 | 하이퍼그래프로 전역 관계 포착 | [arXiv:2506.17733](https://arxiv.org/abs/2506.17733) |
| **YOLO26** | 2026 | Ultralytics | native NMS-free·엣지 최적화 | [docs](https://docs.ultralytics.com/models/yolo26/) |

v5·v8·v11·YOLO26은 회사(Ultralytics)가 만든 제품형이라 공식 논문이 없고 문서·코드로 공개됩니다. 나머지는 학술 논문이 있습니다.

---

## 1세대 (v1–v3): 탐지를 다시 정의하다

YOLO 이전의 탐지 모델은 2단계 방식이었습니다. ① 먼저 "물체가 있을 법한 후보 영역"을 잔뜩 추려내고 ② 각 후보를 하나씩 분류하는 식이죠. 정확하지만 느려서 실시간이 어려웠습니다.

**YOLOv1**(2016)의 혁신은 이 일을 한 번에 푼 것입니다. 사진을 격자(grid)로 나누고, 각 칸이 "내 안에 물체가 있나? 있다면 네모는 어디고 종류는 뭐지?"를 동시에 답하게 했습니다. 그래서 이름이 *You Only Look Once*입니다. 정확도는 2단계 방식보다 조금 낮았지만 속도가 압도적이라 실시간 탐지의 문을 열었습니다.
*(원 논문: Redmon, Divvala, Girshick, Farhadi, "You Only Look Once", CVPR 2016 — [arXiv:1506.02640](https://arxiv.org/abs/1506.02640))*

**YOLOv2/YOLO9000**(2017)부터 **anchor(앵커)** 개념이 들어옵니다.

```{admonition} anchor가 뭔가
:class: note

모델이 맨바닥에서 네모 크기를 추측하긴 어렵습니다. 그래서 "가로로 긴 네모, 세로로 긴 네모, 정사각형…" 같은 미리 정해둔 기준 상자들(anchor box)을 깔아두고, 모델은 "이 기준 상자를 얼마나 옮기고 늘릴까?"만 학습합니다. 출제 범위를 좁혀주니 학습이 쉬워집니다.
```

*(원 논문: Redmon, Farhadi, "YOLO9000: Better, Faster, Stronger", CVPR 2017 — [arXiv:1612.08242](https://arxiv.org/abs/1612.08242))*

**YOLOv3**(2018)는 세 가지 크기로 동시에 탐지하게 만들어, 멀리 있는 작은 물체까지 잘 잡게 했습니다. 오늘날까지 쓰이는 안정적인 기본기를 다진 버전입니다.
*(원 논문: Redmon, Farhadi, "YOLOv3: An Incremental Improvement" — [arXiv:1804.02767](https://arxiv.org/abs/1804.02767))*

---

## 2세대 (v4–v7): 정확도와 생태계 경쟁

여기서부터 원작자(Redmon)가 손을 떼고 여러 팀이 YOLO라는 이름을 이어받습니다.

**YOLOv4**(2020)는 그동안 흩어져 있던 학습 잘 되게 하는 잔기술들을 총정리했습니다. 대표적으로 **Mosaic 증강**(사진 4장을 한 장으로 붙여 학습시켜 다양한 상황에 강해지게 하는 기법)이 있습니다.
*(원 논문: Bochkovskiy, Wang, Liao, "YOLOv4" — [arXiv:2004.10934](https://arxiv.org/abs/2004.10934))*

**YOLOv5**(2020, Ultralytics)는 논문이 아니라 "쓰기 편한 제품"으로 등장했습니다. 설치 한 줄, 학습 한 줄, 배포까지 매끄럽게 되는 생태계를 만들어 사실상 산업 현장의 표준이 됩니다. "성능보다 편의성이 보급을 만든다"는 걸 보여준 분기점입니다.
*(출처: [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) — Glenn Jocher / Ultralytics)*

**YOLOv6**(2022, 중국 Meituan)는 배달로봇 같은 산업 현장에 맞춰 최적화됐고 실제 운영에 투입됐습니다. *([arXiv:2209.02976](https://arxiv.org/abs/2209.02976))* **YOLOv7**(2022)은 정확도를 끌어올리고 자세 추정(pose) 같은 기능을 더했습니다. *([arXiv:2207.02696](https://arxiv.org/abs/2207.02696))*

---

## 3세대 (v8): anchor-free와 멀티태스크

**YOLOv8**(2023, Ultralytics)의 가장 큰 변화는 anchor를 없앤 것(anchor-free)입니다.

```{admonition} anchor-free가 뭐고, 왜 좋은가
:class: important

앞서 anchor(기준 상자)는 학습을 도와준다고 했지만 단점이 있습니다. 사람이 데이터셋마다 기준 상자의 크기·비율을 미리 맞춰줘야 하고, 새 데이터에선 다시 튜닝해야 합니다. **anchor-free**는 이 기준 상자를 버리고 물체의 중심점과 경계를 직접 예측합니다.

**장점 3가지**: ① 사람이 손볼 설정값이 줄어 편하고, ② 처음 보는 데이터에도 더 잘 적응하며, ③ 구조가 단순해져 더 빨라집니다.
```

또한 v8은 탐지뿐 아니라 분할(segmentation)·자세(pose)·분류(classification)를 한 틀에서 처리하는 멀티태스크 모델로 자리 잡아, 현재도 가장 널리 쓰이는 버전 중 하나입니다. 단, 아직 NMS라는 후처리에는 의존합니다.
*(출처: [docs.ultralytics.com](https://docs.ultralytics.com/models/yolov8/) — Ultralytics)*

---

## 4세대 (v9–v11): 후처리(NMS)를 없애기 시작하다

**YOLOv9**(2024)는 "깊은 신경망을 통과하며 정보가 새어나가는" 문제를 줄이는 기법(**PGI·GELAN**)을 제안해, v8보다 부품(파라미터)을 49%, 계산을 43% 줄이면서도 정확도는 더 높였습니다. 즉 가벼우면서 정확하게.
*(원 논문: Wang, Yeh, Liao, "YOLOv9", ECCV 2024 — [arXiv:2402.13616](https://arxiv.org/abs/2402.13616))*

이 시대의 진짜 전환점은 **YOLOv10**(2024, 칭화대)입니다. NMS라는 후처리 단계를 아예 없앤(NMS-free) 첫 YOLO입니다.

```{admonition} NMS를 없애면 왜 좋은가
:class: important

NMS는 모델이 같은 물체에 겹쳐 그린 네모를 정리하는 후처리 단계입니다. 그런데 ① 순서대로 처리돼 느리고 ② 다른 기기로 옮겨 심기(export)가 까다롭고 ③ 현장마다 설정을 다시 맞춰야 합니다. **NMS-free**는 모델이 애초에 물체당 네모 하나만 내도록 학습시켜 정리 단계를 통째로 없앱니다 → 더 빠르고, 배포가 단순해집니다.
```

YOLOv10은 학습할 때는 풍부하게 가르치고(one-to-many), 실제 추론할 때는 물체당 하나만 내놓는(one-to-one) 이중 구조로 이를 달성했습니다. 그 결과 비슷한 정확도에서 지연(latency)을 크게 줄였습니다.
*(원 논문: Wang 등, "YOLOv10: Real-Time End-to-End Object Detection", NeurIPS 2024 — [arXiv:2405.14458](https://arxiv.org/abs/2405.14458))*

**YOLO11**(2024, Ultralytics)은 효율 모듈과 공간 attention(C2PSA)을 더해 정확도·속도 균형을 다듬은, 현재 프로덕션에서 가장 무난한 선택지 중 하나입니다.

---

## 5세대 (v12–v13): attention의 도입

**YOLOv12**(2025)는 YOLO 역사상 처음으로 **attention(어텐션)을 설계의 중심**에 놓았습니다.

```{admonition} attention이 뭔가
:class: note

이미지의 모든 부분을 똑같이 보지 않고 중요한 영역에 더 "집중"하도록 가중치를 주는 기법입니다. 사람이 사진을 볼 때 핵심에 눈이 먼저 가는 것과 비슷합니다. 정확도엔 아주 좋은데, 전통적으로 계산이 무거워 느리다는 게 큰 약점이었습니다. v12는 *Area Attention*·*FlashAttention* 같은 기법으로 이 속도 문제를 상당히 해결했습니다.
```

작은 모델 기준 정확도 40.6% mAP를 1.64ms(T4 GPU)에 달성해, v10·v11보다 정확하면서 속도는 비슷합니다.
*(원 논문: Tian, Ye, Doermann, "YOLOv12: Attention-Centric Real-Time Object Detectors", NeurIPS 2025 — [arXiv:2502.12524](https://arxiv.org/abs/2502.12524))*

**YOLOv13**(2025)은 여기서 한 발 더 나아가, 떨어져 있는 여러 부분의 관계까지 한꺼번에 보는 하이퍼그래프(hypergraph) 기법으로 복잡한 장면에서의 탐지를 강화했습니다.
*(원 논문: Lei 등, "YOLOv13: ... Hypergraph-Enhanced Adaptive Visual Perception" — [arXiv:2506.17733](https://arxiv.org/abs/2506.17733))*

단, v12·v13는 정확도를 높였지만 NMS와 복잡한 손실 계산(DFL)을 그대로 유지해 지연·배포 부담이 남았습니다. 이 한계가 다음 버전의 동기가 됩니다.

---

## 6세대 (YOLO26): NMS-free의 표준화, 엣지의 시대

**YOLO26**(2026, Ultralytics)은 v10이 연 방향을 표준으로 굳혔습니다. 처음부터 NMS 없이(native NMS-free) 동작하고, 복잡한 손실 계산(DFL)도 제거해 다른 기기로 옮겨 심기 쉽게 만들었습니다. 특히 CPU만 있는 저전력 기기에서 이전(YOLO11) 대비 추론이 43% 빨라져, 로봇·모바일·현장 카메라 같은 엣지(edge) 환경을 정조준했습니다.
*(출처: [docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26/) · 리뷰 논문 [arXiv:2509.25164](https://arxiv.org/abs/2509.25164))*

```{admonition} 엣지(edge)가 뭔가
:class: note

클라우드 서버가 아니라 현장의 작은 기기(라즈베리파이, NVIDIA Jetson, 스마트폰 등)에서 직접 AI를 돌리는 것입니다. 인터넷 없이도 즉시 반응하고 개인정보가 밖으로 안 나가는 장점이 있지만, 성능이 약해서 모델이 가벼워야 합니다. 그래서 "경량화·속도"가 핵심 화두입니다.
```

---

## 심화: 후처리(NMS)와 경량화 — 논문 인사이트를 실무 배포로

여기까지가 "어떻게 발전해 왔는가"였다면, 이 절은 "현장에 올릴 때 무엇이 중요한가"를 다룹니다. 핵심은 **mAP만 높은 모델은 논문에서 끝나고, 실무에서 살아남는 모델은 후처리가 단순하고 엣지에서 빠르다**는 점입니다.

### ① YOLOv8 vs YOLOv10 — 무엇이 바뀌었나

| 항목 | YOLOv8 (2023, Ultralytics) | YOLOv10 (2024, 칭화대 THU-MIG) |
|---|---|---|
| Head | anchor-free + decoupled head | dual head (one-to-many + one-to-one) |
| 후처리 | **NMS 필요** | **NMS-free** (추론 시 one-to-one head만 사용) |
| 핵심 모듈 | C2f (v5의 C3 대체) | lightweight cls head · spatial-channel decoupled downsampling · rank-guided block · partial self-attention |
| 학습 supervision | 단일 assignment | one-to-many로 풍부한 supervision + 일관성 매칭(consistent matching metric) |
| 강점 | 멀티태스크(detect/seg/pose/cls), 성숙한 생태계 | end-to-end 추론, 낮은 latency |
| 벤치마크 | — | COCO 52.7% AP, YOLOv9-C 대비 latency 46%↓ |

YOLOv10의 dual-label assignment는 "one-to-one만 쓰면 supervision이 약해진다"는 문제를, 학습 때는 one-to-many로 강하게 가르치고 추론 때만 one-to-one을 쓰는 방식으로 해결합니다.

### ② 후처리(NMS)가 왜 실무의 골칫거리인가

NMS는 언뜻 "중복 박스 제거"처럼 보이지만, 실제 배포에선 세 가지 문제를 만듭니다.

1. **순차 연산** → GPU 병렬화가 어렵고 latency에 직접 기여
2. **export 복잡도** → ONNX/TensorRT 변환 시 별도 커스텀 op으로 빠져 파이프라인을 복잡하게 만듦
3. **하이퍼파라미터 민감도** → IoU·confidence threshold가 현장(카메라 각도·혼잡도)마다 재튜닝 필요

```{admonition} 논문을 읽을 때
:class: important

mAP뿐 아니라 **후처리 latency와 export 가능성**을 함께 보아야 합니다. YOLOv10·YOLO26이 NMS를 없앤 진짜 이유가 바로 이 지점입니다.
```

### ③ 경량화 — 실무에서 다룰 네 가지

| 기법 | 핵심 아이디어 | 실무 주의점 |
|---|---|---|
| **Pruning(가지치기)** | BN scale factor·L1 norm 기준으로 중요도 낮은 채널/필터 제거 | **구조적(structured) pruning만 실속이 있다** |
| **Quantization(양자화)** | FP16 / INT8 (PTQ·QAT)로 정밀도를 낮춰 크기·속도 개선 | TensorRT·NCNN에선 보통 export 파이프라인 안에서 수행 |
| **Distillation(증류)** | 큰 teacher 모델의 출력을 작은 student가 모방 (CWD 등) | latency·메모리 예산상 풀사이즈 모델을 못 쓸 때 유용 |
| **Runtime(추론 엔진)** | TensorRT / ONNX Runtime / NCNN | NCNN=ARM·모바일, TensorRT=NVIDIA(layer fusion + FP32/FP16/INT8) |

```{admonition} 논문과 현실의 차이
:class: warning

논문이 "sparsity 90% 달성"이라고 자랑해도, **비구조적(unstructured) sparsity는 TensorRT 같은 표준 컴파일러에서 latency 이득이 거의 없습니다.** 실제 프로덕션에서는 보통 **pruning → quantization-aware training → distillation**을 순차 결합합니다.
```

실제 배포에서의 속도 감각을 주는 수치는 다음과 같습니다.

- Jetson Nano + `yolov8n`: 프레임당 약 163~170 ms(≈ 6 FPS) → "엣지에선 nano급도 실시간이 빠듯하다"
- LAMP pruning + 채널 distillation + TensorRT: latency 57.6 ms, 미최적화 대비 25.4% 개선(Jetson Nano)
- YOLO26n: YOLO11n 대비 CPU ONNX 추론 43% 빠름, COCO 40.9~57.5 mAP를 T4 TensorRT 1.7~11.8 ms에 처리

```{admonition} 핵심 키워드
:class: tip

**구조적 Pruning + Quantization + 적합한 Runtime**. 셋 다 "정확도를 조금 양보하고 속도·크기를 크게 얻는" 같은 철학이며, 엣지 배포의 표준 조합입니다.
```

---

## 실무 사용 후기와 현장의 목소리

학술 성능과 별개로, 실제로 회사에서 쓸 때 부딪히는 이야기들입니다.

### ① 가장 중요한 함정 — 라이선스(AGPL)

Ultralytics가 만든 YOLO(v5·v8·v11·YOLO26 등)는 **AGPL-3.0**이라는 라이선스를 씁니다. 무료지만 조건이 강합니다.

```{admonition} 쉽게 말하면
:class: warning

이 코드·모델·**가중치(학습 결과물)** 를 회사 제품이나 웹서비스에 쓰면, 내 프로젝트 전체 소스코드를 외부에 공개해야 합니다. 공개하기 싫다면 유료 **Enterprise License**를 사야 합니다. 심지어 내가 파인튜닝한 가중치까지 이 조건이 적용됩니다.
```

즉 "성능 좋다고 그냥 상용 제품에 넣었다간 법적 리스크"가 생깁니다. 실무자들은 이 때문에 상업적으로 더 자유로운 라이선스의 모델(예: **YOLOX** — Apache 2.0)을 쓰거나, 아키텍처를 직접 다시 구현하고 가중치를 처음부터 재학습해서 AGPL을 피하기도 합니다.

- Ultralytics 공식 라이선스: [ultralytics.com/license](https://www.ultralytics.com/license)
- 실무자 토론(Hacker News) — AGPL 회피 재구현·재학습 경험담: [news.ycombinator.com](https://news.ycombinator.com/threads?id=bevenky)


### ② 최신 ≠ 항상 최선 — 프로덕션 선택

번호가 높다고 무조건 좋은 게 아닙니다. **YOLOv12**는 attention 블록 때문에 학습이 불안정하고 메모리를 많이 먹으며 CPU에서 느릴 수 있어, 만든 회사(Ultralytics)조차 프로덕션에는 **YOLO11 또는 YOLO26**을 권장합니다. 실무에선 "가장 새 버전"보다 "안정적이고 내 기기에서 빠른 버전"을 고르는 게 정답에 가깝습니다.

- 번호판 탐지 과제로 v5·v6·v7·v8을 같은 조건에서 비교한 후기: [techchinmay (Substack)](https://techchinmay.substack.com/p/yolo-v8-the-real-state-of-the-art)
- YOLOv9의 PGI·GELAN을 실무자 관점에서 풀어 쓴 글: [AI Trends (Medium)](https://medium.com/ai-trends/yolov9-object-detection-with-programmable-gradient-information-pgi-and-generalized-efficient-4fa3352409cc)
- 토론이 활발한 커뮤니티: Reddit **r/computervision**, **[r/ultralytics](https://reddit.com/r/ultralytics)**

---

## 핵심 요약

1. **개념의 뿌리**: anchor·sliding window·후보 영역·NMS·특징 등 지금 YOLO에서 쓰는 단어 대부분이 이전 시대에 태어남.
2. **줄거리**: v8 → v10 → v26의 흐름은 "정확도 경쟁"에서 "배포 친화성(후처리·속도) 경쟁"으로의 패러다임 이동.
3. **배포의 두 축**: 후처리(NMS) 제거와 경량화(구조적 Pruning·Quantization·Distillation + 적합한 Runtime)가 엣지 실시간 성능을 좌우. "논문의 sparsity 수치"와 "실제 latency"는 다를 수 있다.
4. **그래서 무엇을 쓰나**: 빠른 학습·시제품은 **YOLOv8/YOLO11**, 엣지·저전력은 **YOLO26**, 상업 제품은 **라이선스 먼저 확인**(AGPL이 부담되면 YOLOX 등 대안 확인), 연구·최고 정확도는 **YOLOv12/v13**(학습 안정성·속도 트레이드오프 감안).

---

## 참고 문헌

**학술 논문 (원문 링크)**

- YOLOv1 — Redmon 등, [*You Only Look Once*](https://arxiv.org/abs/1506.02640), CVPR 2016
- YOLOv2/9000 — Redmon, Farhadi, [*YOLO9000*](https://arxiv.org/abs/1612.08242), CVPR 2017
- YOLOv3 — Redmon, Farhadi, [*YOLOv3: An Incremental Improvement*](https://arxiv.org/abs/1804.02767), 2018
- YOLOv4 — Bochkovskiy 등, [*YOLOv4*](https://arxiv.org/abs/2004.10934), 2020
- YOLOv6 — Li 등(Meituan), [*YOLOv6*](https://arxiv.org/abs/2209.02976), 2022
- YOLOv7 — Wang 등, [*YOLOv7*](https://arxiv.org/abs/2207.02696), 2022
- YOLOv9 — Wang, Yeh, Liao, [*YOLOv9 (PGI·GELAN)*](https://arxiv.org/abs/2402.13616), ECCV 2024
- YOLOv10 — Wang 등(Tsinghua), [*YOLOv10*](https://arxiv.org/abs/2405.14458), NeurIPS 2024
- YOLOv12 — Tian, Ye, Doermann, [*YOLOv12*](https://arxiv.org/abs/2502.12524), NeurIPS 2025
- YOLOv13 — Lei 등, [*YOLOv13*](https://arxiv.org/abs/2506.17733), 2025

**제품형(논문 없음) · 공식 문서/코드**

- YOLOv5 — [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- YOLOv8 / YOLO11 — [Ultralytics Docs](https://docs.ultralytics.com/)
- YOLO26 — [Ultralytics Docs](https://docs.ultralytics.com/models/yolo26/) (리뷰 논문 [arXiv:2509.25164](https://arxiv.org/abs/2509.25164))
- YOLO 계열 종합 리뷰 — [*Ultralytics YOLO Evolution*](https://arxiv.org/abs/2510.09653)

**경량화 · 엣지 배포**

- [YOLOv8 경량화 — Pruning & Quantization (TensorRT)](https://drpress.org/ojs/index.php/mmaa/article/view/27891)
- [HQP: Sensitivity-Aware Hybrid Quantization and Pruning](https://arxiv.org/abs/2602.06069) (structured vs unstructured 논의)
- [YOLOv8 on Jetson Nano — 실측 latency](https://app.readytensor.ai/publications/accelerating-edge-vision-yolov8-object-detection-on-jetson-nano-4D88m4ggztQt)
- [Pruning·Quantization·Knowledge Distillation 개요](https://deepwiki.com/coderonion/awesome-yolo-object-detection/5.3-pruning-quantization-and-knowledge-distillation)

**라이선스 & 실무 후기**

- [Ultralytics License](https://www.ultralytics.com/license)
- [YOLO Model Licenses: A Developer's Guide (Medium)](https://medium.com/@bingbai.jp/yolo-model-licenses-a-developers-guide-da722767b6f8)
- [Hacker News — AGPL 회피 토론](https://news.ycombinator.com/threads?id=bevenky)
- [YOLOv8 직접 비교 후기 (Substack)](https://techchinmay.substack.com/p/yolo-v8-the-real-state-of-the-art)
- [Reddit r/ultralytics](https://reddit.com/r/ultralytics)
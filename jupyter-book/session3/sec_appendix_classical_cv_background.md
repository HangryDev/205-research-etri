# 고전 CV와 초기 탐지 모델 — YOLO 이전의 세계

---

```{admonition} 이 문서의 위치
:class: note

객체 탐지 모델 **YOLO**가 등장하기 전, 컴퓨터 비전이 어떻게 물체를 찾아냈는지 다루는 **배경 자료**입니다. 뒤에 이어지는 〈YOLO 변천사〉의 첫 장면("YOLO 이전의 탐지 모델은 2단계 방식이었습니다…")으로 자연스럽게 연결됩니다.

이 문서를 관통하는 줄거리는 하나입니다 — **"사람이 직접 '무엇이 물체인지' 규칙(특징)을 손으로 설계하던 시대" → "기계가 데이터로 그 규칙을 스스로 배우는 시대"** 로의 이동. 특히 고전 CV는 옛 기술이 아니라 지금도 최신 AI와 결합해 쓰이며, 마지막 절에서 그 결합을 따로 다룹니다.
```

---

## 먼저 풀고 가는 용어

탐지·mAP·FPS·Backbone 등 공통 용어는 〈YOLO 변천사〉와 겹치므로, 여기서는 이 문서에서 새로 등장하는 용어 위주로 정리합니다.

- **컴퓨터 비전(CV, Computer Vision)**: 컴퓨터가 사진·영상을 "보고" 이해하게 만드는 분야 전체. 객체 탐지는 그 일부입니다.
- **특징(Feature)**: 이미지에서 물체를 알아보는 데 쓸모 있는 단서. 예를 들어 가장자리(edge), 모서리(corner), 밝기 변화, 색, 질감 같은 것들입니다. 사람이 얼굴을 알아볼 때 "눈·코·입의 배치"를 단서로 쓰는 것과 비슷합니다.
- **분류기(Classifier)**: 특징을 받아 "이건 사람/고양이/배경" 하고 판정하는 부분. 고전 CV에서 자주 쓴 두 가지는 두 종류를 가장 깔끔하게 가르는 경계선을 찾는 **SVM(서포트 벡터 머신)**, 그리고 약한 판정기 여러 개를 모아 강한 판정기 하나로 만드는 **AdaBoost** 입니다.
- **Sliding Window(창문 밀기)**: 작은 네모(창)를 이미지 위에서 조금씩 옮겨 가며 전수조사하는 방식. 단순하지만 느리고 중복이 많습니다.
- **Region Proposal(후보 영역 추리기)**: 전수조사 대신 "여기쯤 물체가 있을 것 같다"는 후보 구역만 똑똑하게 추려내는 기법. 딥러닝 2단계 검출기의 핵심 부품입니다.
- **IoU(겹침 비율)**: 모델이 그린 네모와 정답 네모가 얼마나 겹치는지를 0~1로 잰 값. 탐지가 맞았는지 채점할 때 씁니다.
- **2단계(two-stage) vs 1단계(one-stage)**: 2단계는 ① 후보 영역 추리기 → ② 각 후보 분류로 정확하지만 느리고, 1단계는 위치와 종류를 한 번에 예측해 빠릅니다(실시간). YOLO가 1단계에 해당합니다.

```{admonition} 이 문서의 핵심 개념 — 손으로 만든 특징 vs 학습된 특징
:class: important

- **손으로 만든 특징(hand-crafted)**: 사람이 "이런 패턴이 나오면 물체다"라고 **공식을 직접 설계**합니다. (고전 CV 시대)
- **학습된 특징(learned)**: 어떤 패턴이 중요한지를 **기계가 데이터를 보고 스스로** 찾아냅니다. (딥러닝 시대)

이 둘의 교체가 2012년 전후로 일어난 대전환이며, 이 문서 전체의 줄거리입니다.
```

---

## 한눈에 보는 변천사 (YOLO 직전까지)

| 시대 | 기법 | 연도 | 한 줄 핵심 | 원 논문/참고 |
|---|---|---|---|---|
| 고전 CV | Sliding Window | ~ | 작은 창으로 전수조사(모든 방식의 토대) | [개념 설명](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/) |
| 고전 CV | **Viola–Jones** | 2001 | Haar 특징+AdaBoost, 최초의 실시간 얼굴 탐지 | [DOI](https://doi.org/10.1109/CVPR.2001.990517) · [설명](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) |
| 고전 CV | **HOG + SVM** | 2005 | 기울기 방향 분포로 보행자 탐지 | [PDF](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) · [설명](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) |
| 고전 CV | **DPM** | 2008–2010 | 물체=부품들의 조합, 고전 CV의 정점 | [DOI](https://doi.org/10.1109/TPAMI.2009.167) |
| 고전 CV | SIFT / SURF / ORB | 2004~2011 | 특징점으로 매칭·정합(지금도 현역) | [SIFT PDF](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) · [OpenCV](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html) |
| 딥러닝 2단계 | **R-CNN** | 2014 | 후보영역 + CNN, 정확도 도약(하지만 느림) | [arXiv:1311.2524](https://arxiv.org/abs/1311.2524) |
| 딥러닝 2단계 | SPP-Net / **Fast R-CNN** | 2014–2015 | 특징을 재활용해 속도 개선 | [arXiv:1504.08083](https://arxiv.org/abs/1504.08083) |
| 딥러닝 2단계 | **Faster R-CNN** | 2015 | 후보영역까지 신경망(RPN), 2단계의 완성 | [arXiv:1506.01497](https://arxiv.org/abs/1506.01497) |
| 다리 | OverFeat / **SSD** | 2013 / 2016 | 1단계 검출의 등장 → YOLO로 | [arXiv:1512.02325](https://arxiv.org/abs/1512.02325) |

---

## 고전 CV 시대: 사람이 규칙을 손으로 설계하다

딥러닝 이전에는 사람이 직접 "이런 모양·패턴이 보이면 물체다"라는 공식을 만들어 넣었습니다. 이 시대를 이해하면, 지금의 AI가 무엇을 자동화한 것인지가 선명해집니다.

### ① Sliding Window — 모든 것의 출발점

가장 단순한 아이디어입니다. 작은 네모(창)를 이미지 왼쪽 위부터 오른쪽 아래까지 조금씩 밀면서, 각 위치에서 "이 안에 물체가 있나?"를 검사합니다. 물체 크기가 제각각이니 창 크기도 여러 개로 바꿔 가며 반복합니다.

```{admonition} 왜 문제였나
:class: note

한 장에 검사할 위치가 **수만~수십만 개**가 됩니다. 느리고, 같은 물체에 네모가 여러 개 겹쳐 나옵니다(이 "겹친 네모 정리" 문제가 나중에 **NMS**로 이어집니다). 그래서 "전수조사 말고 똑똑하게 후보만 추리자"는 Region Proposal 아이디어가 나옵니다.
```

직접 보는 설명: [PyImageSearch — Sliding Windows](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)

### ② Viola–Jones (2001) — 최초의 실시간 탐지

디지털카메라·스마트폰의 "얼굴 네모"의 원조입니다. 세 가지 아이디어로 처음으로 실시간 얼굴 탐지를 해냈습니다.

```{admonition} 무엇이 똑똑했나
:class: note

- **Haar 특징**: "이 영역은 밝고 옆은 어둡다" 같은 밝기 대비 패턴으로 눈·코 같은 부위를 잡습니다.
- **적분 영상(integral image)**: 이 밝기 합 계산을 엄청 빠르게 하는 수학 트릭.
- **Cascade(폭포식 검사)**: 쉬운 검사부터 차례로 걸러, "명백히 얼굴이 아닌 곳"을 초반에 빠르게 버립니다. 그래서 빠릅니다.
```

지금도 OpenCV의 `haarcascade`로 간단한 얼굴 탐지에 쓰입니다.

원 논문: Viola, Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", CVPR 2001 — [DOI](https://doi.org/10.1109/CVPR.2001.990517) · [OpenCV 실습](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

### ③ HOG + SVM (2005) — 보행자를 찾아라

자율주행·CCTV의 사람 탐지 계보가 여기서 시작됩니다.

```{admonition} HOG가 뭔가
:class: note

*Histogram of Oriented Gradients(기울기 방향 히스토그램)*. 이미지를 작은 칸으로 나눠, 각 칸에서 **밝기가 어느 방향으로 변하는지(=윤곽선의 방향)** 를 집계합니다. 사람 몸의 실루엣(외곽선) 패턴을 잘 잡아내며, 이렇게 뽑은 특징을 **SVM**이 "사람이다/아니다"로 판정합니다.
```

색이나 조명이 바뀌어도 윤곽 방향은 잘 안 변해서 강건합니다.

원 논문: Dalal, Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005 — [PDF](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) · [설명](https://learnopencv.com/histogram-of-oriented-gradients/)

### ④ DPM (2008–2010) — "물체는 부품들의 조합이다"

고전 CV의 정점으로 평가받는 기법입니다(2010년 당시 최고 성능).

```{admonition} 핵심 직관
:class: note

사람을 통째로 한 덩어리로 보지 않고 머리·몸통·팔·다리 같은 **부품(part)** 으로 나눠 표현합니다. 그리고 "부품들이 어느 정도는 움직여도(deformable) 사람"이라고 허용합니다. 그래서 자세가 바뀌어도 잘 찾습니다. 이름 그대로 **Deformable Part Model(변형 가능한 부품 모델)** 입니다.
```

흥미로운 사실은, DPM을 만든 연구자(Ross Girshick)가 곧이어 **R-CNN**으로 딥러닝 시대를 연다는 점입니다. 고전 CV의 마지막 거장이 다음 시대의 문을 열었습니다.

원 논문: Felzenszwalb, Girshick, McAllester, Ramanan, "Object Detection with Discriminatively Trained Part-Based Models", PAMI 2010 — [DOI](https://doi.org/10.1109/TPAMI.2009.167)

### ⑤ 특징점 3형제 — SIFT / SURF / ORB (지금도 현역)

탐지(네모 그리기)와는 조금 다르지만, 고전 CV의 또 다른 큰 줄기입니다. 두 이미지에서 같은 지점을 찾아 잇는(매칭) 기술로, 파노라마 합성·3D 복원·로봇 위치추정의 기반입니다.

```{admonition} 특징점(keypoint)이 뭔가
:class: note

사진에서 어디서 봐도 알아볼 수 있는 튀는 지점(모서리, 무늬의 교차점 등)입니다. 크기·회전이 바뀌어도 같은 점을 찾을 수 있게 만든 게 **SIFT**(2004, Lowe)이고, **SURF**는 그 빠른 버전, **ORB**(2011)는 더 가볍고 무료라 실시간·모바일에 인기입니다.
```

이 셋은 옛 기술이 아니라 지금도 SLAM·AR·산업검사에서 핵심으로 쓰입니다(아래 「고전 CV × 최신 AI」 참조).

원 논문(SIFT): Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004 — [PDF](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) · [OpenCV 튜토리얼](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)

```{admonition} 그 밖의 고전 CV 도구 — 현장에서 자주 쓰임
:class: tip

가장자리 검출 **Canny(1986)**, 직선·원 찾기 **Hough 변환**, 모양 따기 **Contour/윤곽**, 잘못된 매칭을 걸러내는 기하 검증 **RANSAC(1981)**, 형태 정리 **모폴로지(팽창·침식)**. 모두 OpenCV에 기본 탑재되어 있고, 지금도 전처리·후처리에서 널리 쓰입니다.
```

---

## 딥러닝 2단계 검출기: 기계가 특징을 스스로 배우다

2012년, 이미지 분류 대회(ImageNet)에서 **CNN(합성곱 신경망)** 이 압도적으로 우승하며 판도가 바뀝니다. "특징을 사람이 설계하지 말고, 기계가 데이터로 배우게 하자."

### ① R-CNN (2014) — 딥러닝 탐지의 시작

아이디어는 직관적입니다. ① **Selective Search**(고전 CV 기법)로 후보 영역 약 2000개를 추린 뒤 ② 각 영역을 **CNN**에 넣어 특징을 뽑고 ③ **SVM**으로 분류합니다. 정확도가 고전 CV 대비 크게 도약했습니다.

```{admonition} 재미있는 점
:class: note

R-CNN조차 후보 추리기는 여전히 고전 CV(Selective Search)에 의존했습니다. 즉 초기 딥러닝은 고전 CV와 동거하며 시작했습니다.
```

치명적 약점은 속도였습니다. 영역 2000개를 하나씩 CNN에 넣으니 한 장에 수십 초가 걸렸습니다.

원 논문: Girshick, Donahue, Darrell, Malik, "Rich feature hierarchies..." (R-CNN), CVPR 2014 — [arXiv:1311.2524](https://arxiv.org/abs/1311.2524)

### ② SPP-Net / Fast R-CNN (2014–2015) — 속도 개선

매번 영역마다 CNN을 처음부터 돌리는 게 낭비라는 걸 깨닫고, 이미지 전체에서 특징을 한 번만 뽑아 재활용하도록 바꿉니다. R-CNN보다 학습 9배, 추론 200배 이상 빨라졌습니다.

원 논문: Girshick, "Fast R-CNN", ICCV 2015 — [arXiv:1504.08083](https://arxiv.org/abs/1504.08083) · SPP-Net: He 등 — [arXiv:1406.4729](https://arxiv.org/abs/1406.4729)

### ③ Faster R-CNN (2015) — 2단계의 완성형

마지막 병목이던 후보 영역 추리기마저 **신경망(RPN, Region Proposal Network)** 으로 대체합니다. 드디어 고전 CV(Selective Search)를 완전히 떼어내고 전 과정을 신경망으로 통일했습니다. GPU에서 초당 약 17장 수준까지 올라 실시간을 넘봤습니다.

```{admonition} 여기서 등장하는 개념
:class: note

그 유명한 **anchor(앵커)** 개념이 본격화됩니다. 〈YOLO 변천사〉에서 다시 만나는 그 개념입니다.
```

원 논문: Ren, He, Girshick, Sun, "Faster R-CNN", NeurIPS 2015 — [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)

---

## YOLO로 가는 다리: "한 번에 보자"

2단계는 정확하지만, 후보를 추리고 다시 분류하는 두 단계 때문에 태생적으로 무겁습니다. 그래서 위치와 종류를 한 번에 예측하자는 **1단계(one-stage)** 흐름이 등장합니다.

```{mermaid}
flowchart LR
    A["2단계: 후보 추리기 → 분류"] -->|"정확하지만 느림"| B["1단계: 위치+종류 동시 예측"]
    B -->|"빠름 → 실시간"| C["YOLO (2016)"]
```

- **OverFeat (2013)**: 1단계 검출의 초기 아이디어
- **SSD (2016)**: 여러 해상도에서 한 번에 검출하는 1단계 모델
- **YOLO (2016)**: 이미지를 격자로 나눠 한 번에 푸는 1단계의 대표주자

```{admonition} 다음 문서로
:class: tip

바로 이 지점이 〈YOLO 변천사〉의 시작점입니다. "YOLO 이전의 탐지 모델은 2단계 방식이었습니다…"라는 첫 문장이 여기서 이어집니다.
```

---

## 고전 CV는 죽지 않았다: 최신 AI와의 결합

많은 사람이 "딥러닝이 고전 CV를 대체했다"고 오해합니다. 실제로는 역할을 나눠 함께 쓰입니다. 오히려 최신 AI 시스템일수록 고전 CV가 곳곳에 박혀 있습니다.

### ① 왜 고전 CV가 사라지지 않는가

1. **가볍다** — 학습이 필요 없고 CPU·저전력 기기에서도 즉시 돈다(엣지 환경에 유리).
2. **해석 가능하다** — "왜 그렇게 판정했는지" 사람이 따라갈 수 있다(안전·의료·검사처럼 설명이 중요한 분야).
3. **데이터가 거의 없어도 된다** — 딥러닝은 라벨이 산더미로 필요하지만, 고전 CV는 규칙 기반이라 소량·무데이터로도 동작.
4. **기하학적으로 정확하다** — 거리·각도·정합 같은 수학적으로 정밀한 계산은 여전히 고전 기법이 강하다.

### ② 실제 결합 사례 (현장에서 지금 쓰이는 것들)

- **SLAM / 로봇·AR·자율주행 위치추정**: 고전 특징점 기반 **ORB-SLAM**이 여전히 표준 축이고, 여기에 딥러닝 특징(**SuperPoint** 등)을 얹는 하이브리드 SLAM이 활발합니다. "기하학은 고전, 인식·강건성은 딥러닝" 식의 역할 분담입니다. ([VSLAM 서베이](https://www.sciencedirect.com/science/article/pii/S0957417422010156) · [안전모 통합 검사 사례](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12349484/))
- **특징 매칭의 진화**: 고전 SIFT/ORB → 딥러닝 **SuperPoint·LoFTR**로 발전했지만, 둘을 비교·혼용하는 게 현업의 일상입니다. ([SIFT vs ORB 실습 글](https://medium.com/@beauc_37732/comparing-sift-and-orb-for-feature-matching-a-visual-and-practical-exploration-6c194c72e4d6) · [특징 매칭 서베이](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.13032))
- **산업 검사(결함 탐지)**: 이미지 정합·스티칭(SIFT 등)으로 제품 표면을 펼쳐 보고, 그 위에 딥러닝 결함 분류를 얹는 고전+딥러닝 파이프라인. ([차체 검사 사례](https://pmc.ncbi.nlm.nih.gov/articles/PMC10891783/))
- **전처리(딥러닝의 입력 다듬기)**: Canny 에지·모폴로지·색공간 변환·ROI 잘라내기 등 고전 기법으로 불필요한 영역을 걸러 모델을 가볍고 빠르게 만듭니다.
- **추적(Tracking)**: 딥러닝 검출 + 고전 **칼만 필터**를 결합한 **ByteTrack** 같은 추적기가 사실상 표준입니다(탐지는 딥러닝, 프레임 간 연결은 고전 수학).
- **기하 검증**: 잘못된 매칭을 걸러내는 **RANSAC**은 딥러닝 매칭 결과에도 여전히 후처리로 붙습니다.

```{admonition} 결합의 한 줄 정리
:class: important

"인식·의미 파악은 딥러닝, 기하·정밀계산·전처리·후처리는 고전 CV" — 이 분담 구조가 최신 비전 시스템의 표준 설계입니다. 안전 모니터링에서도 (검출=딥러닝) + (좌표·거리·정합·추적=고전) 조합이 흔합니다.
```

---

## 핵심 요약

1. **개념의 뿌리**: anchor·sliding window·후보 영역·NMS·특징 등 지금 YOLO에서 쓰는 단어 대부분이 이 시대에 태어났다.
2. **대전환의 의미**: "사람이 특징을 설계" → "기계가 특징을 학습"으로 바뀐 것이 딥러닝 혁명의 본질이다.
3. **현재 진행형**: 고전 CV는 가볍고·해석가능하고·기하학적으로 정확해서, 최신 AI와 결합되어 더 오래 쓰일 가능성이 크다.

---

## 참고 문헌

**고전 CV (원문/DOI)**

- Viola, Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*, CVPR 2001 — [DOI](https://doi.org/10.1109/CVPR.2001.990517) · [위키 설명](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)
- Dalal, Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005 — [PDF](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) · [위키 설명](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
- Felzenszwalb, Girshick, McAllester, Ramanan, *Object Detection with Discriminatively Trained Part-Based Models* (DPM), PAMI 2010 — [DOI](https://doi.org/10.1109/TPAMI.2009.167)
- Lowe, *Distinctive Image Features from Scale-Invariant Keypoints* (SIFT), IJCV 2004 — [PDF](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- Bay 등, *SURF: Speeded Up Robust Features*, ECCV 2006 — [DOI](https://doi.org/10.1007/11744023_32)
- Rublee 등, *ORB: an efficient alternative to SIFT or SURF*, ICCV 2011 — [OpenCV 문서](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
- Uijlings 등, *Selective Search for Object Recognition*, IJCV 2013 — [DOI](https://doi.org/10.1007/s11263-013-0620-5)

**딥러닝 2단계·다리 (arXiv)**

- Girshick 등, [*Rich feature hierarchies (R-CNN)*](https://arxiv.org/abs/1311.2524), CVPR 2014
- He 등, [*SPP-Net*](https://arxiv.org/abs/1406.4729), 2014
- Girshick, [*Fast R-CNN*](https://arxiv.org/abs/1504.08083), ICCV 2015
- Ren, He, Girshick, Sun, [*Faster R-CNN*](https://arxiv.org/abs/1506.01497), NeurIPS 2015
- Sermanet 등, [*OverFeat*](https://arxiv.org/abs/1312.6229), ICLR 2014
- Liu 등, [*SSD: Single Shot MultiBox Detector*](https://arxiv.org/abs/1512.02325), ECCV 2016

**고전 CV × 딥러닝 결합 (사례·서베이)**

- [Visual SLAM 종합 서베이](https://www.sciencedirect.com/science/article/pii/S0957417422010156)
- [딥러닝 강화 하이브리드 SLAM (Rover-SLAM)](https://arxiv.org/pdf/2405.03413)
- [특징 매칭 방법 서베이 (고전+딥러닝)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.13032)
- [산업용 이미지 정합·검사 사례](https://pmc.ncbi.nlm.nih.gov/articles/PMC10891783/)
- [VSLAM 안전모 통합 검사 리뷰](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12349484/)

**더 배우기 좋은 자료 (입문용)**

- Szeliski, *Computer Vision: Algorithms and Applications* — 무료 PDF · [szeliski.org/Book](https://szeliski.org/Book/)
- First Principles of Computer Vision (Shree Nayar, 컬럼비아대) · [fpcv.cs.columbia.edu](https://fpcv.cs.columbia.edu/)
- OpenCV 공식 튜토리얼 · [docs.opencv.org](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- PyImageSearch · [pyimagesearch.com](https://pyimagesearch.com/)
- LearnOpenCV · [learnopencv.com](https://learnopencv.com/)
- 객체 탐지 역사 종합 리뷰(고전→CNN) · [arXiv:2412.05252](https://arxiv.org/abs/2412.05252)

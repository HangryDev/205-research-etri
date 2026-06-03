# 섹션 1 | Object Detection — YOLO가 어떻게 그렇게 빠른가

---

## 1-1. 문제 제기

**상황**: 공장 CCTV로 실시간 안전 모니터링을 구축하려면 어떤 방식이 필요한가?

### 클라우드 분석의 현실적 문제

- 모든 영상을 GPU 서버로 보내서 분석 → 세 가지 치명적 문제
  - **지연 시간**: 0.5~2초 (영상 왕복). 사고는 0.5초 안에도 일어남
  - **네트워크 비용**: 30FPS 풀HD 1채널 = 시간당 약 1.5GB. 카메라 50대면 하루 수 TB
  - **보안**: 공장 내부 영상(라인 배치, 공정 순서, 인력 동선)이 외부로 전송됨

```{mermaid}
flowchart LR
    A[카메라] --> B[인터넷] --> C[GPU 서버] --> D[결과] --> E[카메라]
```

![GPU starvation — CPU가 데이터를 준비하는 동안 GPU가 대기](../../lecture/images/s3_1_img01.png)

- CPU가 데이터를 준비하는 동안 **GPU가 놀고 있는(starvation)** 상황을 도식화
- 연산 자원이 있어도 입력 공급이 늦으면 처리량이 안 나옴 → 파이프라인 병목의 개념
- 영상을 멀리 보내 처리하는 클라우드 방식의 비효율(지연·대기)을 보여주는 맥락

![nvidia-smi 터미널 출력 — GPU 활용률 확인](../../lecture/images/s3_1_img02.png)

- `nvidia-smi` 터미널 출력 — 실제 **GPU 활용률·메모리 사용량**을 확인하는 화면
- 활용률이 낮게 찍히면 모델이 아니라 데이터 공급이 병목일 가능성
- 성능 문제를 진단하는 가장 기본적인 도구

![TensorBoard 프로파일러 타임라인 — GPU 유휴 시간 시각화](../../lecture/images/s3_1_img03.png)

- TensorBoard 프로파일러의 타임라인 — **GPU 유휴 구간**을 시각적으로 드러냄
- 어느 단계에서 시간이 새는지(데이터 로딩 vs 연산)를 한눈에 파악
- "어디를 최적화할지"에 대한 근거를 제공

### 해결 방향: 엣지 컴퓨팅

- 카메라 옆에 **작은 컴퓨터(엣지 디바이스)** 를 두고 현장에서 바로 분석
- Jetson Nano, Raspberry Pi 등 활용
- 영상은 로컬 처리, 결과 한 줄만 서버 전송 ("12번 라인 안전모 미착용 의심")

```{mermaid}
flowchart LR
    A[카메라] --> B["Jetson Nano / 라즈베리파이"]
    B --> C[즉시 결과]
```

```{admonition} 핵심 질문
:class: important

GPU가 없는, 손바닥만 한 컴퓨터에서 실시간 객체 탐지가 가능한가?
**정확도 vs 속도의 트레이드오프**를 어떻게 풀 것인가?
```

---

## 1-2. 이론

### ① Object Detection의 발전: 왜 YOLO인가

> *Practical Deep Learning* Appendix A — CNN 기본 구조 / Ch.14 — CNN 기반 탐지 모델 발전 흐름

**2-Stage Detector: R-CNN 계열** (R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN)

```{mermaid}
flowchart TD
    A[입력 이미지] --> B["Region Proposal<br/>(후보 영역 추출)"]
    B --> C["각 영역에서 분류 + 위치 보정"]
    C --> D["정확도: 높음<br/>속도: 느림 2~10 FPS → 실시간 불가"]
```

R-CNN 계열은 먼저 후보 영역을 수백~수천 개 생성한 뒤 각 영역을 분류하는 2단계 방식입니다. 정확도는 높지만 속도가 느려 **실시간이 불가능** 한데, CCTV의 30FPS는커녕 그 1/10도 따라가지 못합니다.

**1-Stage Detector: YOLO 계열** (YOLO, SSD, RetinaNet)

```{mermaid}
flowchart LR
    A[입력 이미지] --> B["단 한 번의 추론으로<br/>위치 + 클래스 동시 예측"]
    B --> C["정확도: 약간 낮음<br/>속도: 빠름 30~100+ FPS → 실시간 가능"]
```

반면 YOLO 계열은 단 한 번의 추론으로 위치와 클래스를 **동시에 예측** 합니다. 이름 그대로 "You Only Look Once", 한 번만 본다는 뜻입니다. 정확도는 약간 낮지만 **30~100+ FPS** 로 실시간 처리가 가능합니다.

```{admonition} 핵심
:class: important

**YOLO**: 이미지를 한 번만 보고 모든 객체를 동시에 예측.
제조 현장 실시간 안전 모니터링에는 YOLO 계열이 **사실상의 표준**.
정확도 1~2% 양보하고 속도 10배를 얻는 트레이드오프.
```

**YOLO의 한계** (*Practical ML for Computer Vision* Ch.4 지적):

다만 YOLO에도 한계가 있습니다. 한 그리드 셀당 하나의 클래스만 예측하기 때문에 작은 객체가 떼로 몰려 있으면 잘 못 잡고, 마지막 feature map만 사용해 작은 객체 탐지가 약합니다. 이런 한계는 RetinaNet, FPN 같은 후속 아키텍처가 보완합니다.

![Object Detection 아키텍처 발전 타임라인](../../lecture/images/s3_1_img16.png)

- R-CNN→YOLO 등 객체 탐지 **아키텍처의 발전 흐름**을 연표로 정리
- 2-stage(정확·느림)에서 1-stage(빠름)로 이어지는 큰 흐름을 보여줌
- "왜 YOLO인가"에 대한 역사적 배경

![탐지 아키텍처와 백본이 mAP에 미치는 영향](../../lecture/images/s3_1_img17.png)

- 탐지 구조와 백본(backbone) 선택이 **정확도(mAP)** 에 미치는 영향을 비교
- 백본이 강할수록 정확도↑이지만 연산량도↑ → 트레이드오프
- 모델을 고를 때의 판단 근거

![배치 사이즈 변화에 따른 에포크당 시간과 GPU 활용률](../../lecture/images/s3_1_img05.png)

- 배치 크기를 키울 때 **에포크당 시간과 GPU 활용률**이 어떻게 변하는지
- 배치가 커지면 GPU를 더 꽉 채워 효율이 올라감(메모리 한계까지)
- 학습 처리량을 튜닝하는 직관

---

### ② YOLOv8 구조 직관

> *Practical Deep Learning* Ch.14 — Backbone/Neck/Head 구조 / Ch.6 — 추론 최적화와 모델 경량화

YOLOv8은 **세 부분**으로 구성됩니다.

```{mermaid}
flowchart TD
    A["입력 이미지 (640×640)"] --> B["Backbone (CSPDarknet)<br/>역할: 이미지에서 특징 추출<br/>→ 이미지 안에 무엇이 있는지 감지"]
    B --> C["Neck (FPN + PAN)<br/>역할: 다양한 크기의 특징 결합<br/>→ 멀리 있는 것도, 가까운 것도 탐지"]
    C --> D["Head (Decoupled)<br/>역할: 위치(bbox) + 클래스 동시 예측<br/>→ 어디에, 무엇이, 얼마나 확실한가"]
```

**Backbone(CSPDarknet)** 은 이미지에서 특징을 추출해 텍스처·모서리·형태·색상을 점점 더 추상화된 표현으로 압축합니다. **Neck(FPN + PAN)** 은 깊은 층(의미는 풍부하나 해상도가 낮음)과 얕은 층(해상도는 좋으나 의미가 빈약함)을 결합해 크기에 상관없이 탐지할 수 있게 합니다. **Head(Decoupled)** 는 위치 예측(회귀)과 클래스 예측(분류)을 분리하는데, 학습 신호가 서로 다르기 때문에 분리하면 학습이 더 안정적입니다.

**YOLOv8 모델 크기 계열**:

| 모델 | 파라미터 수 | 속도 | 정확도 | 배포 대상 |
|------|-----------|------|--------|----------|
| YOLOv8n (nano) | 3.2M | 최고 | 낮음 | 엣지 디바이스 |
| YOLOv8s (small) | 11.2M | 빠름 | 중간 | 중간 사양 |
| YOLOv8m (medium) | 25.9M | 중간 | 높음 | 서버 |
| YOLOv8l (large) | 43.7M | 느림 | 더 높음 | 서버 |
| YOLOv8x (xlarge) | 68.2M | 최저 | 최고 | GPU 서버 |

```{admonition} 팁
:class: tip

같은 모델 패밀리 안에서 **정확도-속도 트레이드오프를 슬라이더처럼 조절** 가능.
엣지 → n/s, 서버 → m/l, 최고 정확도 → x.
```

![학습률 변화에 따른 손실 그래프](../../lecture/images/s3_1_img06.png)

- 학습률(learning rate)을 바꿀 때 **손실 곡선**이 어떻게 달라지는지
- 너무 크면 발산, 너무 작으면 너무 느림 → 적정 학습률이 존재
- 하이퍼파라미터 튜닝의 기본 감각

![학습률 변화에 따른 손실 변화율](../../lecture/images/s3_1_img07.png)

- 학습률에 따른 **손실 감소 속도(변화율)** 를 본 그림
- 손실이 가장 빠르게 줄어드는 구간이 좋은 학습률 후보 (LR finder 아이디어)
- 적정 학습률을 체계적으로 고르는 방법

![모델별 크기, 정확도, 초당 연산량 비교](../../lecture/images/s3_1_img10.png)

- 여러 모델의 **크기 vs 정확도 vs 초당 연산량**을 한 번에 비교
- 정확도와 속도가 서로 상충함을 수치로 보여줌
- 배포 환경(엣지/서버)에 맞는 모델 선택의 근거

---

### ③ 경량화의 핵심: Depthwise Separable Convolution

> *Practical Deep Learning* Appendix A — 일반 Convolution 연산량 분석 / Ch.6 — MobileNet의 Depthwise Separable Convolution

**일반 Convolution의 연산량**:

$$\text{FLOPs} = H \times W \times K^2 \times C_{in} \times C_{out}$$

예를 들어 224×224 이미지에 3×3 필터, $C_{in}=64$, $C_{out}=128$ 이면 한 레이어에만 약 **3.7억 번** 의 곱셈이 필요합니다.

**MobileNet의 해결책**: 한 단계 연산을 **두 단계로 쪼갬**

```{mermaid}
flowchart LR
    A["일반 Conv<br/>K×K×C_in 필터<br/>× C_out"] --> B["Depthwise Conv<br/>K×K×1 필터<br/>× C_in"]
    B --> C["Pointwise Conv 1×1<br/>1×1×C 필터<br/>× C_out"]
```

| 단계 | 역할 | 설정 |
|------|------|------|
| **Depthwise Conv** | 각 채널마다 따로 3×3 필터 적용 (공간만 처리) | `groups=in_channels` |
| **Pointwise Conv** | 1×1 필터로 채널 간 정보 결합 (채널만 처리) | `kernel_size=1` |

이렇게 쪼개면 **연산량이 약 1/8~1/9 로 줄어드는** 대신, **정확도 손실은 1~2%에 불과** 합니다.

```python
import torch.nn as nn

# 일반 Convolution
standard_conv = nn.Conv2d(
    in_channels=64, out_channels=128,
    kernel_size=3, padding=1
)

# Depthwise Separable Convolution (MobileNet 방식)
depthwise = nn.Conv2d(
    in_channels=64, out_channels=64,
    kernel_size=3, padding=1,
    groups=64  # groups=in_channels → 채널별 독립 적용
)
pointwise = nn.Conv2d(
    in_channels=64, out_channels=128,
    kernel_size=1  # 1×1 conv로 채널 수 조정
)
```

**양자화(Quantization)** (*Practical Deep Learning* Ch.6 "Quantize the Model"):

양자화는 FP32(32비트) 가중치를 **INT8(8비트)** 로 저장하는 기법입니다. 모델 크기가 **4배 축소** 되고 추론 속도가 **2~3배 향상** 되며 정확도 손실은 **1% 미만** 으로, TensorFlow Lite·Core ML·TensorRT 등이 지원합니다.

```{admonition} 팁
:class: tip

**실무 판단 기준**:
- 엣지 디바이스 → YOLOv8n 또는 MobileNet 기반 모델
- 서버 배포 → YOLOv8m 이상으로 정확도 우선
- 엣지 + 정확도 모두 중요 → **Quantization** 적용

**기억할 키워드**: Depthwise Separable Convolution + Quantization.
둘 다 "정확도를 조금 양보하고 속도와 크기를 크게 얻는" 같은 철학.
```

![모바일 친화적 모델 비교](../../lecture/images/s3_1_img11.png)

- MobileNet 등 **경량 모델들의 정확도-효율**을 비교
- 적은 연산으로도 쓸 만한 정확도를 내는 모델군을 보여줌
- 엣지 배포 후보를 고르는 참고 자료

![float32에서 int8로 양자화 — 저장 공간 축소](../../lecture/images/s3_1_img12.png)

- 32비트(float32) 가중치를 **8비트(int8)로 양자화**해 저장 공간이 줄어드는 모습
- 모델 크기 약 4배 축소·추론 가속, 정확도 손실은 1% 미만
- 엣지 배포를 위한 핵심 경량화 기법

---

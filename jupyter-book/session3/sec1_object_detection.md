# 섹션 1 | Object Detection — YOLO가 어떻게 그렇게 빠른가

```{admonition} 참고 교재
:class: note

**참고 교재**: *Practical Deep Learning for Cloud, Mobile, and Edge* (Anirudh Koul et al., O'Reilly)
```

---

## 1-1. 문제 제기

**상황**: 공장 CCTV로 실시간 안전 모니터링을 하려고 합니다.
그런데 GPU 서버에 영상을 보내서 분석하면 어떤 문제가 생길까요?

```{mermaid}
flowchart LR
    A[카메라] --> B[인터넷] --> C[GPU 서버] --> D[결과] --> E[카메라]
```

```
[클라우드 분석의 현실적 문제]

지연 시간: 0.5~2초 (이미 사고가 난 후)
네트워크 비용: 24시간 영상 전송 = 막대한 트래픽
보안: 공장 내부 영상이 외부로 전송됨
```

**해결 방향**: 카메라 옆에 작은 컴퓨터(엣지 디바이스)를 두고 현장에서 바로 분석

```{mermaid}
flowchart LR
    A[카메라] --> B["Jetson Nano / 라즈베리파이"]
    B --> C[즉시 결과]
```

```
[엣지 컴퓨팅 구조]

         모델이 여기서 실행
         지연 없음, 네트워크 불필요
```

**핵심 질문**: GPU가 없는 작은 컴퓨터에서 실시간 탐지가 가능할까?
→ 정확도 vs 속도의 트레이드오프를 어떻게 해결할 것인가?

---

## 1-2. 이론

### ① Object Detection의 발전: 왜 YOLO인가

```{admonition} 참고 교재
:class: note

📖 *Practical Deep Learning for Cloud, Mobile, and Edge* **Ch.3 Preparing Your Data** — 딥러닝 모델 구조의 이해와 CNN 기반 탐지 모델의 발전 흐름
```

딥러닝 기반 Object Detection은 크게 두 계열로 나뉩니다.

```{mermaid}
flowchart TD
    A[입력 이미지] --> B["Region Proposal<br/>(후보 영역 추출)"]
    B --> C["각 영역에서 분류 + 위치 보정"]
    C --> D["정확도: 높음<br/>속도: 느림 (2~10 FPS) → 실시간 불가"]
```

```
[2-Stage Detector: R-CNN 계열]

입력 이미지
    ↓
Region Proposal (후보 영역 추출)  ← 1단계: 어디를 볼까?
    ↓
각 영역에서 분류 + 위치 보정      ← 2단계: 무엇이 있나?

정확도: 높음
속도: 느림 (2~10 FPS) → 실시간 불가
```

```{mermaid}
flowchart LR
    A[입력 이미지] --> B["단 한 번의 추론으로<br/>위치 + 클래스 동시 예측"]
    B --> C["정확도: 약간 낮음<br/>속도: 빠름 (30~100+ FPS) → 실시간 가능"]
```

```
[1-Stage Detector: YOLO 계열]

입력 이미지
    ↓
단 한 번의 추론으로 위치 + 클래스 동시 예측

정확도: 약간 낮음
속도: 빠름 (30~100+ FPS) → 실시간 가능
```

```{admonition} 핵심
:class: important

**YOLO(You Only Look Once)**: 이름 그대로 이미지를 한 번만 보고 모든 객체를 동시에 예측합니다.
실시간 산업 안전 모니터링에는 YOLO 계열이 사실상 표준입니다.
```

---

### ② YOLOv8 구조 직관

```{admonition} 참고 교재
:class: note

📖 *Practical Deep Learning for Cloud, Mobile, and Edge* **Ch.3 Preparing Your Data** — CNN 특징 추출 구조 / **Ch.6 Maximizing Speed of an Existing Mobile Model** — Backbone 경량화와 추론 최적화
```

YOLOv8은 세 부분으로 구성됩니다.

```{mermaid}
flowchart TD
    A["입력 이미지 (640×640)"] --> B["Backbone (CSPDarknet)<br/>역할: 이미지에서 특징 추출<br/>→ 이미지 안에 무엇이 있는지 감지"]
    B --> C["Neck (FPN + PAN)<br/>역할: 다양한 크기의 특징 결합<br/>→ 멀리 있는 것도, 가까운 것도 탐지"]
    C --> D["Head (Decoupled)<br/>역할: 위치(bbox) + 클래스 동시 예측<br/>→ 어디에, 무엇이, 얼마나 확실한가"]
```

**YOLOv8 모델 크기 계열**:

```
YOLOv8n (nano)   → 3.2M 파라미터,  최고 속도, 낮은 정확도  ← 엣지용
YOLOv8s (small)  → 11.2M 파라미터, 빠름, 중간 정확도
YOLOv8m (medium) → 25.9M 파라미터, 중간 속도, 높은 정확도
YOLOv8l (large)  → 43.7M 파라미터, 느림, 더 높은 정확도
YOLOv8x (xlarge) → 68.2M 파라미터, 최저 속도, 최고 정확도 ← 서버용
```

---

### ③ 경량화의 핵심: Depthwise Separable Convolution

```{admonition} 참고 교재
:class: note

📖 *Practical Deep Learning for Cloud, Mobile, and Edge* **Ch.6 Maximizing Speed of an Existing Mobile Model** — MobileNet의 Depthwise Separable Convolution 원리와 연산량 절감 분석
```

**왜 일반 Convolution은 엣지 디바이스에서 느린가?**

```
[일반 Convolution 연산량]

입력: H × W × C_in (이미지 크기 × 채널 수)
필터: K × K × C_in × C_out

연산량 = H × W × K² × C_in × C_out

예) 224×224 이미지, 3×3 필터, C_in=64, C_out=128
   = 224 × 224 × 9 × 64 × 128 = 약 3.7억 번 연산
```

**MobileNet의 해결책: Depthwise Separable Convolution**

```{mermaid}
flowchart LR
    A["일반 Conv<br/>K×K×C_in 필터<br/>× C_out"] --> B["Depthwise Conv<br/>K×K×1 필터<br/>× C_in"]
    B --> C["Pointwise Conv (1×1)<br/>1×1×C 필터<br/>× C_out"]
```

```
[일반 Conv]           [Depthwise Separable Conv]
하나의 필터가           Depthwise Conv: 채널별로 따로 적용
모든 채널을 한 번에      + Pointwise Conv(1×1): 채널 결합
처리

┌──────────────┐      ┌──────────────┐ ┌────────────┐
│ K×K×C_in 필터│  →   │ K×K×1 필터   │+│ 1×1×C 필터  │
│ × C_out      │      │ × C_in       │ │ × C_out    │
└──────────────┘      └──────────────┘ └────────────┘

연산량 감소: 약 1/8 ~ 1/9 수준
정확도 손실: 경미 (1~2%)
```

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

```{admonition} 팁
:class: tip

**실무 판단 기준**:
엣지 디바이스 목표 → YOLOv8n 또는 MobileNet 기반 모델 선택
서버 배포 목표 → YOLOv8m 이상으로 정확도 우선
엣지 + 정확도 모두 중요 → Quantization 적용 (심화 파트 참조)
```

---

## 1-3. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: 모델 크기에 따른 속도-정확도 트레이드오프를 **숫자로 확인**하는 과정에 집중하세요.
```

**Claude에게 던질 프롬프트 예시**:
```
YOLOv8로 안전모 탐지 데모를 만들어줘.
- ultralytics 라이브러리 사용
- 모델 3개 비교: YOLOv8n, YOLOv8s, YOLOv8m
- 동일한 테스트 이미지(공장 작업자 사진)로 각 모델 추론
- 결과 출력:
  1. 각 모델의 추론 시간(ms)과 탐지된 객체 수
  2. bbox 시각화 (subplot 3개)
  3. 속도 vs mAP 비교 막대그래프
- 테스트 이미지는 numpy로 합성해줘 (실제 이미지 없이 테스트)
```

**시연 흐름**:
1. YOLOv8 모델 3개 로드
2. 동일 이미지에서 각각 추론 실행
3. 추론 시간 측정 및 결과 bbox 시각화
4. 속도·정확도 비교 그래프 출력
5. **Claude에게 추가 질문**: *"모델 크기를 반으로 줄이면 속도가 두 배가 되는 건 아닌데, 왜 그런 거야?"*

---

## 1-4. 실습

### 과제

YOLOv8 모델 크기별 속도-정확도 트레이드오프를 직접 측정하세요.

| 모델 | 파라미터 수 | 추론 시간(ms) | 탐지율 | 선택 기준 |
|------|-----------|------------|------|---------|
| YOLOv8n | 3.2M | 측정 | 측정 | 엣지 디바이스 |
| YOLOv8s | 11.2M | 측정 | 측정 | 중간 사양 |
| YOLOv8m | 25.9M | 측정 | 측정 | 서버 |

**제출 항목**:
- 3개 모델의 추론 시간 + 탐지 결과 비교 표
- 각 모델의 bbox 시각화 이미지
- "라즈베리파이 환경(CPU only)에서 어떤 모델을 선택할 것인가?" 한 문단 + 이유

### 실습 시작 코드

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 테스트 이미지 생성 (실제 이미지가 없을 때)
# 실제 실습 시에는 공장 작업자 이미지 파일 경로로 교체
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

models = {
    'YOLOv8n': YOLO('yolov8n.pt'),
    'YOLOv8s': YOLO('yolov8s.pt'),
    'YOLOv8m': YOLO('yolov8m.pt'),
}

results_summary = {}

for model_name, model in models.items():
    # 워밍업 (첫 추론은 느림)
    _ = model(test_image, verbose=False)

    # 추론 시간 측정 (10회 평균)
    times = []
    for _ in range(10):
        start = time.time()
        result = model(test_image, verbose=False)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

    avg_time = np.mean(times)
    n_detections = len(result[0].boxes) if result[0].boxes else 0

    results_summary[model_name] = {
        'avg_time_ms': avg_time,
        'n_detections': n_detections,
    }
    print(f"{model_name}: {avg_time:.1f}ms, 탐지 수: {n_detections}")

# TODO: 결과를 subplot으로 시각화
# TODO: 속도 vs 탐지 수 막대그래프 추가
```

---

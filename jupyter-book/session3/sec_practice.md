# [실습 시 프롬프트 제안] 세션 3 — Claude Code 시연 & 실습 모음


> 세션 3(Vision AI 안전 모니터링)의 각 강의 섹션에 있던 **Claude Code 시연**과 **실습** 부분을 한곳에 모았습니다.
> 각 항목의 이론·배경 설명은 해당 강의 섹션을 참고하세요.


---

```{admonition} 출처: Object Detection
:class: note dropdown
아래 시연/실습은 원래 Object Detection** 섹션에 있던 내용입니다.
```

## 1-3. Claude Code 시연

**시연 포인트**: 모델 크기에 따른 **속도-정확도 트레이드오프**를 숫자로 확인하는 과정에 집중.

### 프롬프트

```
YOLOv8로 안전모 탐지 데모를 만들어줘.
- ultralytics 라이브러리 사용
- 모델 3개 비교: YOLOv8n, YOLOv8s, YOLOv8m
- 동일한 테스트 이미지로 각 모델 추론
- 결과: 각 모델의 추론 시간(ms), 탐지 수, bbox 시각화, 속도 vs mAP 비교
```

### 시연 흐름

- `from ultralytics import YOLO` — 가중치 파일 자동 다운로드
- 테스트 이미지 준비 (640×640 더미 이미지로 속도 비교)
- **워밍업** 필수: 첫 추론은 CUDA 초기화 비용 포함 → 측정값에서 제외
- 10회 반복 측정 후 **평균** 산출

```{admonition} 시연 후 질문
:class: warning

"모델 크기를 반으로 줄이면 속도가 정확히 두 배가 되는 건 아닌데, 왜 그런 거야?"

이유 3가지:
1. GPU 메모리 대역폭과 연산 처리량 비율이 모델마다 다름 (작은 모델은 메모리 접근이 병목)
2. 후처리(NMS 등) 시간은 모델 크기와 무관한 **고정 비용**
3. GPU 워프(warp) 단위로 작업이 묶이므로 일정 크기 전까지는 모델이 커도 안 느려짐
```

---

## 1-4. 실습

**목표**: 세 가지 모델로 추론 시간과 탐지 결과를 비교하는 표를 완성

| 모델 | 파라미터 수 | 추론 시간(ms) | 탐지율 | 선택 기준 |
|------|-----------|-------------|--------|----------|
| YOLOv8n | 3.2M | 측정 | 측정 | 엣지 디바이스 |
| YOLOv8s | 11.2M | 측정 | 측정 | 중간 사양 |
| YOLOv8m | 25.9M | 측정 | 측정 | 서버 |

### STEP 1 — 테스트 이미지 준비

```python
import time, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO

test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
```

### STEP 2 — 3가지 모델 로드

```python
models = {
    'YOLOv8n': YOLO('yolov8n.pt'),
    'YOLOv8s': YOLO('yolov8s.pt'),
    'YOLOv8m': YOLO('yolov8m.pt'),
}
```

### STEP 3 — 추론 시간 측정

```python
results_summary = {}
for model_name, model in models.items():
    _ = model(test_image, verbose=False)      # 워밍업 (측정에서 제외)
    times = []
    for _ in range(10):
        start = time.time()
        result = model(test_image, verbose=False)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    avg_time = np.mean(times)
    n_detections = len(result[0].boxes) if result[0].boxes else 0
```

### STEP 4 — 결과 시각화

```python
    results_summary[model_name] = {'avg_time_ms': avg_time, 'n_detections': n_detections}
    print(f"{model_name}: {avg_time:.1f}ms, 탐지 수: {n_detections}")
# TODO: 속도 vs 탐지 수 막대그래프
```

```{admonition} 성공 체크리스트
:class: tip

- n < s < m 순서로 추론 시간이 **단조 증가**하면 성공
- 막대그래프나 산점도로 속도 대 탐지 수 트레이드오프를 한 화면에서 확인

**도전 과제**:
- 입력 해상도를 320, 640, 1280으로 변경하며 영향 측정
- `device='cpu'` vs `device='cuda'` 비교
```

---

```{admonition} 출처: Action Recognition
:class: note dropdown
아래 시연/실습은 원래 Action Recognition** 섹션에 있던 내용입니다.
```

## 2-3. Claude Code 시연

**시연 포인트**: MediaPipe 출력을 LSTM 입력으로 **어떻게 변환하는가**. 두 도구 사이의 데이터 흐름을 손으로 잡는 게 목표.

### 프롬프트

```
작업자 위험 행동 인식 모델을 만들어줘.
- MediaPipe Pose로 웹캠/영상에서 키포인트 추출
- window_size=30 프레임으로 슬라이딩 윈도우 생성
- LSTM(64) → Dropout(0.3) → LSTM(32) → Dense(3, softmax)
- 행동 클래스: ["정상작업", "허리위험각도", "고소작업위험"]
- 합성 데이터로 학습
```

### 시연 단계별 흐름

1. **합성 데이터 생성**: 정상은 무작위 잡음만, 허리 위험은 엉덩이 키포인트 점진 변화, 고소 위험은 y좌표 전반 위쪽
2. **윈도우 생성**: `create_sequences`로 (배치, 30, 99) shape 생성
3. **학습**: 패턴이 명확하므로 빠르게 수렴 (합성 데이터 기준 정확도 90% 후반)

```{admonition} 시연 후 질문
:class: warning

"카메라가 정면이 아니라 **측면**에서 찍히면 키포인트가 어떻게 달라지고, 어떻게 대응해야 할까?"

측면 문제: 한쪽 팔/다리가 가려짐 → visibility 값 하락

대응 3가지:
1. **visibility 가중치**: 낮은 키포인트의 영향을 마스킹 또는 가중치로 감소
2. **다중 카메라 융합**: 정면 + 측면 카메라, 더 잘 보이는 쪽 우선 사용 (비싸지만 효과적)
3. **학습 데이터에 다양한 각도 포함**: 측면, 후면, 비스듬한 각도를 골고루 섞어 각도 강건화 (가장 근본적)
```

---

## 2-4. 실습

**목표**: 윈도우 크기를 바꾸면서 **지연 시간과 정확도의 트레이드오프**를 직접 체험

| 실험 | window_size | 지연 시간 | 정확도 | 관찰 포인트 |
|------|-----------|----------|--------|------------|
| A | 15 프레임 | 0.5초 | 측정 | 빠른 반응, 짧은 맥락 |
| B | 30 프레임 | 1.0초 | 측정 | 기본값 |
| C | 60 프레임 | 2.0초 | 측정 | 긴 맥락, 느린 반응 |

### STEP 1 — 합성 키포인트 데이터 생성

```python
import numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(42)
N_SAMPLES, N_CLASSES, N_KEYPOINTS = 300, 3, 99

def generate_synthetic_data(window_size, n_samples=N_SAMPLES):
    X, y = [], []
    for cls in range(N_CLASSES):
        for _ in range(n_samples // N_CLASSES):
            base = np.random.randn(window_size, N_KEYPOINTS) * 0.1
            if cls == 1: base[:, 23*3:25*3] += np.linspace(0, 1, window_size)[:, None]
            elif cls == 2: base[:, 1::3] -= 0.5
            X.append(base); y.append(cls)
    return np.array(X, dtype=np.float32), np.array(y)
```

### STEP 2 — LSTM 모델 빌더

```python
def build_model(window_size, n_classes=N_CLASSES):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(window_size, N_KEYPOINTS),
                             return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
```

### STEP 3 — 세 가지 window_size로 실험

```python
results = {}
for window_size in [15, 30, 60]:
    X, y = generate_synthetic_data(window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(window_size)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # TODO: 학습 → 평가 → 추론 시간 측정
```

```{admonition} 성공 체크리스트
:class: tip

- window_size가 커질수록 정확도가 약간 **오르고** 추론 시간도 **늘어나는** 단조 관계가 보이면 성공
- 정확도가 60→30→15 순으로 떨어지지 않거나 추론 시간이 역전되면 데이터/측정 오류

**도전 과제**: window_size=30 고정, step을 5, 15, 30으로 변경 → 결과 갱신 빈도와 계산 비용의 트레이드오프 관찰
```

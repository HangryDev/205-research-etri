# 세션 3 실습 — Claude Code 시연 & 실습 모음


> 세션 3(Vision AI 안전 모니터링)의 각 강의 섹션에 흩어져 있던 **Claude Code 시연**과 **실습** 부분을 한곳에 모았습니다.
> 각 항목의 이론·배경 설명은 해당 강의 섹션을 참고하세요.


---

```{admonition} 출처: 섹션 1 · Object Detection
:class: note dropdown
아래 시연/실습은 원래 **섹션 1 · Object Detection** 섹션에 있던 내용입니다.
```

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

```{admonition} 출처: 섹션 2 · Action Recognition
:class: note dropdown
아래 시연/실습은 원래 **섹션 2 · Action Recognition** 섹션에 있던 내용입니다.
```

## 2-3. Claude Code 시연

```{admonition} 팁
:class: tip

**시연 포인트**: MediaPipe로 키포인트를 추출하고 LSTM 입력으로 변환하는 **연결 과정**에 집중하세요.
```

**Claude에게 던질 프롬프트 예시**:
```
작업자 위험 행동 인식 모델을 만들어줘.
- MediaPipe Pose로 웹캠/영상에서 키포인트 추출
- window_size=30 프레임으로 슬라이딩 윈도우 생성
- LSTM(64) → Dropout(0.3) → LSTM(32) → Dense(3, softmax)
- 행동 클래스: ["정상작업", "허리위험각도", "고소작업위험"]
- 합성 데이터로 학습 (실제 데이터 없이 테스트 가능하게)
- 실시간 예측 결과를 프레임에 오버레이 (행동명 + 확률)
```

**시연 흐름**:
1. MediaPipe로 샘플 영상에서 키포인트 추출 및 시각화
2. 키포인트 → LSTM 입력 형태 변환 확인
3. 합성 데이터로 LSTM 학습
4. 실시간 예측 결과 프레임 오버레이
5. **Claude에게 추가 질문**: *"카메라 각도가 정면이 아닌 측면에서 찍히면 키포인트 좌표가 어떻게 달라지고, 이걸 어떻게 대응해야 해?"*

---

## 2-4. 실습

### 과제

`window_size`(분류에 사용할 프레임 수)를 바꿔가며 행동 인식 성능과 지연 시간의 트레이드오프를 분석하세요.

| 실험 | window_size | 지연 시간 | 정확도 | 관찰 포인트 |
|------|-----------|---------|------|----------|
| A | 15 프레임 | 0.5초 | 측정 | 빠른 반응, 짧은 맥락 |
| B | 30 프레임 | 1.0초 | 측정 | 기본값 |
| C | 60 프레임 | 2.0초 | 측정 | 긴 맥락, 느린 반응 |

**제출 항목**:
- window_size별 행동 분류 정확도(합성 데이터 기준) 비교 표
- "실제 현장에서 window_size를 얼마로 설정할 것인가?" 한 문단 + 이유
  (힌트: 위험 행동이 얼마나 짧게 발생하는가를 고려)

### 실습 시작 코드

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 합성 키포인트 데이터 생성 (실제 영상 없이 테스트)
np.random.seed(42)
N_SAMPLES = 300    # 총 샘플 수
N_CLASSES = 3      # 행동 클래스 수
N_KEYPOINTS = 99   # 33관절 × 3좌표

def generate_synthetic_data(window_size, n_samples=N_SAMPLES):
    """행동별 특징이 다른 합성 키포인트 시퀀스 생성"""
    X, y = [], []
    for cls in range(N_CLASSES):
        for _ in range(n_samples // N_CLASSES):
            # 클래스별로 다른 패턴 부여
            base = np.random.randn(window_size, N_KEYPOINTS) * 0.1
            if cls == 1:  # 허리 위험 각도: 허리 키포인트 값 변화
                base[:, 23*3:25*3] += np.linspace(0, 1, window_size)[:, None]
            elif cls == 2:  # 고소작업 위험: y좌표 전체 이동
                base[:, 1::3] -= 0.5
            X.append(base)
            y.append(cls)
    return np.array(X, dtype=np.float32), np.array(y)

def build_model(window_size, n_classes=N_CLASSES):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(window_size, N_KEYPOINTS),
                             return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

results = {}
for window_size in [15, 30, 60]:
    X, y = generate_synthetic_data(window_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model(window_size)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # TODO: 모델 학습
    # TODO: 정확도 및 지연 시간 측정
    # TODO: 결과 저장
    pass

# TODO: 결과 비교 시각화
```

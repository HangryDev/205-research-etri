# 섹션 2 | Action Recognition — 사람의 행동을 숫자로 읽는다

> **참고 교재**: *Practical Machine Learning for Computer Vision* (Valliappa Lakshmanan et al., O'Reilly)
> **소요 시간**: 45분 (문제 제기 3 + 이론 13 + 시연 10 + 실습 19)

---

## 2-1. 문제 제기 (3분)

**섹션 1에서 배운 것**: YOLO로 "무엇이 있는가(객체)"를 탐지
**이번 섹션의 질문**: "그 사람이 무엇을 하고 있는가(행동)"를 인식할 수 있을까?

```
[탐지와 인식의 차이]

Object Detection:   [사람] [안전모] [지게차]  ← 무엇이 있는가
Action Recognition: [사람이 난간 없이 높은 곳 작업 중]  ← 무엇을 하는가
```

**제조 현장의 위험 행동 예시**:

```
① 2인 1조 작업 규정 위반 → 혼자 중량물 이동
② 안전 난간 없는 고소 작업
③ 보호구 미착용 상태로 위험 구역 진입
④ 허리 굽힘 각도 위험 수준 (근골격계 질환)
```

→ **핵심 질문**: CCTV 영상에서 이런 위험 행동을 자동으로 감지할 수 있을까?

---

## 2-2. 이론 (13분)

### ① 비디오 이해의 접근법: 왜 골격 기반인가 (3분)

:::{admonition} 참고 교재
:class: note
📖 *Practical ML for Computer Vision* **Ch.11 Image and Video Understanding** — 비디오 분류 접근법 비교, 골격 기반 행동인식의 장점
:::

비디오에서 행동을 인식하는 방법은 크게 두 가지입니다.

```
[픽셀 기반 접근 (3D CNN / Two-Stream)]

장점: 배경 정보까지 활용, 높은 정확도
단점: 연산량 매우 큼 → 엣지 불가
     사생활 침해 우려 (얼굴, 개인 식별 가능)
     조명·카메라 각도 변화에 취약

[골격 기반 접근 (Skeleton / Keypoint)]

장점: 연산량 적음 → 엣지 가능
     사생활 보호 (픽셀 없이 좌표만 사용)
     조명·배경 변화에 강건
     해석 가능 (어떤 관절이 이상한지 설명 가능)
단점: 가려짐(occlusion)에 취약
     골격 추출 단계가 추가됨
```

:::{admonition} 팁
:class: tip
**제조 현장 선택 기준**: 24시간 운영, 엣지 배포, 개인정보 보호 요건
→ **골격 기반 방식이 현실적인 선택**
:::

---

### ② MediaPipe로 골격 추출 (5분)

:::{admonition} 참고 교재
:class: note
📖 *Practical ML for Computer Vision* **Ch.11 Image and Video Understanding** — 포즈 추정(Pose Estimation) 원리와 키포인트 정의 / **Ch.5 Creating Vision Datasets** — 비디오 데이터에서 키포인트 추출 및 라벨링
:::

MediaPipe Pose는 이미지/영상에서 **33개의 키포인트** 좌표를 실시간으로 추출합니다.

```
[MediaPipe 33개 키포인트]

         0 코
    1·2·3·4·5·6·7·8 얼굴
         11·12 어깨
    13·14 팔꿈치  15·16 손목
    17·18·19·20·21·22 손
         23·24 엉덩이
    25·26 무릎    27·28 발목
         29·30·31·32 발
```

**단일 프레임의 출력**:

```python
# 하나의 프레임 → [33 × 3] 행렬
# 3: x좌표, y좌표, visibility(가시성)

keypoints = np.array([
    [x0, y0, v0],   # 코
    [x1, y1, v1],   # 왼쪽 눈 안쪽
    ...
    [x32, y32, v32] # 오른쪽 발 인덱스
])  # shape: (33, 3)
```

**시퀀스로 확장** (N 프레임):

```python
import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose

def extract_keypoints(frame):
    """단일 프레임에서 키포인트 추출"""
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            kp = np.array([
                [lm.x, lm.y, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ])  # shape: (33, 3)
            return kp
        return np.zeros((33, 3))

# 30프레임 시퀀스 → shape: (30, 33, 3) → flatten → (30, 99)
```

---

### ③ 키포인트 시퀀스를 LSTM으로 분류 (5분)

:::{admonition} 참고 교재
:class: note
📖 *Practical ML for Computer Vision* **Ch.11 Image and Video Understanding** — 시계열 키포인트 분류, RNN 기반 행동인식 모델 구조
:::

세션 2에서 배운 LSTM 구조를 그대로 재활용합니다.

```{mermaid}
flowchart TD
    A["비디오 프레임 시퀀스"] --> B["MediaPipe"]
    B --> C["키포인트 시퀀스 [30 × 99]<br/>(30프레임 × 33관절 × 3좌표)"]
    C --> D["LSTM"]
    D --> E["행동 분류 결과<br/>정상 작업 / 위험 행동A / 위험 행동B"]
```

```
[행동 인식 파이프라인]

비디오 프레임 시퀀스
      ↓ MediaPipe
키포인트 시퀀스 [30 × 99]  (30프레임 × 33관절 × 3좌표)
      ↓ LSTM
행동 분류 결과 ["정상 작업" / "위험 행동A" / "위험 행동B"]
```

```python
import tensorflow as tf

# 세션 2 LSTM 구조와 동일한 패턴
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(30, 99),
                         return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 행동 클래스 수
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**입력 준비 — 슬라이딩 윈도우** (세션 2와 동일 개념):

```python
def create_sequences(keypoint_data, labels, window_size=30, step=15):
    """키포인트 시계열 → LSTM 입력 시퀀스"""
    X, y = [], []
    for i in range(0, len(keypoint_data) - window_size, step):
        X.append(keypoint_data[i:i+window_size].reshape(window_size, -1))
        y.append(labels[i+window_size])
    return np.array(X), np.array(y)
```

:::{admonition} 팁
:class: tip
**실무 판단 기준**:
위험 행동은 종류가 적고 패턴이 명확함 → 클래스 수 3~5개로 시작
데이터가 부족할 경우 → 정상 행동만 학습한 Autoencoder로 이상탐지 (세션 2 섹션 1 재활용)
:::

---

## 2-3. Claude Code 시연 (10분)

:::{admonition} 팁
:class: tip
**시연 포인트**: MediaPipe로 키포인트를 추출하고 LSTM 입력으로 변환하는 **연결 과정**에 집중하세요.
:::

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

## 2-4. 실습 (19분)

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

---

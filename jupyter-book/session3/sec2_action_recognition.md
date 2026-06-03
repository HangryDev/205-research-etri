# Action Recognition — 사람의 행동을 숫자로 읽는다

---

## 2-1. 문제 제기

**섹션 1**: YOLO로 **"무엇이 있는가(객체)"** 를 탐지
**섹션 2**: **"그 사람이 무엇을 하고 있는가(행동)"** 를 인식

```
[탐지와 인식의 차이]

Object Detection:   [사람] [안전모] [지게차]  ← 무엇이 있는가
Action Recognition: [사람이 난간 없이 높은 곳 작업 중]  ← 무엇을 하는가
```

**제조 현장의 위험 행동 예시**:

- **2인 1조 작업 규정 위반**: 혼자 중량물 이동 (객체 탐지만으로는 "사람 한 명 있음"으로만 보임)
- **안전 난간 없는 고소 작업**: 추락 위험 (자세 패턴으로 포착 가능)
- **위험 구역 무단 진입**: 보호구 미착용 상태로 진입 (객체 탐지 + 행동 인식 결합 필요)
- **근골격계 위험 자세**: 허리 과도 굽힘 반복 작업 → 관절 각도 측정 필요

```{admonition} 핵심 질문
:class: important

CCTV 영상에서 이런 위험 행동을 **실시간으로 자동 감지**할 수 있을까?
```

---

## 2-2. 이론

### ① 비디오 이해의 접근법: 왜 골격 기반인가

> *Practical ML for Computer Vision* Ch.11 — 비디오 분류 접근법 비교

**픽셀 기반 접근**(3D CNN, Two-Stream)은 영상 프레임을 그대로 입력해 시공간 패턴을 학습합니다. 배경 정보까지 활용해 높은 정확도를 낼 수 있다는 장점이 있지만, 연산량이 매우 커서 **엣지에서는 불가능** 하고, 얼굴 등 개인 식별이 가능해 사생활 침해 우려가 있으며, 조명과 카메라 각도 변화에도 취약합니다.

**골격 기반 접근**(Skeleton/Keypoint)은 영상에서 사람의 **관절 좌표만** 추출해 시퀀스로 분류합니다. 33개 좌표만 처리하므로 연산량이 적어 **엣지에서 동작** 하고, 픽셀 없이 좌표만 쓰므로 개인 식별이 불가능해 사생활을 보호하며, 조명·배경 변화에 강건하고 어떤 관절이 이상한지 설명할 수 있어 해석 가능성도 높습니다. 다만 기계 뒤에 가려지는 **가림(occlusion)** 에 취약하고, 손가락 움직임 같은 미세한 동작은 33개 키포인트로 표현하기 어렵습니다.

```{admonition} 팁
:class: tip

**제조 현장 선택 기준**: 24시간 운영 + 엣지 배포 + 개인정보 보호
→ **골격 기반 방식이 현실적인 선택** (*Practical ML for Computer Vision* Ch.11 결론과 동일)
```

```{figure} ../../lecture/images/s3_2_img13.png
:alt: 거리에 따른 이미지 왜곡 — 가까이 찍으면 크기가 다르게 보임
:width: 67%

거리에 따른 이미지 왜곡 — 가까이 찍으면 크기가 다르게 보임
```

- 같은 대상도 **카메라 거리·각도**에 따라 크기·형태가 다르게 찍힘을 보여줌
- 픽셀 기반 분석이 거리·각도 변화에 취약한 이유
- 좌표 정규화·스케일 보정이 필요한 배경

```{figure} ../../lecture/images/s3_2_img14.png
:alt: 참조물(신용카드)을 이용한 스케일 보정
:width: 67%

참조물(신용카드)을 이용한 스케일 보정
```

- 신용카드처럼 **크기를 아는 참조물**로 영상 속 실제 스케일을 보정
- 거리에 따른 크기 왜곡을 기준 물체로 교정하는 아이디어
- 키포인트 좌표를 일관된 척도로 만드는 전처리 개념

```{figure} ../../lecture/images/s3_2_img15.png
:alt: 데이터 증강 예시
:width: 67%

데이터 증강 예시
```

- 회전·이동·밝기 변형 등으로 학습 데이터를 늘리는 **데이터 증강** 예시
- 다양한 각도·조명을 미리 학습시켜 모델을 강건하게 만듦
- 측면 촬영 같은 각도 변화에 대응하는 근본책

---

### ② MediaPipe로 골격 추출

> *Practical ML for Computer Vision* Ch.11 — 포즈 추정 원리 / Ch.5 — 비디오 데이터 키포인트 추출

**MediaPipe Pose**: Google 오픈소스, 이미지/영상에서 **33개 키포인트** 좌표를 실시간 추출

MediaPipe는 코(0), 얼굴(1~8), 어깨(11·12), 팔꿈치(13·14)와 손목(15·16), 손(17~22), 엉덩이(23·24), 무릎(25·26)과 발목(27·28), 발(29~32) 등 몸 전체에 걸쳐 33개의 키포인트를 추출합니다.

각 키포인트는 세 개의 값(x, y, visibility)을 갖습니다. **x, y** 는 이미지 크기와 무관한 0~1 정규화 좌표이고, **visibility** 는 0~1 사이의 가시성으로 가려지면 낮아집니다. 따라서 한 프레임은 33 × 3 = **99차원 벡터** 가 됩니다.

```python
import mediapipe as mp, cv2, numpy as np

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
```

```{admonition} 주의
:class: warning

- `min_detection_confidence=0.5`: 50% 미만 확신은 무시. 너무 낮추면 잡음, 너무 높이면 진짜 사람 놓침
- `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` **필수**: OpenCV는 BGR, MediaPipe는 RGB. 빼먹으면 결과 미묘하게 이상 (디버깅 어려운 함정)
- 키포인트 미검출 시 **영행렬 반환** (시퀀스 입력 차원 유지)
```

**MediaPipe 핵심 장점**: CPU만으로도 **초당 30프레임** 가능. GPU 불필요 → Raspberry Pi급 엣지에서도 동작

```{figure} ../../lecture/images/s3_2_img19.png
:alt: 객체 감지 예시 — 식물 위의 열매
:width: 67%

객체 감지 예시 — 식물 위의 열매
```

- 일반적인 객체 탐지의 예시(식물 위 열매 검출)
- "무엇이 있는가"를 박스로 찾는 탐지 방식의 사례
- 다음의 포즈/관절 인식과 대비되는 출발점

```{figure} ../../lecture/images/s3_2_img25.png
:alt: 신체 주요 부위의 상대적 위치 파악
:width: 67%

신체 주요 부위의 상대적 위치 파악
```

- 사람 몸의 **주요 부위 상대 위치**를 파악하는 모습
- 절대 픽셀이 아니라 부위 간 상대 배치로 자세를 표현
- 골격 기반 표현이 무엇인지에 대한 직관

```{figure} ../../lecture/images/s3_2_img26.png
:alt: 인간 포즈 식별 — 주요 관절 위치
:width: 67%

인간 포즈 식별 — 주요 관절 위치
```

- 사람의 **주요 관절(키포인트)** 위치를 찍어 포즈를 식별
- MediaPipe가 추출하는 33개 키포인트의 개념을 보여줌
- 픽셀 없이 좌표만으로 자세를 다루는 핵심

---

### ③ 키포인트 시퀀스를 LSTM으로 분류

> *Practical ML for Computer Vision* Ch.11 — 시계열 키포인트 분류

행동은 한 프레임으로 판단할 수 없음. "허리를 굽히는 동작 중이다"는 **시간에 따른 변화**이므로 본질적으로 **시퀀스 분류 문제**.

→ **Session 2에서 배운 LSTM**이 여기서 재등장. 센서 시계열 → 키포인트 시계열로 입력만 교체.

```{mermaid}
flowchart TD
    A["비디오 프레임 시퀀스"] --> B["MediaPipe"]
    B --> C["키포인트 시퀀스 30 × 99<br/>(30프레임 × 33관절 × 3좌표)"]
    C --> D["LSTM"]
    D --> E["행동 분류 결과<br/>정상 작업 / 위험 행동A / 위험 행동B"]
```

**슬라이딩 윈도우**:

`window_size=30` 은 30FPS 영상에서 1초 분량으로 대부분의 작업 동작 단위를 포착하고, `step=15` 는 15프레임(절반)씩 겹쳐 이동하므로 0.5초마다 결과가 갱신되어 반응 속도가 두 배가 됩니다.

```python
def create_sequences(keypoint_data, labels, window_size=30, step=15):
    """키포인트 시계열 → LSTM 입력 시퀀스"""
    X, y = [], []
    for i in range(0, len(keypoint_data) - window_size, step):
        X.append(keypoint_data[i:i+window_size].reshape(window_size, -1))
        y.append(labels[i+window_size])
    return np.array(X), np.array(y)
```

**LSTM 행동 분류 모델**:

```python
import tensorflow as tf

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

| 레이어 | 역할 |
|--------|------|
| `LSTM(64, return_sequences=True)` | 첫 번째 LSTM. 64차원 은닉 상태. 시퀀스 전체를 다음 층에 전달 |
| `Dropout(0.3)` | 학습 중 30% 뉴런 무작위 off → 과적합 방지 |
| `LSTM(32)` | 두 번째 LSTM. 마지막 타임스텝 출력만 반환 |
| `Dense(3, softmax)` | 3개 클래스 확률 출력 |
| `sparse_categorical_crossentropy` | 라벨이 원-핫이 아닌 정수(0, 1, 2)일 때 사용 |

```{admonition} 핵심
:class: important

**Session 2의 LSTM과 본질적으로 같은 구조**.
입력만 센서 시계열 → 키포인트 시계열로 바뀐 것.
한 번 익힌 모듈은 도메인을 옮겨서 재활용 가능.
```

```{admonition} 팁
:class: tip

**실무 판단 기준**:
- 위험 행동은 종류가 적고 패턴이 명확 → 클래스 수 **3~5개**로 시작
- 데이터가 부족할 경우 → 정상 행동만 학습한 **Autoencoder**로 이상탐지 (세션 2 섹션 1 재활용)
```

```{figure} ../../lecture/images/s3_2_img27.png
:alt: PoseNet 출력으로 어노테이션된 이미지
:width: 67%

PoseNet 출력으로 어노테이션된 이미지
```

- PoseNet 출력으로 관절을 **이미지 위에 표시(어노테이션)** 한 결과
- 추출된 키포인트가 실제 몸에 얼마나 잘 들어맞는지 확인
- 포즈 추정 품질을 눈으로 검증하는 방법

```{figure} ../../lecture/images/s3_2_img28.png
:alt: 여러 사람의 포즈 동시 식별
:width: 67%

여러 사람의 포즈 동시 식별
```

- 한 장면에서 **여러 사람의 포즈를 동시에** 추출
- 다인 작업 규정(2인 1조 등) 감시에 적용 가능함을 시사
- 실제 CCTV의 다중 인원 상황 대응

---

# CCTV 안전 위반 탐지 통합 실습

---

## 3-1. 파이프라인 설계 + 데이터 준비

> *Practical ML for Computer Vision* Ch.6 — 영상 전처리 파이프라인 / Ch.4 — YOLO 기반 다단계 탐지 파이프라인

### 통합 파이프라인 구조

섹션 1(YOLO)과 섹션 2(Action Recognition)를 **하나의 파이프라인**으로 엮는 것이 섹션 3의 목표.

```{mermaid}
flowchart TD
    A["CCTV 영상 입력 30 FPS"] --> B["프레임 추출"]
    B --> C["단일 프레임"]
    C --> D{"병렬 처리"}
    D --> E["YOLO → 안전모 탐지<br/>bbox 신뢰도 > 0.5?<br/>→ 미착용 위반 판정"]
    D --> F["MediaPipe + LSTM → 행동 인식<br/>위험 행동 확률 > 0.7?<br/>→ 행동 위반 판정"]
    E --> G{"위반 판정 로직<br/>위반 조건 충족?"}
    F --> G
    G -->|Yes| H["경보 발생<br/>프레임 저장<br/>로그 기록"]
    G -->|No| I["정상 처리<br/>계속 모니터링"]
```

### 병렬 처리가 핵심인 이유

두 모델은 **서로 다른 호흡** 으로 동작합니다. **YOLO** 는 단일 프레임만으로 매 프레임 추론할 수 있어 "안전모가 있다/없다"를 1프레임으로 판단하지만, **LSTM** 은 30프레임 시퀀스가 모여야 추론하므로 첫 30프레임 동안은 결과가 없고 호흡이 느립니다. 둘을 직렬로 묶으면("YOLO가 끝나면 LSTM") LSTM이 대기하는 동안 YOLO 결과까지 멈춰 버립니다. 그래서 **각 스트림을 자기 속도로 돌리고, 결과를 OR 조건으로 합쳐 위반을 판정** 합니다.

```{figure} ../../lecture/images/s3_3_img01.png
:alt: Object Detection, Instance Segmentation, Semantic Segmentation 비교
:width: 67%

Object Detection, Instance Segmentation, Semantic Segmentation 비교
```
```{figure} ../../lecture/images/s3_3_img02.png
:alt: Object Detection 태스크 예시
:width: 67%

Object Detection 태스크 예시
```
```{figure} ../../lecture/images/s3_3_img03.png
:alt: Arthropods 데이터셋 예시
:width: 67%

Arthropods 데이터셋 예시
```

### 데이터 준비 시 주의사항

> *Practical ML for Computer Vision* Ch.5 — 데이터셋 설계 주의점

데이터 준비에서는 세 가지를 특히 주의해야 합니다. 첫째, **카메라 위치와 각도** 입니다. 안전모 탐지에는 머리가 잘 보이는 각도가, 행동 인식에는 전신이 잘 보이는 각도가 필요하므로 약간 높은 위치에 설치하면 한 대로 둘 다 커버할 수 있습니다. 둘째, **클래스 라벨링 규칙의 명확성** 입니다. "안전모 미착용"의 경계 케이스(턱끈을 안 매면? 머리에 얹기만 하면?)를 미리 정의하고, 라벨링 가이드라인을 데이터셋과 함께 관리해야 합니다. 셋째, **데이터 불균형** 입니다. 위반 상황은 정상보다 훨씬 드물어 그대로 학습하면 "정상"만 외워도 90%대 정확도가 나오므로, 위반 클래스를 오버샘플링하거나 클래스 가중치를 조정해야 합니다.

```{admonition} 주의
:class: warning

실무에서는 **데이터 준비 단계가 전체 프로젝트 공수의 절반 이상**을 차지.
오늘은 합성 데이터와 사전 학습 모델을 사용하지만, 실제 배포 시 데이터 준비에 충분한 시간을 배정해야 함.
```

---

## 3-2. 통합 탐지 + 알림 트리거

> *Practical ML for Computer Vision* Ch.4 — 다단계 탐지 파이프라인 조건 결합 전략

### 위반 판정 함수

하나의 함수에 YOLO와 MediaPipe+LSTM을 모두 묶고, 위반 여부와 시각화된 프레임을 반환.

```python
def detect_violation(frame, yolo_model, action_model, keypoint_buffer,
                     helmet_threshold=0.5, action_threshold=0.7):
    """
    단일 프레임에서 안전 위반 판정
    Returns: (is_violation, violation_type, annotated_frame)
    """
    violations = []
    annotated = frame.copy()
```

**매개변수**:

- `keypoint_buffer`: `collections.deque(maxlen=30)` — 항상 최근 30프레임만 유지 (LSTM 시퀀스 메모리)
- `helmet_threshold=0.5`: YOLO 바운딩 박스 최소 신뢰도
- `action_threshold=0.7`: LSTM 행동 클래스 최소 확률 (보수적으로 높게 설정)

### YOLO 스트림 — 안전모 탐지

```python
    # 1. YOLO: 안전모 탐지
    yolo_results = yolo_model(frame, verbose=False)
    for box in yolo_results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls == 0 and conf > helmet_threshold:  # 0: 안전모 미착용
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f'NO HELMET {conf:.2f}',
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 255), 2)
            violations.append('helmet_violation')
```

### MediaPipe + LSTM 스트림 — 행동 인식

```python
    # 2. MediaPipe + LSTM: 행동 인식
    keypoints = extract_keypoints(frame)
    keypoint_buffer.append(keypoints.flatten())

    if len(keypoint_buffer) >= 30:
        sequence = np.array(list(keypoint_buffer)[-30:])
        pred = action_model.predict(
            sequence[np.newaxis, ...], verbose=0
        )[0]
        action_cls = np.argmax(pred)
        action_conf = pred[action_cls]

        action_names = ['정상작업', '허리위험', '고소위험']
        if action_cls != 0 and action_conf > action_threshold:
            label = f'{action_names[action_cls]} {action_conf:.2f}'
            cv2.putText(annotated, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            violations.append(f'action_{action_names[action_cls]}')

    is_violation = len(violations) > 0
    return is_violation, violations, annotated
```

- 키포인트를 flatten하여 **99차원 벡터**로 버퍼에 추가
- 버퍼가 30개에 도달했을 때만 LSTM 추론 호출 (그 전에는 패스)
- `action_cls != 0`: 정상작업(인덱스 0)이 아닌 위반 행동만 처리

```{figure} ../../lecture/images/s3_3_img04.png
:alt: YOLO 그리드 — 각 셀이 바운딩 박스 예측
:width: 67%

YOLO 그리드 — 각 셀이 바운딩 박스 예측
```
```{figure} ../../lecture/images/s3_3_img05.png
:alt: YOLO Detection Head — bbox, confidence, class 동시 예측
:width: 67%

YOLO Detection Head — bbox, confidence, class 동시 예측
```
```{figure} ../../lecture/images/s3_3_img07.png
:alt: IoU(Intersection over Union) 지표
:width: 67%

IoU(Intersection over Union) 지표
```

### 위반 이벤트 로깅

탐지 결과를 CSV로 기록하여 사후 분석을 가능하게 함.

```python
import csv
from datetime import datetime

def log_violation(frame_idx, fps, violation_types, log_path='violations.csv'):
    """위반 이벤트를 CSV로 기록"""
    timestamp = frame_idx / fps  # 초 단위
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 벽시계 시간
            f'{timestamp:.1f}s',                              # 영상 내 상대 시간
            ', '.join(violation_types)
        ])
```

- **두 가지 타임스탬프** 기록: 벽시계 시간(언제 일어났는지) + 영상 내 상대 시간(영상의 몇 초 지점인지)
- 둘 다 있으면 영상에서 해당 장면을 다시 찾기 쉬움

### 오탐 감소: 연속 프레임 규칙 (Temporal Smoothing)

**실무 최대 이슈**: 단일 프레임에서 YOLO가 살짝 흔들리거나 MediaPipe가 미세 오차 → 거짓 알림 폭발 → 현장에서 시스템 꺼버림 (**알람 피로**, alarm fatigue)

**해결책**: **N개 연속 프레임 규칙** — 1~2프레임 위반은 무시, N프레임 연속 위반 시에만 알림

```python
class ViolationDetector:
    def __init__(self, min_consecutive=15):
        self.min_consecutive = min_consecutive
        self.consecutive_count = {}

    def update(self, violation_types):
        active = set(violation_types)
        triggered = []
        for v in active:
            self.consecutive_count[v] = self.consecutive_count.get(v, 0) + 1
            if self.consecutive_count[v] == self.min_consecutive:
                triggered.append(v)
        # 더 이상 보이지 않는 위반은 카운트 리셋
        for v in list(self.consecutive_count.keys()):
            if v not in active:
                self.consecutive_count[v] = 0
        return triggered
```

30FPS에서 `min_consecutive=15` 로 두면 **0.5초 이상 지속** 되는 위반만 알림이 울리고, `detect_violation` 결과와 함께 사용하면 일시적인 오탐이 제거됩니다.

```{admonition} 핵심
:class: important

CCTV 안전 모니터링 파이프라인의 본질:
1. **두 개의 독립적인 스트림** (YOLO + MediaPipe+LSTM)
2. **각각의 임계값** (helmet_threshold, action_threshold)
3. **OR 결합** (둘 중 하나라도 위반이면 알림)
4. **시간적 평활화** (연속 프레임 규칙으로 오탐 감소)
```

```{figure} ../../lecture/images/s3_3_img08.png
:alt: CNN 단계별 특징 맵 — 공간 해상도 감소, 의미 정보 증가
:width: 67%

CNN 단계별 특징 맵 — 공간 해상도 감소, 의미 정보 증가
```
```{figure} ../../lecture/images/s3_3_img09.png
:alt: YOLO, SSD, FPN 아키텍처 비교
:width: 67%

YOLO, SSD, FPN 아키텍처 비교
```
```{figure} ../../lecture/images/s3_3_img10.png
:alt: Feature Pyramid Network 상세 구조
:width: 67%

Feature Pyramid Network 상세 구조
```

---

## 3-3. 자유 실습

난이도를 세 단계로 나눴습니다. 한두 가지를 깊게 파는 걸 추천합니다.

### 기본 — 파이프라인 실행 및 결과 확인

1. 샘플 영상(또는 합성 영상)으로 파이프라인 실행
2. `cv2.VideoWriter`로 위반 탐지 결과가 오버레이된 영상을 mp4 저장
3. 영상 종료 시 누적 위반 이벤트 수 출력

**제출**: 결과 영상 파일 + 위반 이벤트 수

### 중급 — 신뢰도 임계값 조정 실험

1. `helmet_threshold`를 **0.3, 0.5, 0.7**로 변경하며 탐지 이벤트 수 기록
2. 각 임계값에서 잘못 잡힌(또는 놓친) 대표 프레임 추출하여 비교

**제출**: 임계값별 탐지 이벤트 수 표 + "제조 현장에서 어떤 임계값을 선택할 것인가?" 한 문단

```{admonition} 팁
:class: tip

**안전 모니터링에서는 Recall이 Precision보다 우선**.
"한 번이라도 놓치면 안 된다"가 "가짜로 잡힌 게 좀 있다"보다 비쌈.
→ 임계값을 낮춰 Recall을 높이고, 오탐은 연속 프레임 규칙으로 잡는 것이 일반적.
단, 너무 낮추면 알람 피로가 폭발하므로 균형점이 필요.
```

### 심화 — 위반 이벤트 로그 분석 대시보드

1. 위반 이벤트 로그(`violations.csv`) 생성
   - 컬럼: 타임스탬프 | 경과시간(초) | 위반 유형
2. 분석 시각화 3개:
   - **시간대별 위반 빈도 히스토그램** (새벽/오전/점심/오후/저녁)
   - **위반 유형별 비율 파이차트** (안전모 미착용 / 허리 위험 / 고소 위험)
   - **누적 위반 시계열** (x축: 시간, y축: 누적 위반 수. 기울기가 가파른 구간 = 위반 다발 시간대)
3. `matplotlib` 또는 `plotly`로 충분. 더 도전한다면 `streamlit`으로 인터랙티브 대시보드

**제출**: 시각화 3개 + 분석 결론 한 문단 ("이 시간대, 이 유형이 가장 많고, 그 원인은 ~로 추정, 따라서 ~을 개선해야 한다")

```{admonition} Claude에게 추가 질문
:class: tip

> "이 위반 패턴 데이터를 보고 어떤 시간대에 안전 점검을 강화해야 하는지 결정하는 로직을 만들어줘"

데이터에서 시작해 의사 결정까지 가는 게 데이터 분석의 본질.
```

```{figure} ../../lecture/images/s3_3_img11.png
:alt: 다양한 크기와 비율의 앵커 박스 예시
:width: 67%

다양한 크기와 비율의 앵커 박스 예시
```
```{figure} ../../lecture/images/s3_3_img15.png
:alt: RetinaNet 아키텍처 전체 구조
:width: 67%

RetinaNet 아키텍처 전체 구조
```
```{figure} ../../lecture/images/s3_3_img18.png
:alt: NMS 적용 전후 — 중복 탐지 제거
:width: 67%

NMS 적용 전후 — 중복 탐지 제거
```

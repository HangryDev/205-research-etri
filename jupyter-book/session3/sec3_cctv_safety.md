# 섹션 3 | CCTV 안전 위반 탐지 통합 실습

```{admonition} 참고 교재
:class: note

**참고 교재**: *Practical Machine Learning for Computer Vision* (Valliappa Lakshmanan et al., O'Reilly)
```

---

## 3-1. 파이프라인 설계 + 데이터 준비

### 통합 파이프라인 구조

```{admonition} 참고 교재
:class: note

📖 *Practical ML for Computer Vision* **Ch.6 Image Classification and Preprocessing** — 영상 프레임 추출과 전처리 파이프라인 / **Ch.8 Faster and More Efficient Models** — 다단계 모델 파이프라인 설계
```

섹션 1(YOLO)과 섹션 2(Action Recognition)를 하나로 연결합니다.

```{mermaid}
flowchart TD
    A["CCTV 영상 입력 (30 FPS)"] --> B["프레임 추출"]
    B --> C["단일 프레임"]
    C --> D{"병렬 처리"}
    D --> E["YOLO → 안전모 탐지<br/>bbox 신뢰도 > 0.5?<br/>→ 미착용 위반 판정"]
    D --> F["MediaPipe + LSTM → 행동 인식<br/>위험 행동 확률 > 0.7?<br/>→ 행동 위반 판정"]
    E --> G{"위반 판정 로직<br/>위반 조건 충족?"}
    F --> G
    G -->|Yes| H["경보 발생<br/>프레임 저장<br/>로그 기록"]
    G -->|No| I["정상 처리<br/>계속 모니터링"]
```

```
[CCTV 안전 위반 탐지 파이프라인]

CCTV 영상 입력 (30 FPS)
      ↓ 프레임 추출
단일 프레임
      ↓
┌─────────────────────────────────────────┐
│  병렬 처리                               │
│                                          │
│  [YOLO] → 안전모 탐지                    │
│    bbox 신뢰도 > 0.5? → 미착용 위반 판정  │
│                                          │
│  [MediaPipe + LSTM] → 행동 인식          │
│    위험 행동 확률 > 0.7? → 행동 위반 판정  │
└─────────────────────────────────────────┘
      ↓ 위반 판정 로직
  위반 조건 충족?
      ↓ Yes                ↓ No
  경보 발생             정상 처리
  프레임 저장           계속 모니터링
  로그 기록
```

**Claude로 프레임 추출 코드 요청**:
```
CCTV mp4 영상에서 분석용 프레임을 추출하는 코드를 만들어줘.
- OpenCV로 영상 열기
- 매 N번째 프레임만 추출 (FPS 조절 가능하게)
- 각 프레임을 640×640으로 리사이즈
- 추출된 프레임 수와 원본 영상 정보 출력
- 샘플 영상 없을 때를 위해 numpy로 합성 프레임도 생성 가능하게
```

---

## 3-2. 통합 탐지 + 알림 트리거

### 위반 판정 로직

```{admonition} 참고 교재
:class: note

📖 *Practical ML for Computer Vision* **Ch.8 Faster and More Efficient Models** — 다단계 탐지 파이프라인의 조건 결합 전략
```

```python
def detect_violation(frame, yolo_model, action_model, keypoint_buffer,
                     helmet_threshold=0.5, action_threshold=0.7):
    """
    단일 프레임에서 안전 위반 판정
    Returns: (is_violation, violation_type, annotated_frame)
    """
    violations = []
    annotated = frame.copy()

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

**위반 이벤트 로깅**:

```python
import csv
from datetime import datetime

def log_violation(frame_idx, fps, violation_types, log_path='violations.csv'):
    """위반 이벤트를 CSV로 기록"""
    timestamp = frame_idx / fps  # 초 단위
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f'{timestamp:.1f}s',
            ', '.join(violation_types)
        ])
```

**Claude에게 추가 질문**:
> *"오탐(정상인데 위반으로 판정)을 줄이려면 위반 판정 조건에 무엇을 추가해야 해?
> 예를 들어 '연속 N프레임에서 위반이어야 알림' 같은 조건은 어떻게 구현해?"*

---

## 3-3. 자유 실습

본인의 수준에 맞는 과제를 선택하세요.

### ★ 기본 — 파이프라인 실행 및 결과 확인
```
목표: 통합 파이프라인을 처음부터 끝까지 돌린다

1. 제공된 샘플 영상(또는 합성 영상)으로 파이프라인 실행
2. 위반 탐지 결과가 오버레이된 영상 저장
3. 탐지된 위반 이벤트 수 출력

제출: 결과 영상 파일 + 위반 이벤트 수
```

### ★★ 중급 — 신뢰도 임계값 조정 실험
```
목표: 임계값 변화가 오탐/미탐에 미치는 영향을 수치로 확인

helmet_threshold를 [0.3, 0.5, 0.7]로 변경하며:
  → 각 설정에서 탐지된 위반 이벤트 수 기록
  → 임계값이 낮을 때 vs 높을 때 대표 프레임 비교

제출: 임계값별 탐지 이벤트 수 표 +
      "제조 현장에서 어떤 임계값을 선택할 것인가?" 한 문단
```

### ★★★ 심화 — 위반 이벤트 로그 분석 대시보드
```
목표: 탐지 결과를 시계열로 분석하여 위험 패턴을 시각화

1. 위반 이벤트 로그(violations.csv) 생성
   컬럼: 타임스탬프 | 경과시간(초) | 위반 유형

2. 분석 시각화 3개:
   ① 시간대별 위반 빈도 히스토그램
   ② 위반 유형별 비율 파이차트
   ③ 위반 발생 시계열 (x축: 시간, y축: 누적 위반 수)

3. Claude에게 질문:
   "이 위반 패턴 데이터를 보고 어떤 시간대에 안전 점검을 강화해야 하는지
    결정하는 로직을 만들어줘"

제출: 시각화 3개 + 분석 결론 한 문단
```

---

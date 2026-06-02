# 심화 학습 자료

```{admonition} 심화 학습 자료
:class: note

강의에서 시간 관계상 다루지 못한 심화 내용입니다. 강의 후 궁금증이 생겼을 때 아래 순서로 탐색하시길 권장합니다.
```

---

## 📘 섹션 1 심화: 모델 경량화와 엣지 배포

```{admonition} 참고 교재
:class: note

📖 **교재**: *Practical Deep Learning for Cloud, Mobile, and Edge* (Anirudh Koul et al., O'Reilly)
**심화 챕터 로드맵**:

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.3** Preparing Your Data | CNN 구조 이해, 데이터 준비 | 모델 선택의 배경 이해 |
| 2 | **Ch.6** Maximizing Speed of an Existing Mobile Model | MobileNet, 속도 최적화 | 경량화 원리 심화 |
| 3 | **Ch.13** Exploring AI at the Edge | 엣지 디바이스 종류와 제약 | 배포 환경 이해 |
| 4 | **Ch.14** Building an Augumented Reality App | 실제 엣지 배포 실습 | 프로덕션 수준 구현 |
```

### Quantization (양자화): float32 → int8

```python
import tensorflow as tf

# 학습된 모델을 TFLite로 변환하며 양자화 적용
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 동적 범위 양자화

# 더 강한 양자화: 정수 전용 (엣지 TPU용)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# 크기 비교
print(f"원본 모델: {len(tf.keras.utils.get_file_size(model)):,} bytes")
print(f"양자화 모델: {len(tflite_model):,} bytes")  # 약 4배 감소
```

**양자화 방법 비교**:

| 방법 | 크기 감소 | 속도 향상 | 정확도 손실 | 비고 |
|------|---------|---------|----------|------|
| 동적 범위 양자화 | 4× | 2~3× | 미미 | 가장 쉬움 |
| Float16 양자화 | 2× | GPU에서 향상 | 없음 | GPU 엣지용 |
| 전체 정수 양자화 | 4× | 3~4× | 약간 | 엣지 TPU 필수 |
| QAT (학습 중 양자화) | 4× | 3~4× | 최소 | 정확도 보존 최선 |

### Pruning (가지치기)

```python
import tensorflow_model_optimization as tfmot

# 학습 중 가지치기 적용
pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,   # 50%의 가중치를 0으로
    begin_step=0,
    end_step=1000
)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, pruning_schedule=pruning_schedule
)
```

### YOLOv10 vs YOLOv8 비교

| 항목 | YOLOv8 | YOLOv10 |
|------|--------|---------|
| NMS 필요 여부 | 필요 | 불필요 (NMS-free) |
| 추론 속도 | 빠름 | 더 빠름 |
| 정확도 | 높음 | 동등 이상 |
| 엣지 적합성 | 좋음 | 더 좋음 |
| 라이브러리 지원 | 풍부 | 아직 성장 중 |

---

## 📘 섹션 2 심화: 행동인식 심화와 데이터셋 구축

```{admonition} 참고 교재
:class: note

📖 **교재**: *Practical Machine Learning for Computer Vision* (Valliappa Lakshmanan et al., O'Reilly)
**심화 챕터 로드맵**:

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.5** Creating Vision Datasets | 비디오 라벨링, 데이터셋 설계 | 실제 데이터 수집 능력 |
| 2 | **Ch.11** Image and Video Understanding | 골격 행동인식 심화, 3D CNN | 고급 모델 선택 기준 |
```

### 비디오 라벨링 워크플로우

```{mermaid}
flowchart TD
    A["1. 영상 수집<br/>CCTV 영상 → 위험 행동 발생 구간 식별<br/>→ 해당 구간 클립 추출 (ffmpeg)"] --> B["2. 라벨링 도구<br/>CVAT (무료, 오픈소스): https://cvat.org<br/>Label Studio: https://labelstud.io<br/>→ 행동 클래스별 시작/종료 프레임 지정"]
    B --> C["3. 데이터 증강 (Data Augmentation)<br/>키포인트 뒤집기 (좌우 반전)<br/>속도 변환 (0.8× ~ 1.2× 재샘플링)<br/>노이즈 추가 (관절 좌표에 가우시안 노이즈)"]
```

```
[현장 데이터 수집 → 라벨링 → 학습 파이프라인]

1. 영상 수집
   CCTV 영상 → 위험 행동 발생 구간 식별
   → 해당 구간 클립 추출 (ffmpeg)

2. 라벨링 도구
   CVAT (무료, 오픈소스): https://cvat.org
   Label Studio: https://labelstud.io
   → 행동 클래스별 시작/종료 프레임 지정

3. 데이터 증강 (Data Augmentation)
   키포인트 뒤집기 (좌우 반전)
   속도 변환 (0.8× ~ 1.2× 재샘플링)
   노이즈 추가 (관절 좌표에 가우시안 노이즈)
```

### ST-GCN: 그래프 기반 행동인식

```python
# 골격을 그래프로 표현 → GCN으로 공간적 관계 학습
# 관절 간 연결(엣지)이 명시적으로 모델에 반영됨

# 예: 오른쪽 어깨(12) - 오른쪽 팔꿈치(14) - 오른쪽 손목(16)
# 이 연결 관계가 "팔을 드는 동작"을 학습하는 데 직접 활용됨

# 구현: torch-geometric 또는 mmaction2 라이브러리
# pip install mmaction2
```

### 카메라 각도 대응: 정규화 전략

```python
def normalize_keypoints(keypoints, reference_joints=(11, 12)):
    """
    어깨 중심점을 기준으로 키포인트 정규화
    → 카메라 위치나 피사체 위치 변화에 강건
    """
    left_shoulder = keypoints[reference_joints[0], :2]
    right_shoulder = keypoints[reference_joints[1], :2]
    center = (left_shoulder + right_shoulder) / 2
    scale = np.linalg.norm(left_shoulder - right_shoulder)

    normalized = keypoints.copy()
    normalized[:, :2] = (keypoints[:, :2] - center) / (scale + 1e-8)
    return normalized
```

---

## 📘 섹션 3 심화: 성능 평가와 실시간 최적화

```{admonition} 참고 교재
:class: note

📖 **교재**: *Practical Machine Learning for Computer Vision* (Valliappa Lakshmanan et al., O'Reilly)
**심화 챕터 로드맵**:

| 순서 | 챕터 | 핵심 내용 | 이유 |
|------|------|----------|------|
| 1 | **Ch.6** Image Classification and Preprocessing | 데이터 전처리 파이프라인 | 실제 영상 처리 능력 |
| 2 | **Ch.8** Faster and More Efficient Models | 다단계 파이프라인 최적화 | 실시간 성능 향상 |
```

### 탐지 성능 평가 지표

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# mAP (mean Average Precision): Object Detection 표준 지표
# IoU(Intersection over Union) 기반 bbox 정확도 평가

def calculate_iou(box1, box2):
    """두 bbox의 겹침 비율 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)

# 안전 모니터링 맥락의 평가 우선순위
# Recall 우선: 위반을 놓치는 것이 더 위험
# Precision 참고: 너무 많은 오탐은 현장 피로도 증가
```

### 오탐 감소 전략: 연속 프레임 조건

```python
from collections import deque

class ViolationDetector:
    """연속 N프레임에서 위반이 지속될 때만 알림 발생"""
    def __init__(self, consecutive_frames=5):
        self.history = deque(maxlen=consecutive_frames)
        self.threshold = consecutive_frames

    def update(self, is_violation):
        self.history.append(is_violation)
        # 연속 N프레임 모두 위반이어야 알림
        if len(self.history) == self.threshold:
            return all(self.history)
        return False

detector = ViolationDetector(consecutive_frames=5)
# 일시적 오탐(1~2프레임)은 알림 발생 안 함
# 지속적 위반(5프레임 연속)만 알림 발생
```

### 실시간 처리 최적화

```python
import threading
from queue import Queue

# 멀티스레딩으로 프레임 캡처와 추론을 병렬 처리
frame_queue = Queue(maxsize=5)
result_queue = Queue(maxsize=5)

def capture_thread(cap, frame_queue):
    """프레임 캡처 스레드"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

def inference_thread(frame_queue, result_queue, yolo_model, action_model):
    """추론 스레드"""
    while True:
        frame = frame_queue.get()
        result = detect_violation(frame, yolo_model, action_model)
        result_queue.put(result)
```

---

## 📚 추천 학습 경로

### Object Detection 심화 (1~2주) — *Practical Deep Learning for Cloud, Mobile, and Edge*

| 순서 | 챕터 | 핵심 내용 | 읽어야 하는 이유 |
|------|------|----------|--------------|
| 1 | **Ch.3** | CNN 구조와 데이터 준비 | 탐지 모델의 이론적 토대 |
| 2 | **Ch.6** | MobileNet, 경량화 기법 | 엣지 배포를 위한 핵심 |
| 3 | **Ch.13** | 엣지 디바이스 종류와 제약 | 배포 환경 선택 기준 |
| 4 | **Ch.14** | 실제 엣지 배포 실습 | 프로덕션 수준 구현 |

### Action Recognition 심화 (1주) — *Practical ML for Computer Vision*

| 순서 | 챕터 | 핵심 내용 | 읽어야 하는 이유 |
|------|------|----------|--------------|
| 1 | **Ch.5** | 비디오 데이터셋 라벨링 | 직접 데이터 수집 시 필수 |
| 2 | **Ch.11** | 골격 행동인식, 3D CNN | 고급 모델로 확장할 때 |

### 통합 파이프라인 심화 (1주) — *Practical ML for Computer Vision*

| 순서 | 챕터 | 핵심 내용 | 읽어야 하는 이유 |
|------|------|----------|--------------|
| 1 | **Ch.6** | 영상 전처리 파이프라인 | 실제 CCTV 영상 처리 |
| 2 | **Ch.8** | 다단계 파이프라인 최적화 | 실시간 성능 향상 |

### 실무 적용 프로젝트 아이디어

- [ ] **Open Images Dataset**: YOLO 안전모 탐지 모델 파인튜닝 ([Roboflow Universe](https://universe.roboflow.com)에서 안전모 데이터셋 검색)
- [ ] **NTU RGB+D Dataset**: ST-GCN으로 고급 행동인식 모델 구현
- [ ] **라즈베리파이 배포**: YOLOv8n + TFLite 양자화 모델을 실제 엣지 디바이스에 올리기
- [ ] **Streamlit 대시보드**: 위반 탐지 결과를 실시간 웹 대시보드로 시각화

---

*이 문서는 세션 3 강의 자료로 제작되었습니다.*
*질문 및 피드백: Claude Code를 활용해 실습 중 막히는 부분을 언제든지 질문하세요.*
*이전 세션 자료: `session1_manufacturing_data_analysis.md` / `session2_deep_learning_anomaly_detection.md`*

# 출처
## 섹션 1. 텍스트 북 : [Practical Deep Learning Cloud/Mobile/Edge](https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/)
## 섹션 2. 텍스트 북 : [Practical ML for Computer Vision](https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/)
## 섹션 3. 텍스트 북 : [Practical ML for Computer Vision](https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/)

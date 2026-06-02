# 섹션 2 | 데이터 불균형 — 99% 정확도의 함정

```{admonition} 참고 교재
:class: note

**참고 교재**: *Machine Learning for Imbalanced Data* (Kumar Abhishek & Mounir Abdelaziz)  
**소요 시간**: 45분 (문제 제기 3 + 이론 13 + 시연 10 + 실습 19)
```

---

## 2-1. 문제 제기 (3분)

**상황**: 공장 불량 탐지 모델을 만들었더니 정확도가 **99%** 가 나왔습니다.  
이 모델은 잘 만든 걸까요?

```
[데이터 구성]
정상 제품:  9,900개  (99%)
불량 제품:    100개  (1%)

[더미 모델의 전략]
"모든 샘플을 정상으로 예측한다"

정확도 = (맞게 예측한 수) / (전체) = 9,900 / 10,000 = 99% ✓
```

**하지만 실제로는**:
- 불량 100개를 **전부 놓쳤습니다** (불량을 잡으러 만든 모델이 불량을 하나도 못 잡음)
- 이것이 **정확도의 역설(Accuracy Paradox)** 입니다

```{admonition} 핵심
:class: important

**제조업에서의 현실**: 불량 탐지 실패는 리콜, 안전사고, 고객 클레임으로 이어집니다.  
정확도가 아닌 **불량을 얼마나 잡는가(Recall)** 가 핵심 지표입니다.
```

---

## 2-2. 이론 (13분)

### ① 정확도의 역설 → 올바른 평가 지표 (5분)

```{admonition} 참고 교재
:class: note

📖 *ML for Imbalanced Data* **Ch.1 Overcoming the Challenge of Imbalanced Data** — 정확도의 역설과 올바른 평가 지표 선택 / **Ch.3 Metrics** — Precision, Recall, F1, ROC-AUC 상세 설명
```

**Confusion Matrix** 로 실제 상황을 분석합니다.

```
                 예측: 정상    예측: 불량
실제: 정상         9,900          0       ← True Negative(TN), False Positive(FP)
실제: 불량           100          0       ← False Negative(FN), True Positive(TP)
```

```
Precision (정밀도) = TP / (TP + FP)
  → "불량이라고 예측한 것 중 실제 불량의 비율"
  → 더미 모델: 0 / 0 = 정의 불가

Recall (재현율) = TP / (TP + FN)
  → "실제 불량 중 모델이 잡아낸 비율"
  → 더미 모델: 0 / (0 + 100) = 0%  ← 핵심 지표

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
  → Precision과 Recall의 조화평균
  → 더미 모델: 0  ← 명확하게 실패를 드러냄
```

```{admonition} 핵심
:class: important

**제조업 판단 기준**:
- 불량 탐지: **Recall을 최우선** (놓치는 불량이 없어야 함)
- 정밀 검사 자원이 제한적: Precision도 함께 고려
- 보통 F1-Score를 종합 지표로 사용
```

---

### ② SMOTE: 소수 클래스 합성 원리 (4분)

```{admonition} 참고 교재
:class: note

📖 *ML for Imbalanced Data* **Ch.2 Oversampling Methods** — SMOTE 원리와 k-NN 기반 합성 샘플 생성 / **Ch.3 Metrics** — 오버샘플링 후 평가 지표 해석
```

**SMOTE(Synthetic Minority Over-sampling Technique)** 는 불량 샘플을 **인공적으로 새로 만들어** 균형을 맞춥니다.

```{mermaid}
flowchart LR
    A["소수 클래스 샘플 x_i 선택"] --> B["k-최근접 이웃 x_nn 탐색"]
    B --> C["그 사이 임의 지점에 새 샘플 생성\nx_new = x_i + λ × (x_nn - x_i)"]
    C --> D["그럴듯한 새 불량 샘플 생성"]
```

```
2D 특징 공간에서의 예시:

  불량 샘플 ●─────────────● 불량 샘플
                  ↑
           SMOTE가 생성한
           합성 샘플 ☆
```

**SMOTE의 장점**: 단순 복사(Oversampling)보다 모델이 더 다양한 경계를 학습  
**SMOTE의 주의점**: 노이즈 샘플 근처에서 생성된 합성 샘플은 오히려 혼란을 줄 수 있음

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"원본: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"SMOTE 후: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
```

---

### ③ class_weight: 손실함수 레벨에서 조정 (2분)

```{admonition} 참고 교재
:class: note

📖 *ML for Imbalanced Data* **Ch.5 Cost-Sensitive Learning** — 클래스 가중치를 이용한 비용 민감 학습 원리
```

SMOTE가 **데이터를 늘리는** 접근이라면, `class_weight`는 **학습 과정에서 불량 샘플의 영향력을 키우는** 접근입니다.

```python
from sklearn.ensemble import RandomForestClassifier

# class_weight='balanced': sklearn이 자동으로 비율에 맞게 가중치 설정
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# 또는 수동 설정
model = RandomForestClassifier(
    class_weight={0: 1, 1: 99},  # 불량(1)을 99배 중요하게
    random_state=42
)
```

```{admonition} 팁
:class: tip

**직관**: 불량 샘플을 하나 틀리면 정상 샘플 99개를 틀린 것과 같은 페널티를 줌  
→ 모델이 불량 샘플을 더 신중하게 학습
```

---

### ④ 임계값 조정: 0.5가 항상 최선이 아니다 (2분)

```{admonition} 참고 교재
:class: note

📖 *ML for Imbalanced Data* **Ch.10 Threshold Adjustment and Model Calibration** — 최적 임계값 탐색과 Precision-Recall 트레이드오프
```

대부분의 분류 모델은 **확률값**을 출력하고, 0.5를 기준으로 클래스를 결정합니다.

```
모델 출력: P(불량) = 0.3  →  기본값(0.5 기준): "정상"으로 분류

임계값을 0.2로 낮추면:
모델 출력: P(불량) = 0.3  →  "불량"으로 분류  ← 더 민감하게

트레이드오프:
- 임계값 ↓: Recall ↑, Precision ↓ (더 많이 잡지만 오탐도 증가)
- 임계값 ↑: Recall ↓, Precision ↑ (덜 잡지만 잡으면 맞음)
```

```python
# Precision-Recall Curve로 최적 임계값 탐색
from sklearn.metrics import precision_recall_curve

probas = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probas)

# F1이 최대인 임계값 찾기
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"최적 임계값: {best_threshold:.3f}")
```

---

## 2-3. Claude Code 시연 (10분)

**Claude에게 던질 프롬프트 예시**:
```
불량률 1%인 제조 데이터를 시뮬레이션해줘.
1. 기본 RandomForest 모델 학습
2. SMOTE 적용 후 재학습
3. class_weight='balanced' 적용 후 재학습
세 모델의 Confusion Matrix와 F1-score를 나란히 비교하는 시각화를 만들어줘.
```

**시연 후 Claude에게 추가 질문**:
> *"SMOTE가 항상 최선의 방법인가? 어떤 상황에서는 안 쓰는 게 나을까?"*

---

## 2-4. 실습 (19분)

### 과제

SMOTE의 `k_neighbors` 파라미터가 결과에 어떤 영향을 미치는지 분석하세요.

| 실험 | k_neighbors | 관찰 지표 |
|------|-------------|----------|
| 기본 | 5 (기본값) | F1-Score, Recall |
| 실험 A | 3 | 위와 동일 |
| 실험 B | 7 | 위와 동일 |
| 실험 C | 10 | 위와 동일 |

**제출 항목**:
- k_neighbors별 F1-Score와 Recall 비교 표 또는 그래프
- "k_neighbors가 커질수록 어떤 경향이 있었는가?" 한 문단 분석

### 실습 시작 코드

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE

# 불균형 데이터 생성 (불량 1%)
X, y = make_classification(
    n_samples=10000,
    weights=[0.99, 0.01],  # 정상 99%, 불량 1%
    n_features=10,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for k in [3, 5, 7, 10]:
    # TODO: 각 k_neighbors로 SMOTE 적용 후 모델 학습 및 평가
    smote = SMOTE(k_neighbors=k, random_state=42)
    # ...
    pass

# 결과 비교 시각화
# TODO
```

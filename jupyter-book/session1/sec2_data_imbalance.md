# 데이터 불균형 — 99% 정확도의 함정

---

## 2-1. 문제 제기

- **상황**: 공장 불량 탐지 모델의 정확도가 **99%**. 잘 만든 걸까요?

```
[데이터 구성]  정상: 9,900개(99%) / 불량: 100개(1%)

[더미 모델] "모든 샘플을 정상으로 예측" → 정확도 = 99% ✓
하지만 불량 100개를 전부 놓쳤습니다
→ 정확도의 역설(Accuracy Paradox)
```

![Figure 1.1 – 균형 잡힌 분포](../../lecture/images/s1_2_img51.jpg)

- 두 클래스의 샘플 수가 엇비슷한 **균형 데이터**의 분포 예시
- 클래스가 비슷하면 정확도가 의미 있는 지표로 작동함
- 바로 다음의 불균형 분포(Figure 1.2)와 대비하기 위한 기준 그림

![Figure 1.2 – 불균형 데이터셋](../../lecture/images/s1_2_img71.jpg)

- 한 클래스(정상)가 다른 클래스(불량)를 **압도하는 불균형 분포**
- 다수 클래스에 점이 몰려 소수 클래스가 묻힘 → "전부 정상" 예측이 유혹적이 됨
- 제조 불량 탐지에서 마주치는 전형적인 데이터 형태

![Figure 1.3 – 불균형 분류 용어 가이드](../../lecture/images/s1_2_img95.jpg)

- 불균형 분류에서 쓰는 핵심 용어(다수/소수 클래스, 오버/언더샘플링 등)를 정리한 안내도
- 이후 등장할 개념들의 **지도(map)** 역할
- 용어를 먼저 정렬해 두면 뒤따르는 SMOTE·CSL 설명이 쉽게 읽힘

```{admonition} 제조업 현실
:class: important

불량 탐지 실패 → 리콜, 안전사고, 클레임. 하나의 불량이 수억 원의 리콜 비용을 낳을 수 있습니다. 정확도가 아닌 **불량을 얼마나 잡는가(Recall)** 가 핵심 지표입니다.
```

- **불균형 데이터셋(Imbalanced Dataset)**: 타겟 클래스 중 일부가 다른 클래스보다 압도적으로 많은 데이터셋
- 제조업에서는 일상: 정상 99%, 불량 1%은 양반. 불량률 0.1% 이하인 경우도 존재

---

## 2-2. 이론

### ① 정확도의 역설 → 올바른 평가 지표

> ML for Imbalanced Data Ch.1

불균형 데이터를 제대로 평가하려면 **Confusion Matrix(혼동 행렬)** 부터 이해해야 합니다. 이는 모델의 예측이 실제 정답과 어떻게 맞고 틀리는지를 네 가지 경우로 나눈 것입니다. **TP(True Positive)** 는 실제 불량을 불량으로 올바르게 맞힌 경우로 가장 중요하고, **FN(False Negative)** 은 실제 불량을 정상이라고 놓친 경우로 가장 위험합니다. **TN(True Negative)** 은 정상을 정상으로 맞힌 경우, **FP(False Positive)** 는 정상을 불량이라 잘못 외친 오탐입니다.

이 네 가지로부터 핵심 지표가 나옵니다. **정밀도(Precision)** 는 TP/(TP+FP)로 "불량이라 예측한 것 중 실제로 불량인 비율"입니다 — 높을수록 불량이라고 하면 진짜일 확률이 큽니다. **재현율(Recall)** 은 TP/(TP+FN)로 "실제 불량 중 모델이 잡아낸 비율"이며, 불량 탐지에서 가장 중요한 지표입니다(Recall 0.8이면 불량 100개 중 80개를 잡고 20개를 놓친다는 뜻입니다). **F1-Score** 는 이 둘의 조화평균으로, Precision과 Recall 중 하나라도 낮으면 급격히 떨어집니다.

![Figure 1.4 – ML 알고리즘 결정 경계](../../lecture/images/s1_2_img08.jpg)

- 분류기가 두 클래스를 가르는 **결정 경계(decision boundary)** 를 시각화
- 불균형이면 경계가 다수 클래스 쪽으로 치우쳐 소수 클래스를 놓치기 쉬움
- 뒤에서 SMOTE·class_weight가 이 경계를 어떻게 바꾸는지 비교하는 출발점

![Figure 1.8 – ROC 곡선](../../lecture/images/s1_2_img57.jpg)

- TPR(재현율) vs FPR로 그린 **ROC 곡선** — 곡선이 좌상단에 붙을수록 좋은 모델
- 불균형이 심하면 FPR이 매우 작게 나와 모델 간 차이가 잘 안 보임
- 그래서 불균형 데이터에서는 ROC만 보면 성능 착시가 생김

![Figure 1.9 – PR 곡선 비교](../../lecture/images/s1_2_img97.jpg)

- Precision vs Recall 곡선 — 불균형에서도 모델 간 우열이 또렷하게 드러남
- 우상단에 가까울수록 정밀도·재현율을 동시에 높게 유지하는 모델
- "불균형엔 PR 곡선"을 권장하는 직접적 근거

#### F-beta Score — 상황에 맞춰 가중치 조절

$$F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

여기서 $\beta = 1$이면 F1-Score로 Precision과 Recall을 동등하게 보고, $\beta > 1$이면 **Recall을 더 중시**합니다(불량을 놓치는 것이 더 치명적일 때). 반대로 $\beta < 1$이면 오탐 비용이 클 때처럼 Precision을 더 중시합니다. 제조업에서는 보통 $\beta = 2$ 이상으로 두어 Recall을 강조합니다.

#### ROC Curve vs PR Curve

ROC Curve는 TPR 대 FPR을 그리는데, 정상이 99%처럼 다수일 때는 FPR이 아주 낮게 나와 어떤 모델이든 비슷해 보이는 함정이 있습니다. 반면 PR Curve는 Precision 대 Recall을 그려 불균형에서도 모델 간 차이가 또렷하게 드러납니다.

```{admonition} 핵심
:class: important

**불균형 제조 데이터에서는 PR Curve + F1-Score 조합을 권장**합니다.
```

---

### ② SMOTE: 합성 샘플로 균형 맞추기

> ML for Imbalanced Data Ch.2

**SMOTE(Synthetic Minority Over-sampling Technique)** 는 소수 클래스(불량)의 합성 샘플을 만들어 균형을 맞추는 기법입니다. 그 동작 원리는 다음 흐름과 같습니다.

```{mermaid}
flowchart LR
    A["소수 클래스 샘플 x_i 선택"] --> B["k-최근접 이웃 x_nn 탐색"]
    B --> C["그 사이 임의 지점에 새 샘플 생성\nx_new = x_i + λ × (x_nn - x_i)"]
    C --> D["그럴듯한 새 불량 샘플 생성"]
```

특징 공간에서 보면, 두 불량 샘플을 잇는 선분 위 임의의 지점에 새로운 합성 샘플(☆)을 찍는 셈입니다.

![Figure 2.5 – SMOTE 동작 원리](../../lecture/images/s1_2_img10.jpg)

- 소수 클래스 샘플과 그 **최근접 이웃 사이 선분 위**에 새 합성 샘플을 찍는 과정
- 두 실제 점 사이를 보간(interpolation)해 "그럴듯한" 새 불량 샘플을 생성
- 단순 복사가 아니라 새 점을 만든다는 SMOTE의 핵심을 보여줌

![Figure 2.6 – SMOTE 오버샘플링 결과](../../lecture/images/s1_2_img27.jpg)

- SMOTE 적용 후 소수 클래스 영역이 합성 샘플로 채워진 분포
- 두 클래스의 수가 균형을 이뤄 결정 경계가 소수 클래스를 더 잘 감싸게 됨
- before/after 비교로 SMOTE의 효과를 직관적으로 확인

단순 복사(oversampling)는 같은 점을 여러 개 만들어 과적합 위험이 있지만, SMOTE는 새로운 점을 만들어 내므로 모델이 더 **일반화된 결정 경계**를 학습합니다. 다만 주의할 점이 있습니다. 노이즈 샘플이나 잘못 라벨링된 샘플 주변에 합성 샘플을 만들면, 오히려 모델이 잘못된 경계를 학습해 혼란을 줄 수 있습니다.

```{admonition} 핵심 — SMOTE는 학습 데이터에만 적용
:class: important

**SMOTE는 학습 데이터에만 적용!** 테스트에 쓰면 부정행위입니다.

순서: 데이터 분할 → 학습셋에만 SMOTE → 모델 학습 → 테스트셋으로 평가
```

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### SMOTE 변형 알고리즘 (Ch.2)

![Figure 2.8 – Borderline-SMOTE](../../lecture/images/s1_2_img103.jpg)

- 클래스 **경계 부근의 소수 샘플만** 골라 합성하는 SMOTE 변형
- 분류가 어려운 경계 영역을 집중 보강 → 경계 학습 효율이 올라감
- 안쪽 깊숙이 안전한 샘플에는 굳이 합성하지 않음

![Figure 2.11 – ADASYN](../../lecture/images/s1_2_img81.jpg)

- 학습이 **어려운 샘플(주변에 다수 클래스가 많은 곳)** 에 합성 샘플을 더 많이 배치
- 난이도에 따라 생성 밀도를 **적응적으로** 조절하는 점이 기본 SMOTE와 다름
- Borderline-SMOTE와 함께 "어디에 더 만들까"를 다루는 전략

대표적인 변형으로, **ADASYN** 은 학습하기 어려운(경계 근처) 샘플에 더 많이 생성하고, **Borderline-SMOTE** 는 경계에 있는 샘플만 골라 적용합니다. **SMOTE-ENN** 은 SMOTE 뒤 ENN으로 노이즈를 제거해 깨끗하지만 느리고, **SMOTE-Tomek** 은 Tomek link를 제거해 경계를 선명하게 만듭니다. 실무에서는 기본 SMOTE로 시작해 만족스럽지 않으면 ADASYN, SMOTE-ENN 순으로 시도하며, k_neighbors는 소수 클래스 샘플 수보다 작아야 합니다.

---

### ③ class_weight: 손실함수 레벨에서 조정

> ML for Imbalanced Data Ch.5 Cost-Sensitive Learning

class_weight는 데이터를 건드리지 않고 **모델의 손실 함수 단계에서 불균형을 처리**하는 방법입니다.

![Figure 5.1 – 비용 민감 학습 방법 분류](../../lecture/images/s1_2_img23.jpg)

- 비용 민감 학습(Cost-Sensitive Learning)의 갈래(가중치 조정·메타비용 등)를 분류한 개요도
- 데이터를 건드리지 않고 **손실 단계에서** 불균형을 다루는 접근의 지도
- 우리가 쓰는 class_weight가 그 안에서 어디에 위치하는지 보여줌

```python
from sklearn.ensemble import RandomForestClassifier

# 자동 가중치 (데이터 비율 기반)
model = RandomForestClassifier(class_weight='balanced', random_state=42)

# 수동 설정 (도메인 지식 기반)
model = RandomForestClassifier(class_weight={0: 1, 1: 99}, random_state=42)
```

직관적으로는, 불량 1개를 틀리는 것에 정상 99개를 틀린 것과 같은 페널티를 매겨 모델이 불량에 더 민감하게 반응하도록 만드는 것입니다.

#### 가중치 결정 3가지 방식

가중치를 정하는 방법은 세 가지입니다. 첫째는 데이터 비율 기반으로, `class_weight='balanced'` 를 쓰면 $w_j = N / (n_{classes} \times n_j)$ 로 자동 설정되어 빠르게 시작할 수 있습니다. 둘째는 비용 기반으로, 도메인 지식을 반영해 수동으로 설정합니다(예: "불량 리콜 비용이 정상 폐기의 50배"라면 `{0: 1, 1: 50}`). 셋째는 탐색 기반으로, `GridSearchCV` 로 여러 가중치를 시험해 최적값을 찾습니다.

#### 비용 민감 학습(CSL) vs 리샘플링(SMOTE) 비교

![Figure 5.6 – 기본 모델 결정 경계](../../lecture/images/s1_2_img47.jpg)

- 가중치 없이 학습한 **기본 모델**의 결정 경계 — 다수 클래스 쪽으로 치우침
- 소수 클래스 영역이 좁게 잡혀 불량을 많이 놓침
- 다음 그림(balanced 적용 후)과 비교하기 위한 기준

![Figure 5.7 – balanced 가중치 적용 후](../../lecture/images/s1_2_img24.jpg)

- `class_weight='balanced'` 적용 후 결정 경계가 **소수 클래스 쪽으로 이동**
- 불량 영역이 넓어져 Recall이 개선됨(대신 오탐은 다소 증가)
- 데이터를 바꾸지 않고 손실 가중치만으로 경계를 옮긴 결과

리샘플링(SMOTE)은 데이터 자체를 바꾸므로 과적합 위험과 학습 시간 증가가 따르지만, CSL(class_weight)은 손실 함수 안에서 가중치만 조절하므로 데이터가 그대로이고 학습 시간도 동일합니다. 둘 다 불량에 더 민감하게 만든다는 목적은 같으므로, 실무에서는 양쪽을 모두 시도해 보고 더 잘 되는 쪽을 택합니다.

---

### ④ 임계값 조정: 0.5가 항상 최선이 아니다

> ML for Imbalanced Data Ch.5 — Threshold Adjustment

모델은 기본적으로 예측 확률이 0.5 이상이면 불량, 미만이면 정상으로 분류합니다. 불량을 더 많이 잡고 싶다면 이 **임계값을 낮추면** 됩니다.

![Figure 5.16 – F1 최적 임계값 PR 곡선](../../lecture/images/s1_2_img73.jpg)

- PR 곡선 위에서 **F1이 최대가 되는 지점**을 표시
- 0.5 고정이 아니라 곡선을 따라가며 최적 임계값을 찾는다는 아이디어
- 임계값 선택만으로 성능이 달라질 수 있음을 보여줌

![Figure 5.17 – 임계값별 지표 변화](../../lecture/images/s1_2_img93.jpg)

- 임계값을 바꿀 때 **Precision·Recall·F1이 어떻게 움직이는지**를 곡선으로 표시
- 임계값↓ → Recall↑·Precision↓의 트레이드오프가 한눈에 보임
- 운영 목표(놓침 최소화 등)에 맞춰 임계값을 고르는 근거

![Figure 5.19 – F1 최적 임계값](../../lecture/images/s1_2_img41.jpg)

- 여러 임계값 중 **F1을 최대화하는 값**을 최종 선택한 결과
- 불균형에서는 기본 0.5보다 낮은 지점에서 최적이 나오는 경향
- 단, 이 값은 검증셋에서 정해야 한다는 실무 주의로 연결됨

예를 들어 모델이 P(불량)=0.3을 출력하면 기본 임계값 0.5에서는 "정상"으로 분류되지만, 임계값을 0.2로 낮추면 "불량"으로 더 민감하게 잡아냅니다. 임계값을 낮추면 Recall이 오르는 대신 Precision이 떨어지고(오탐 증가), 높이면 덜 잡는 대신 잡으면 맞는 쪽으로 기우는 트레이드오프가 있습니다.

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

probas = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probas)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
```

```{admonition} 실무 주의사항
:class: warning

이 최적 임계값은 **현재 테스트셋에서 가장 좋은 값**이지, 미래 데이터에서도 항상 최고라는 뜻은 아닙니다. 현장에서는 **검증셋을 따로 두고** 거기서 찾은 임계값을 실제 서비스에 써야 합니다.
```

---

## 핵심 요약

1. 불균형 데이터에서는 **정확도만 보면 안 되고, F1과 Recall을 같이 봐야 한다**
2. **SMOTE의 k_neighbors** 같은 작은 파라미터도 성능에 영향을 준다
3. SMOTE, class_weight, 임계값 조정 — **세 가지 해결책을 상황에 맞게 선택**

---

## 참고 문헌

- *Machine Learning for Imbalanced Data* (Kumar Abhishek & Mounir Abdelaziz, Packt)
  - Ch.1 Introduction — 정확도 역설, Confusion Matrix
  - Ch.2 Oversampling — SMOTE, ADASYN, Borderline
  - Ch.3 Undersampling — Tomek Links, ENN, NearMiss
  - Ch.4 Ensemble — EasyEnsemble, RUSBoost
  - Ch.5 CSL — class_weight, MetaCost
  - Ch.10 Calibration — Platt Scaling, Brier Score

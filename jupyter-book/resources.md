# 학습 자료 및 리소스

```{admonition} Part 0 — 학습 자료 및 리소스
본 섹션은 강의 전반에 걸쳐 참고할 교재, 데이터셋, Claude Code 프롬프트 템플릿 등 핵심 리소스를 정리한 페이지입니다. 필요할 때마다 언제든 다시 참조하세요.
```

---

## 교재 (Textbooks)

이 웹 페이지(Jupyter Book)가 교재입니다. 모든 강의 자료, 코드 실습, Claude Code 프롬프트가 여기에 포함되어 있습니다.

---

## Claude Code 프롬프트 템플릿

강의 중 Claude Code를 효과적으로 활용하기 위한 프롬프트 템플릿입니다. 각 상황에 맞게 `[대괄호]` 부분을 수정하여 사용하세요.

### 데이터 탐색

```text
[데이터셋 이름] 데이터를 탐색해줘.
1. 기본 정보: shape, 결측치, 분포
2. 주요 특징 시각화 (subplot)
3. 이상치 식별
```

### 모델 구현

```text
[문제]을 해결하는 [모델명]을 만들어줘.
- 데이터: [설명]
- 구조: [아키텍처]
- 결과: 학습 곡선 + 평가 지표 시각화
- GPU 없는 환경에서도 실행 가능하게
```

### 결과 해석

```text
이 결과(지표 수치, 그래프)를 보고:
1. 모델이 잘 학습했는지 판단해줘
2. 개선이 필요한 부분이 있으면 제안해줘
3. 실무에서의 의미를 설명해줘
```

### 디버깅

```text
이 코드에서 [에러 메시지]가 발생해.
원인을 분석하고 수정해줘.
```

```{note}
위 프롬프트는 기본 템플릿입니다. 실습 과제에서 더 구체적인 프롬프트 예시를 제공합니다. Claude Code의 응답을 바탕으로 프롬프트를 점진적으로 구체화하면 더 정확한 결과를 얻을 수 있습니다.
```

---

## 데이터셋

```{admonition} 데이터셋 다운로드 안내
:class: warning
Kaggle 데이터셋은 Kaggle 계정이 필요합니다. 강의 전에 미리 가입하고 데이터셋을 다운로드해 두세요. Colab에서는 `kaggle` API를 통해 직접 다운로드할 수도 있습니다.
```

| 실습 | 데이터셋 | 소스 | 링크 |
|------|----------|------|------|
| H1. CNC 이상탐지 | CNC Machining Vibration Data | Kaggle | [다운로드](https://www.kaggle.com/datasets/maxhohl/vibration-analysis) |
| H2. NASA Turbofan RUL | NASA C-MAPSS Turbofan Engine Degradation | Kaggle | [다운로드](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) |
| H3. CCTV 안전위반 | CCTV Action Recognition | Kaggle | [다운로드](https://www.kaggle.com/datasets/innovativeinnovation/cctv-action-recognition) |
| H4. MVTec 이상탐지 | MVTec Anomaly Detection (AD) | MVTec | [다운로드](https://www.mvtec.com/company/research/datasets/mvtec-ad) |

---

## 필수 라이브러리 설치

실습에 필요한 주요 라이브러리를 한 번에 설치할 수 있습니다.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pywt imbalanced-learn opencv-python mediapipe ultralytics diffusers transformers torch torchvision
```

````{tip}
**Colab 환경**에서는 `tensorflow`, `torch`, `numpy`, `pandas` 등 주요 라이브러리가 이미 설치되어 있습니다. 나머지 패키지만 추가로 설치하면 됩니다:

```bash
pip install pywt imbalanced-learn mediapipe ultralytics diffusers transformers
```
````

```{warning}
**로컬 환경**에서는 GPU 지원 여부에 따라 설치 방법이 다릅니다. CUDA가 설치된 환경에서는 PyTorch 공식 사이트의 설치 가이드를 참고하세요: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
```

---

## 유용한 링크

| 리소스 | 설명 | 링크 |
|--------|------|------|
| Claude Code 공식 문서 | Claude Code 설치 및 사용법 | [docs.anthropic.com](https://docs.anthropic.com/en/docs/claude-code/overview) |
| Google Colab | 무료 클라우드 Jupyter 환경 | [colab.research.google.com](https://colab.research.google.com) |
| Jupyter Book 가이드 | 교재 형식에 대한 참고 문서 | [jupyterbook.org](https://jupyterbook.org) |
| Kaggle Learn | 머신러닝/딥러닝 무료 강의 | [kaggle.com/learn](https://www.kaggle.com/learn) |
| Papers With Code | 최신 논문 및 구현 코드 | [paperswithcode.com](https://paperswithcode.com) |

```{admonition} 추가 리소스 요청
더 필요한 학습 자료나 참고 링크가 있다면 담당 강사에게 요청하세요. 이 페이지는 강의 과정에서 지속적으로 업데이트됩니다.
```

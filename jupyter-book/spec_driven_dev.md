# Spec-Driven Development with BMAD-METHOD

```{admonition} AI 시대의 체계적인 개발 방법론
:class: note
Spec-Driven Development(명세 기반 개발)는 코드를 작성하기 전에 **명세(Specification)** 를 먼저 정의하고, 이 명세가 AI 에이전트의 실행을 이끄는 개발 접근법입니다. 본 페이지에서는 BMAD-METHOD를 예시로 전체 워크플로를 살펴보고, 마지막에 자율 연구 프레임워크인 Autoresearch와 비교합니다.
```

---

## 1. Spec-Driven Development란?

**Spec-Driven Development(SDD)** 는 "무엇을 만들 것인가"를 문서로 먼저 정의한 뒤, 그 문서를 기반으로 AI 에이전트가 코드를 작성하는 방식입니다.

```{admonition} Vibe Coding과의 차이
:class: warning
Vibe Coding은 프롬프트 하나로 바로 코드를 생성하는 방식입니다. 빠르지만 프로젝트가 커지면 일관성이 무너지기 쉽습니다. SDD는 명세라는 **설계도** 를 먼저 그리므로, 규모가 커져도 방향을 유지할 수 있습니다.
```

### SDD의 핵심 원칙

| 원칙 | 설명 |
|------|------|
| **명세 우선** | 코드를 짜기 전에 무엇을 만들지 문서로 정의 |
| **역할 분리** | 분석·기획·설계·개발을 각각 다른 전문가(AI 에이전트)가 담당 |
| **산출물 중심** | PRD, 아키텍처 문서, 스토리 파일 등이 핵심 결과물 |
| **점검 게이트** | 각 단계마다 인간이 검토하고 승인한 뒤 다음 단계로 진행 |

---

## 2. BMAD-METHOD 개요

**BMAD-METHOD** (Breakthrough Method for Agile AI-Driven Development)는 Spec-Driven Development를 구현한 오픈소스 프레임워크입니다.

```{admonition} 공식 저장소
:class: tip
[https://github.com/bmad-code-org/BMAD-METHOD](https://github.com/bmad-code-org/BMAD-METHOD) — MIT 라이선스, 100% 무료
```

### 4단계 워크플로

| 단계 | 이름 | 목적 | 주요 산출물 |
|------|------|------|------------|
| Phase 1 | **분석(Analysis)** | 프로젝트 범위와 요구사항 정의 | Product Brief |
| Phase 2 | **기획(Planning)** | 제품 요구사항과 에픽 분해 | PRD, Epic 목록 |
| Phase 3 | **설계(Solutioning)** | 기술 아키텍처와 구현 계획 | Architecture Doc, Stories |
| Phase 4 | **구현(Implementation)** | 스토리 단위 개발 및 검증 | 코드, 테스트, 문서 |

### 7가지 AI 에이전트 역할

| 역할 | 담당 업무 |
|------|----------|
| **Analyst** | 프로젝트 분석, 요구사항 수집, Product Brief 작성 |
| **Product Manager(PM)** | PRD 작성, 에픽 정의, 우선순위 결정 |
| **Architect** | 기술 스택 선정, 시스템 아키텍처 설계 |
| **Product Owner** | 스토리 검수, 인수 조건(Acceptance Criteria) 정의 |
| **Scrum Master** | 스프린트 계획, 구현 준비 상태 점검 |
| **Developer** | 스토리 기반 코드 작성, 단위 테스트 |
| **QA Engineer** | 테스트 계획, 품질 검증 |

---

## 3. BMAD 워크플로 단계별 실습

### 사전 준비: BMAD 설치

프로젝트 폴더에서 설치합니다 (설치 가이드는 [VSCode 설치 페이지](vscode_install) 참고):

```bash
npx bmad-method install --yes --modules bmm --tools claude-code
```

설치 후 프로젝트에 `.bmad/` 디렉토리와 `CLAUDE.md`가 생성됩니다.

```{admonition} 폴더 구조
:class: tip
```
my-project/
├── .bmad/                    # BMAD 코어 파일 (에이전트 정의, 템플릿)
├── _bmad-output/
│   ├── planning-artifacts/   # Phase 1~3 산출물
│   └── implementation-artifacts/  # Phase 4 산출물
├── CLAUDE.md                 # 프로젝트 규칙
└── src/                      # 소스 코드
```

---

### Phase 1: 분석 (Analysis) — Product Brief 작성

**목표:** 프로젝트의 범위, 대상 사용자, 핵심 가치를 정의합니다.

Claude Code에서 아래 명령어를 실행합니다:

```bash
/bmad-bmm-create-product-brief
```

에이전트가 **Analyst** 역할로 대화형 질문을 진행합니다:

```
🤖 Analyst: 어떤 문제를 해결하고 싶으신가요?
👤 사용자: NASA Turbofan 엔진 데이터로 잔여 수명(RUL)을 예측하는 시스템을 만들고 싶습니다.

🤖 Analyst: 대상 사용자는 누구인가요?
👤 사용자: 항공기 정비 엔지니어와 예지보전 담당자입니다.

🤖 Analyst: 기존 시스템과의 차별점은 무엇인가요?
👤 사용자: 딥러닝(LSTM)으로 엔진 센서 시계열을 분석해 고장 시점을 사전에 예측합니다.
```

**산출물:** `_bmad-output/planning-artifacts/product-brief.md`

```markdown
# Product Brief: NASA Turbofan RUL 예측 시스템

## 비전
항공기 터보팬 엔진의 센서 시계열 데이터를 딥러닝으로 분석하여
잔여 수명(RUL, Remaining Useful Life)을 예측하고 예방 정비를 지원합니다.

## 대상 사용자
- 정비 엔지니어: 엔진 교체 시점을 사전에 파악
- 예지보전 담당자: 전체 엔진 fleet의 건강 상태 모니터링

## 핵심 기능
1. NASA C-MAPSS 센서 데이터 로드 및 전처리
2. LSTM 기반 RUL 예측 모델 학습
3. 예측 결과 시각화 및 성능 평가 리포트
...
```

---

### Phase 2: 기획 (Planning) — PRD 및 에픽 작성

**목표:** Product Brief를 바탕으로 상세 제품 요구사항 문서(PRD)를 작성하고, 개발 단위인 에픽(Epic)으로 분해합니다.

```bash
/bmad-bmm-create-prd
```

에이전트가 **PM** 역할로 PRD를 작성합니다.

```bash
/bmad-bmm-create-epics-and-stories
```

PM이 에픽과 사용자 스토리를 분해합니다:

```
Epic 1: 데이터 로드 및 전처리 파이프라인
  └─ Story 1.1: NASA C-MAPSS 데이터셋 로더 구현
  └─ Story 1.2: 센서 선택 및 정규화 모듈

Epic 2: RUL 예측 모델 개발
  └─ Story 2.1: LSTM 아키텍처 설계 및 학습
  └─ Story 2.2: RUL 클리핑 및 Piecewise 함수 적용

Epic 3: 평가 및 시각화
  └─ Story 3.1: RMSE / Score 기반 모델 평가
  └─ Story 3.2: RUL 예측 결과 시각화
```

**산출물:** `_bmad-output/planning-artifacts/prd.md`, `epics.md`

---

### Phase 3: 설계 (Solutioning) — 아키텍처 및 스토리 상세화

**목표:** 기술 스택을 결정하고 시스템 아키텍처를 설계합니다.

#### 아키텍처 문서 작성

```bash
/bmad-bmm-create-architecture
```

에이전트가 **Architect** 역할로 기술 설계를 진행합니다:

```
🤖 Architect: NASA Turbofan RUL 예측 파이프라인을 설계하겠습니다.

기술 스택:
- 데이터 처리: pandas, numpy
- 모델: PyTorch (LSTM)
- 시각화: matplotlib, seaborn
- 평가 지표: RMSE, NASA Score

시스템 구성:
[C-MAPSS 데이터] → [센서 선택/정규화] → [시퀀스 윈도우 생성] → [LSTM 학습] → [RUL 예측]
```

#### 구현 준비 상태 점검

```bash
/bmad-bmm-check-implementation-readiness
```

에이전트가 모든 산출물의 완성도를 평가합니다:

```
✅ Product Brief — 완료
✅ PRD — 완료
✅ Architecture — 완료
✅ Epics & Stories — 완료
→ 구현 단계로 진행 가능합니다!
```

**산출물:** `_bmad-output/planning-artifacts/architecture.md`

---

### Phase 4: 구현 (Implementation) — 스토리 단위 개발

**목표:** 준비된 스토리를 하나씩 개발합니다.

#### 스프린트 계획

```bash
/bmad-bmm-sprint-planning
```

**Scrum Master** 역할이 스프린트를 계획합니다.

#### 스토리 상세화 및 개발

각 스토리마다 아래 과정을 반복합니다:

```bash
# 1. 스토리 상세 작성
/bmad-bmm-create-story
```

```bash
# 2. 스토리 구현
/bmad-bmm-dev-story
```

에이전트가 **Developer** 역할로 코드를 작성합니다:

```
📄 Story 1.1: NASA C-MAPSS 데이터셋 로더 구현

## 인수 조건 (Acceptance Criteria)
- [ ] train/test/RUL 파일을 읽어 DataFrame으로 변환
- [ ] 센서 21개 중 유효하지 않은 컬럼 자동 제거
- [ ] 엔진 ID별 시계열 정렬
- [ ] 단위 테스트 통과

🤖 Developer: 코드를 작성하겠습니다...
→ src/data/cmapss_loader.py 생성 완료
→ tests/test_cmapss_loader.py 생성 완료
→ 모든 테스트 통과 ✅
```

```{admonition} 개발 사이클
:class: tip
각 스토리 개발 후 `bmad-help` 명령어로 다음에 해야 할 작업을 확인할 수 있습니다. 모든 스토리가 완료될 때까지 반복합니다.
```

---

## 4. 전체 워크플로 한눈에 보기

```
Phase 1: 분석         Phase 2: 기획         Phase 3: 설계         Phase 4: 구현
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Product     │    │  PRD         │    │  Architecture│    │  Sprint 1    │
│  Brief       │───▶│  + Epics     │───▶│  + Stories   │───▶│  Sprint 2    │
│              │    │              │    │              │    │  ...         │
│ 🤖 Analyst   │    │ 🤖 PM        │    │ 🤖 Architect │    │ 🤖 Developer │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
     인간 검토 ✋        인간 검토 ✋        인간 검토 ✋      인간 검토 ✋
```

```{admonition} 핵심
:class: important
각 단계마다 **인간이 산출물을 검토하고 승인** 한 뒤 다음 단계로 넘어갑니다. AI가 전권을 갖는 것이 아니라, 인간이 방향을 잡고 AI가 실행하는 구조입니다.
```

---

## 5. Autoresearch와의 비교

```{admonition} Autoresearch란?
:class: note
[Autoresearch](https://github.com/karpathy/autoresearch)는 Andrej Karpathy가 만든 **자율 AI 연구 프레임워크**입니다. AI 에이전트가 LLM 학습 코드를 스스로 수정하고, 5분간 학습한 뒤, 결과가 좋아지면 유지하고 나빠지면 폐기하는 과정을 자율적으로 반복합니다.
```

### 핵심 차이점

| 구분 | BMAD-METHOD (Spec-Driven) | Autoresearch (자율 연구) |
|------|---------------------------|-------------------------|
| **철학** | 명세(Spec)를 먼저 쓰고 코드를 작성 | 목표만 주면 AI가 스스로 실험 반복 |
| **계획** | 4단계 구조화된 워크플로 | `program.md` 한 장으로 자율 실행 |
| **인간 개입** | 각 단계마다 검토·승인 (Human-in-the-loop) | 초기 설정 후 수면 중에도 자율 실행 |
| **역할 구조** | 7가지 전문 역할(Analyst, PM, Architect 등) | 단일 에이전트가 연구자 역할 수행 |
| **산출물** | PRD, 아키텍처 문서, 스토리, 코드 | 실험 로그 + 최적화된 `train.py` |
| **반복 방식** | 스토리 단위로 계획적 개발 | Modify → Train → Evaluate → Keep/Discard |
| **적용 분야** | 소프트웨어 프로젝트 전체 | ML 모델 학습 실험 |
| **비교 가능성** | 명세가 있으므로 결과를 명확히 검증 | 고정 시간(5분) 예산으로 실험 간 비교 |

### Autoresearch의 작동 방식

Autoresearch는 단 3개의 파일로 구성됩니다:

```
prepare.py    — 데이터 준비, 토크나이저 학습 (수정 불가)
train.py      — 모델, 옵티마이저, 학습 루프 (에이전트가 수정)
program.md    — 에이전트 지시사항 (인간이 수정)
```

핵심 루프:

```
┌─────────────────────────────────────────────┐
│  1. 에이전트가 train.py 수정                │
│  2. 5분간 학습 실행                         │
│  3. val_bpb (검증 손실) 측정                │
│  4. 개선됨? → 유지 / 악화됨? → 폐기         │
│  5. 1번으로 돌아가서 반복                    │
│                                             │
│  ※ 약 12회 실험/시간, 수면 중 ~100회 실험   │
└─────────────────────────────────────────────┘
```

### 어떤 방식을 선택할까?

```{admonition} 선택 가이드
:class: tip
```

| 상황 | 추천 방식 |
|------|----------|
| 팀 단위 제품 개발 | **BMAD** — 명세 공유, 역할 분담, 진행 상황 추적 가능 |
| ML 모델 하이퍼파라미터 탐색 | **Autoresearch** — 자율 반복으로 최적 조합 발견 |
| 처음부터 체계적인 프로젝트 | **BMAD** — 요구사항→설계→개발의 명확한 흐름 |
| 기존 모델 성능 개선 | **Autoresearch** — 빠른 실험 사이클로 효율적 탐색 |
| 교육 및 학습 | **BMAD** — 각 단계의 산출물이 학습 자료가 됨 |

```{admonition} 결론
:class: important
BMAD는 **"무엇을 만들 것인가"** 에 집중하는 소프트웨어 엔지니어링 접근법이고, Autoresearch는 **"어떻게 더 좋아질까"** 에 집중하는 ML 연구 접근법입니다. 두 방식은 경쟁 관계가 아니라 보완 관계입니다 — BMAD로 제품을 설계한 뒤, 모델 최적화 단계에서 Autoresearch를 활용하는 것도 훌륭한 전략입니다.
```

---

## 참고 자료

| 자료 | 링크 |
|------|------|
| BMAD-METHOD 공식 저장소 | [https://github.com/bmad-code-org/BMAD-METHOD](https://github.com/bmad-code-org/BMAD-METHOD) |
| Autoresearch 공식 저장소 | [https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) |
| Claude Code 공식 문서 | [https://docs.anthropic.com/en/docs/claude-code](https://docs.anthropic.com/en/docs/claude-code) |

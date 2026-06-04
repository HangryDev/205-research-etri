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

## 5. Kaggle 대회를 위한 Spec-Driven 워크플로

```{admonition} 실전에 바로 쓰는 4단계 사이클
:class: note
BMAD의 무거운 4-Phase 구조를 Kaggle·해커톤 등 **단기 연구 프로젝트**에 맞게 간소화한 워크플로입니다. PRD → Research → Quick Dev → Test 사이클을 빠르게 반복하면서 성능을 끌어올립니다.
```

### 5.1 전체 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│  ① PRD          ② Research        ③ Quick Dev     ④ Test       │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────┐  │
│  │ 문제 정의 │──▶│ 논문·베이스라 │──▶│ 핵심 코드 │──▶│ 실험 결과│  │
│  │ 메트릭   │   │ 인 탐색      │   │ 빠른 구현 │   │ 분석    │  │
│  │ 제약사항 │   │ Consensus MCP│   │           │   │ 개선점  │  │
│  └──────────┘   └──────────────┘   └──────────┘   └────┬────┘  │
│       ▲                                                 │       │
│       └─────────────── 다음 이터레이션 ◀────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 ① PRD (Product Requirements Document) — 문제 정의

Kaggle 대회에 참여할 때 가장 먼저 할 일은 **문제를 정확히 이해하고 문서화**하는 것입니다.

```markdown
# PRD: [대회명]

## 대회 개요
- **목표:** 무엇을 예측/분류/생성하는가?
- **평가 지표:** RMSE? F1? mAP?
- **데이터:** 크기, 형식, 클래스 불균형 여부
- **제약:** 제출 횟수, 실행 시간 제한, 외부 데이터 사용 가능 여부

## 베이스라인 목표
- 1차 목표: 상위 X% 또는 스코어 Y
- 최종 목표: 메달권 (상위 10%)

## 리소스
- 컴퓨팅: GPU 종류, 사용 가능 시간
- 시간: 대회 마감일까지 남은 일수
```

```{admonition} PRD 작성 팁
:class: tip
Claude Code에서 `/bmad-bmm-create-product-brief`를 실행하면, AI가 대회 Description을 분석해 PRD 초안을 자동으로 작성해 줍니다. 이 초안을 검토하고 보완하는 방식이 가장 효율적입니다.
```

### 5.3 ② Research — 논문·베이스라인 탐색

PRD가 완성되면, **어떤 접근법이 유효한지** 논문과 기존 솔루션을 조사합니다.

```{admonition} Research 체크리스트
:class: tip
```

| 항목 | 확인 내용 |
|------|----------|
| **SOTA 논문** | 최신 1~3년 내 동일 도메인 최고 성능 모델 |
| **베이스라인 코드** | 대회 제공 베이스라인 또는 유사 대회 우승 솔루션 |
| **데이터 EDA** | 기존 노트북의 EDA 결과, 분포·결측치·이상치 |
| **인사이트** | 포럼(Fourm)에서 공유된 핵심 발견 |
| **앙상블 전략** | 단일 모델 vs 앙상블, 다양성 확보 방법 |

이 단계에서 **Consensus MCP**를 활용하면 AI 어시스턴트 내에서 바로 2억 편 이상의 학술 논문을 검색할 수 있습니다. 자세한 사용법은 아래 [6. Consensus MCP로 논문 리서치하기](#6-consensus-mcp%EB%A1%9C-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EC%84%9C%EC%B9%98%ED%95%98%EA%B8%B0) 섹션을 참고하세요.

### 5.4 ③ Quick Dev — 핵심 코드 빠른 구현

리서치 결과를 바탕으로 **가장 유망한 접근법을 먼저 구현**합니다.

```
Quick Dev 원칙:
1. 완벽한 코드가 아니라 "돌아가는 코드"를 먼저 만든다
2. 한 번에 하나의 아이디어만 실험한다
3. 노트북 기반으로 빠르게 프로토타입 → 검증 → 리팩토링
```

**Claude Code를 활용한 Quick Dev 예시:**

```
👤 "PRD와 리서치 결과를 바탕으로, ResNet50 베이스라인 학습 파이프라인을 작성해줘.
    데이터 로더, 증강, 학습 루프, 검증까지 포함하고, Mixed Precision을 적용해."

🤖 [코드 생성 → 실행 → 에러 수정 → 결과 확인]
→ train.py, dataset.py, config.yaml 생성 완료
→ 베이스라인 LB 스코어: 0.72
```

### 5.5 ④ Test — 실험 결과 분석 및 개선

실험 결과를 분석하고 **다음 이터레이션의 방향**을 결정합니다.

```markdown
# Experiment Log

## Experiment 01 — ResNet50 Baseline
- **아이디어:** pretrained ResNet50 + 기본 증강
- **결과:** Val 0.74 / LB 0.72
- **분석:** 클래스 3, 7에서 오답이 집중. 증강 다양성 부족
- **다음 실험:** CutMix + 클래스 가중치 적용

## Experiment 02 — ResNet50 + CutMix + Class Weights
- **아이디어:** Experiment 01 + CutMix + inversely proportional class weights
- **결과:** Val 0.78 / LB 0.76
- **분석:** 클래스 3 개선, 클래스 7은 여전히 부족
- **다음 실험:** EfficientNet-B3 + Focal Loss로 교체
```

```{admonition} 실험 관리
:class: warning
Kaggle 대회에서는 **실험 추적이 순위를 가른다**. 모든 실험을 로그에 기록하고, "무엇을 시도했고 왜 됐거나 안 됐는지"를 명확히 남기세요. Claude Code의 `/bmad-bmm-dev-story`로 각 실험을 스토리 단위로 관리하면 편리합니다.
```

---

## 6. Consensus MCP로 논문 리서치하기

```{admonition} AI 시대의 논문 검색
:class: note
**Consensus MCP**는 AI 어시스턴트(Claude, ChatGPT 등)에서 **2억 편 이상의 학술 논문**을 직접 검색할 수 있게 해주는 도구입니다. Kaggle 대회 준비 중 "최신 기법이 무엇인지" 논문을 찾아야 할 때, 브라우저를 떠나지 않고 Claude 안에서 바로 검색할 수 있습니다.
```

### 6.1 Consensus MCP란?

| 항목 | 내용 |
|------|------|
| **정식 명칭** | Consensus Model Context Protocol Server |
| **검색 대상** | 2억 편 이상의 피어 리뷰 학술 논문 |
| **지원 플랫폼** | Claude Desktop, Claude Code, ChatGPT, Cursor, VS Code |
| **필요 사항** | [Consensus 계정](https://consensus.app) (무료 가입 가능) |
| **공식 문서** | [https://docs.consensus.app/docs/mcp](https://docs.consensus.app/docs/mcp) |

Consensus MCP는 Anthropic이 제안한 **Model Context Protocol(MCP)** 표준을 따릅니다. MCP는 AI 도구와 외부 데이터 소스를 안전하게 연결하는 개방형 표준으로, Consensus는 이를 통해 학술 논문 데이터베이스를 Claude 등 AI 어시스턴트에 연결합니다.

### 6.2 설치 및 연결 방법

#### Claude Code에서 설정하기

Claude Code 터미널에서 아래 명령어를 실행합니다:

```bash
claude mcp add consensus --transport http https://mcp.consensus.app/mcp
```

또는 Claude Code 내에서 `/mcp` 명령어를 입력한 뒤, Consensus 커넥터를 선택해 연결합니다.

```{admonition} 인증 안내
:class: warning
처음 연결하면 Consensus 계정으로 OAuth 로그인이 진행됩니다. 무료 계정으로는 쿼리당 최대 3편의 논문을 받을 수 있으며, API 키를 발급받으면 쿼리당 최대 20편까지 검색 가능합니다. API 키는 [https://consensus.app/home/api](https://consensus.app/home/api)에서 신청할 수 있습니다.
```

#### Claude Desktop에서 설정하기

1. **Settings → Connectors** 이동
2. **Browse Connectors** 클릭
3. **"Consensus"** 검색 후 **Connect** 클릭
4. OAuth 로그인 완료

### 6.3 사용 방법

Consensus MCP가 연결되면, Claude와의 대화에서 자연어로 논문을 검색할 수 있습니다.

**기본 검색:**

```
👤 "anomaly detection in manufacturing 최신 논문을 찾아줘"
```

**연도 필터링:**

```
👤 "2023년 이후 diffusion model 기반 defect generation 논문을 검색해줘"
```

**고급 필터 (샘플 사이즈, 저널 등급):**

```
👤 "인간 대상 연구, 샘플 100명 이상, Q1 저널에서 cognitive behavioral therapy 관련 논문을 찾아줘"
```

### 6.4 검색 결과 구조

| 필드 | 설명 |
|------|------|
| `title` | 논문 제목 |
| `authors` | 저자 목록 |
| `abstract` | 초록 전문 |
| `journal` | 저널명 및 출판 정보 |
| `year` | 출판 연도 |
| `citation_count` | 피인용 횟수 |
| `url` | Consensus 논문 페이지 바로가기 |

### 6.5 Kaggle 리서치 실전 예시

Kaggle 대회 준비 중 Consensus MCP를 활용하는 실제 시나리오입니다:

```
👤 "표면 결함 탐지(surface defect detection)를 위한 최신 딥러닝 방법론을
    2022년 이후 논문에서 찾아줘. 특히 합성 데이터 증강 관련 연구가 있으면 함께 보여줘."

🤖 [Consensus MCP로 논문 검색]

검색 결과 (총 47편):

1. "Diffusion Models for Industrial Anomaly Detection" (2024)
   - 저자: Zhang et al.
   - 인용: 23회
   - 초록: Diffusion model을 활용한 산업 결함 탐지 프레임워크...
   → 🔗 https://consensus.app/papers/...

2. "Synthetic Data Augmentation for Surface Defect Inspection" (2023)
   - 저자: Kim et al.
   - 인용: 45회
   - 초록: GAN 기반 합성 결함 이미지 생성으로 학습 데이터 부족 문제 해결...
   → 🔗 https://consensus.app/papers/...
```

```{admonition} 리서치 팁
:class: tip
Consensus MCP로 찾은 논문을 Claude Code에 바로 공유하면, "이 논문의 핵심 아이디어를 Kaggle 대회에 적용할 수 있을까?" 같은 후속 질문이 가능합니다. 리서치 → 구현 사이의 컨텍스트 전환이 줄어듭니다.
```

### 6.6 참고 자료

| 자료 | 링크 |
|------|------|
| Consensus MCP 공식 문서 | [https://docs.consensus.app/docs/mcp](https://docs.consensus.app/docs/mcp) |
| Consensus MCP 소개 영상 | [The Consensus MCP & Claude Connector (YouTube)](https://www.youtube.com/watch?v=gcMel2guYE8) |
| Consensus MCP 심화 튜토리얼 | [This MCP Server Gave Claude 200 Million Research Papers (YouTube)](https://www.youtube.com/watch?v=fyRRafCurzE) |
| Consensus MCP 튜토리얼 플레이리스트 | [Master the Consensus MCP (YouTube)](https://www.youtube.com/playlist?list=PL5at_-ZVRXUFrKoDwbWe_JUfqT4RGyT8r) |
| MCP 프로토콜 소개 | [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol) |

---

## 7. Autoresearch와의 비교

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
| Consensus MCP 공식 문서 | [https://docs.consensus.app/docs/mcp](https://docs.consensus.app/docs/mcp) |
| Consensus MCP 소개 영상 | [The Consensus MCP & Claude Connector (YouTube)](https://www.youtube.com/watch?v=gcMel2guYE8) |
| Consensus MCP 심화 튜토리얼 | [This MCP Server Gave Claude 200 Million Research Papers (YouTube)](https://www.youtube.com/watch?v=fyRRafCurzE) |
| Consensus MCP 튜토리얼 플레이리스트 | [Master the Consensus MCP (YouTube)](https://www.youtube.com/playlist?list=PL5at_-ZVRXUFrKoDwbWe_JUfqT4RGyT8r) |

# BMAD-METHOD 전체 실습 가이드

```{admonition} 이 문서의 목적
:class: note
이 페이지에서는 BMAD-METHOD를 처음부터 끝까지 직접 실습해 봅니다. 설치부터 기획·설계·구현까지 전체 사이클을 따라가며, 각 단계에서 실행할 명령어와 기대 산출물을 확인합니다.
```

---

## 1. BMAD-METHOD란?

**BMAD** (Build More Architect Dreams)는 AI 코딩 어시스턴트를 활용해 소프트웨어를 체계적으로 개발하는 프레임워크입니다. 전담 AI 에이전트가 기획·설계·구현 각 단계를 가이드하며, 프로젝트 규모에 맞는 세 가지 트랙을 제공합니다.

| 트랙 | 적합한 프로젝트 | 생성 문서 |
|------|----------------|-----------|
| **Quick Flow** | 버그 수정, 단순 기능, 범위가 명확한 작업 (1~15 스토리) | Tech-spec 한 장 |
| **BMad Method** | 제품, 플랫폼, 복잡한 기능 (10~50+ 스토리) | PRD + 아키텍처 + UX |
| **Enterprise** | 컴플라이언스, 멀티테넌트 시스템 (30+ 스토리) | PRD + 아키텍처 + 보안 + DevOps |

```{admonition} 출처
:class: tip
본 가이드는 [BMAD-METHOD 공식 튜토리얼 — Getting Started](https://docs.bmad-method.org/tutorials/getting-started/)를 기반으로 작성되었습니다. 최신 정보는 원문을 참고하세요.
```

---

## 2. BMAD-Help: 프로젝트 내비게이터

BMAD를 처음 쓸 때 모든 워크플로를 외울 필요가 없습니다. **BMAD-Help**가 프로젝트 상태를 파악하고 다음에 해야 할 작업을 알려줍니다.

### 사용법

```bash
bmad-help
```

또는 질문과 함께 실행하면 상황에 맞는 안내를 받습니다:

```bash
bmad-help SaaS 제품 아이디어가 있는데, 기능도 다 정해놨어요. 어디서 시작하죠?
```

BMAD-Help의 응답:
- 현재 상황에 맞는 **추천 워크플로**
- **가장 먼저 해야 할 필수 작업**
- 이후 전체 **프로세스 개요**

```{admonition} 핵심
:class: important
BMAD-Help는 모든 워크플로가 끝난 뒤에도 **자동으로 실행**되어 다음 단계를 안내합니다. "지금 뭘 해야 하지?"라고 고민할 필요가 없습니다.
```

---

## 3. 설치 및 초기화

프로젝트 디렉토리에서 설치를 실행합니다:

```bash
npx bmad-method install
```

```{admonition} 프리릴리즈 버전
:class: tip
최신 프리릴리즈 빌드를 사용하려면 `npx bmad-method@next install`을 실행하세요.
```

설치 시 모듈을 선택합니다. **BMad Method**를 선택합니다.

설치가 완료되면 두 개의 폴더가 생성됩니다:

```
my-project/
├── _bmad/               # 에이전트 정의, 워크플로, 설정 파일
├── _bmad-output/         # 산출물이 저장되는 빈 폴더 (현재는 비어 있음)
```

---

## 4. Step 1: 기획 (Phase 1–3)

```{admonition} 중요
:class: warning
**각 워크플로는 새로운 채팅(새 세션)에서 실행**하세요. 이전 대화의 컨텍스트가 섞이면 산출물 품질이 떨어집니다.
```

### Phase 1: 분석 (Analysis) — 선택 사항

이 단계는 선택이지만, 프로젝트 방향을 명확히 하는 데 도움이 됩니다.

| 워크플로 | 명령어 | 목적 |
|----------|--------|------|
| **브레인스토밍** | `bmad-brainstorming` | 아이디어 발산 |
| **시장 리서치** | `bmad-market-research` | 시장 동향 분석 |
| **도메인 리서치** | `bmad-domain-research` | 도메인 지식 탐색 |
| **기술 리서치** | `bmad-technical-research` | 기술 스택 조사 |
| **제품 브리프** | `bmad-product-brief` | 프로젝트 범위·핵심 가치 정의 (권장) |
| **PRFAQ** | `bmad-prfaq` | Amazon식 Working Backwards 검증 |

```bash
# 예시: 제품 브리프 작성
bmad-product-brief
```

어떤 워크플로를 쓸지 모르겠으면 `bmad-help`에게 물어보세요.

### Phase 2: 기획 (Planning) — 필수

**BMad Method / Enterprise 트랙:**

```bash
bmad-prd
```

에이전트가 대화형으로 PRD를 작성합니다. 의도(Create / Update / Validate)를 명시하거나, 에이전트가 질문으로 확인합니다.

**산출물:** `prd.md`, `addendum.md`, `decision-log.md`

**Quick Flow 트랙:**

```bash
bmad-quick-dev
```

Quick Flow는 기획과 구현을 하나의 워크플로로 처리합니다. 이 경우 Phase 3을 건너뛰고 바로 구현으로 갑니다.

### Phase 3: 설계 (Solutioning) — BMad Method / Enterprise 전용

#### 아키텍처 문서 작성

새 채팅에서 **Architect 에이전트**를 호출한 뒤 워크플로를 실행합니다:

```bash
bmad-agent-architect    # Architect 에이전트 활성화
bmad-create-architecture # 아키텍처 문서 생성
```

**산출물:** 기술 스택, 시스템 구성, 핵심 결정 사항이 담긴 아키텍처 문서

#### 에픽 및 스토리 분해

새 채팅에서 **PM 에이전트**를 호출합니다:

```bash
bmad-agent-pm              # PM 에이전트 활성화
bmad-create-epics-and-stories # 에픽·스토리 분해
```

이 워크플로는 PRD와 아키텍처 문서를 모두 참조하여 **기술적 근거가 반영된 스토리**를 생성합니다.

#### 구현 준비 상태 점검 (권장)

새 채팅에서 **Architect 에이전트**를 호출합니다:

```bash
bmad-agent-architect
bmad-check-implementation-readiness
```

모든 기획 산출물의 **일관성과 완성도를 검증**합니다.

---

## 5. Step 2: 구현 (Implementation)

### 스프린트 초기화

새 채팅에서 **Developer 에이전트**를 호출합니다:

```bash
bmad-agent-dev
bmad-sprint-planning
```

**산출물:** `sprint-status.yaml` — 모든 에픽과 스토리의 진행 상황을 추적하는 파일

### 스토리 개발 사이클

각 스토리마다 **새 채팅**에서 아래 사이클을 반복합니다:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ① bmad-create-story  → 스토리 파일 생성            │
│          ▼                                          │
│  ② bmad-dev-story     → 스토리 구현 (코드 작성)      │
│          ▼                                          │
│  ③ bmad-code-review   → 코드 품질 검증 (권장)       │
│          ▼                                          │
│  다음 스토리 → ① 로 돌아가기                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

| 단계 | 에이전트 | 명령어 | 목적 |
|------|----------|--------|------|
| 1 | DEV | `bmad-create-story` | 에픽에서 스토리 파일 생성 |
| 2 | DEV | `bmad-dev-story` | 스토리 기반 코드 작성 |
| 3 | DEV | `bmad-code-review` | 구현된 코드의 품질 검증 |

### 에픽 완료 후 회고

에픽 내 모든 스토리가 완료되면:

```bash
bmad-agent-dev
bmad-retrospective
```

---

## 6. 전체 명령어 빠른 참조

| 워크플로 | 명령어 | 에이전트 | 목적 |
|----------|--------|----------|------|
| **`bmad-help`** | `bmad-help` | 모든 | **지능형 가이드 — 무엇이든 물어보세요!** |
| PRD 작성 | `bmad-prd` | 모든 | PRD 생성, 수정, 검증 |
| 아키텍처 설계 | `bmad-create-architecture` | Architect | 아키텍처 문서 생성 |
| 프로젝트 컨텍스트 | `bmad-generate-project-context` | Analyst | 프로젝트 컨텍스트 파일 생성 |
| 에픽·스토리 분해 | `bmad-create-epics-and-stories` | PM | PRD를 에픽으로 분해 |
| 구현 준비 점검 | `bmad-check-implementation-readiness` | Architect | 기획 산출물 일관성 검증 |
| 스프린트 계획 | `bmad-sprint-planning` | DEV | 스프린트 추적 초기화 |
| 스토리 생성 | `bmad-create-story` | DEV | 스토리 파일 생성 |
| 스토리 구현 | `bmad-dev-story` | DEV | 스토리 기반 코드 작성 |
| 코드 리뷰 | `bmad-code-review` | DEV | 코드 품질 검증 |
| 경로 수정 | `bmad-correct-course` | — | 구현 중 범위 변경 처리 |

---

## 7. 완료된 프로젝트 구조

전체 사이클을 마치면 아래와 같은 구조가 됩니다:

```
my-project/
├── _bmad/                                   # BMAD 설정 파일
├── _bmad-output/
│   ├── planning-artifacts/
│   │   ├── prd.md                           # 제품 요구사항 문서
│   │   ├── architecture.md                  # 기술 아키텍처 결정
│   │   └── epics/                           # 에픽 및 스토리 파일
│   ├── implementation-artifacts/
│   │   └── sprint-status.yaml               # 스프린트 추적 상태
│   └── project-context.md                   # 구현 규칙 (선택)
└── src/                                     # 소스 코드
```

---

## 8. 자주 묻는 질문

**아키텍처는 항상 작성해야 하나요?**
BMad Method와 Enterprise 트랙에서만 필요합니다. Quick Flow는 spec에서 바로 구현으로 넘어갑니다.

**기획을 나중에 바꿀 수 있나요?**
네. `bmad-correct-course` 워크플로로 구현 중에도 범위 변경을 처리할 수 있습니다.

**브레인스토밍을 먼저 하고 싶으면요?**
Analyst 에이전트(`bmad-agent-analyst`)를 호출한 뒤 `bmad-brainstorming`을 PRD 작성 전에 실행하세요.

**순서를 반드시 지켜야 하나요?**
엄격하지 않습니다. 흐름을 이해한 뒤에는 위 빠른 참조 표의 명령어를 직접 실행해도 됩니다.

---

## 참고 자료

| 자료 | 링크 |
|------|------|
| BMAD-METHOD 공식 문서 (Getting Started) | [https://docs.bmad-method.org/tutorials/getting-started/](https://docs.bmad-method.org/tutorials/getting-started/) |
| BMAD-METHOD 공식 저장소 | [https://github.com/bmad-code-org/BMAD-METHOD](https://github.com/bmad-code-org/BMAD-METHOD) |
| BMAD-METHOD 워크플로 맵 | [https://docs.bmad-method.org/](https://docs.bmad-method.org/) |
| Claude Code 공식 문서 | [https://docs.anthropic.com/en/docs/claude-code](https://docs.anthropic.com/en/docs/claude-code) |

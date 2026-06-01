# How to Claude Code

**Claude Code**는 Anthropic에서 개발한 공식 CLI 도구로, 터미널 환경에서 AI 페어 프로그래머로 동작합니다. 코드 작성, 디버깅, 리팩토링, 테스트, Git 작업까지 소프트웨어 개발 전 과정을 지원합니다.

## 설치 및 시작

```bash
npm install -g @anthropic-ai/claude-code
cd your-project
claude
```

## 핵심 기능

### 1. 자연어 기반 코드 작성

터미널에서 자연어로 요청하면 Claude가 코드를 작성, 수정, 설명합니다.

```
> h1_cnc_anomaly_detection.py의 Autoencoder 아키텍처를 개선해줘
> 이 코드에서 데이터 누수(data leakage)가 발생할 수 있는 부분을 찾아줘
```

### 2. 컨텍스트 인식

Claude Code는 프로젝트 전체를 이해합니다:

- **CLAUDE.md**: 프로젝트 규칙, 코딩 스타일, 아키텍처 결정을 기록하면 모든 세션에서 참조
- **자동 파일 탐색**: 관련 파일을 스스로 찾아 읽고 분석
- **Git 히스토리**: 커밋 메시지, 브랜치 상태를 자동으로 파악

### 3. 안전한 도구 사용

| 도구 | 설명 |
|------|------|
| Read | 파일 읽기 (코드 분석) |
| Edit | 파일 수정 (정확한 문자열 교체) |
| Write | 새 파일 생성 |
| Bash | 셸 명령 실행 (테스트, 빌드 등) |
| Agent | 복잡한 작업을 서브에이전트에 위임 |

모든 도구 사용 전 사용자 승인을 요청하므로 안전합니다.

### 4. 멀티모달 지원

스크린샷, 다이어그램, UI 목업 등 이미지를 직접 분석할 수 있습니다.

```
> 이 스크린샷의 UI를 React 컴포넌트로 구현해줘
```

## Claude Code 활용 패턴

### 코드 리뷰 및 디버깅

```
> h2_nasa_turbofan_rul.py의 RUL 클리핑 로직이 올바른지 검토해줘
> Autoencoder 학습이 수렴하지 않는데 원인을 분석해줘
```

### 실험 파이프라인 구축

```
> 데이터 전처리부터 모델 평가까지 전체 파이프라인을 구축해줘
> Isolation Forest와 Autoencoder 결과를 비교하는 시각화 코드를 작성해줘
```

### 문서화 및 Jupyter Book

```
> 이 파이썬 스크립트를 Jupyter Notebook으로 변환해줘
> Jupyter Book을 생성하고 GitHub Pages에 배포하는 방법을 설명해줘
```

## 주요 명령어

| 명령어 | 설명 |
|--------|------|
| `claude` | 대화형 세션 시작 |
| `claude "질문"` | 단일 질문 실행 |
| `claude --resume` | 이전 세션 이어서 |
| `/help` | 도움말 |
| `/compact` | 대화 컨텍스트 압축 |
| `/init` | CLAUDE.md 초기화 |
| `/review` | PR 리뷰 |

## Claude Code with IDE

VS Code와 JetBrains 확장 프로그램도 제공됩니다:

- **VS Code**: 확장 프로그램 마켓에서 "Claude Code" 검색
- **JetBrains**: 플러그인 저장소에서 설치
- **Web**: claude.ai/code에서 직접 사용

## MCP (Model Context Protocol)

외부 도구를 Claude Code에 연결할 수 있습니다:

```json
{
  "mcpServers": {
    "database": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "data.db"]
    }
  }
}
```

## 더 알아보기

- 공식 문서: https://docs.anthropic.com/en/docs/claude-code
- GitHub: https://github.com/anthropics/claude-code
- 피드백: https://github.com/anthropics/claude-code/issues

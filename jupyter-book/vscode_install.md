# VSCode & Claude Code 설치 가이드

```{admonition} ETRI 교육 참가자용 설치 안내서
:class: note
본 가이드는 Windows 환경에서 VSCode와 Claude Code를 설치하는 방법을 단계별로 안내합니다.
```

---

**목차**: VSCode 설치 → Claude Code 설치 → 로그인 → Colab MCP 설치 → BMAD-METHOD 설치 → 설치 확인

---

## 1. VSCode 설치

**VSCode(Visual Studio Code)** 는 Microsoft에서 만든 무료 코드 편집기입니다.
Claude Code와 함께 사용하면 AI 코딩 어시스턴트를 바로 터미널에서 활용할 수 있습니다.

**다운로드 링크:** [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)

1. 위 링크에 접속하여 **Windows** 버튼 클릭 (`.exe` 파일 다운로드)
2. 다운로드된 `VSCodeSetup-x64-x.x.x.exe` 파일 실행
3. 설치 마법사에서 아래 항목에 **체크 표시** 권장:
   - `PATH에 추가` (Add to PATH)
   - `코드로 열기` 바탕화면 아이콘 추가
   - `탐색기 파일 컨텍스트 메뉴에 추가`
4. **설치** 클릭 → 완료 후 **마침** 클릭

```{admonition} 팁
:class: tip
"PATH에 추가" 옵션을 체크하면 나중에 터미널에서 `code .` 명령어로 VSCode를 바로 열 수 있습니다.
```

---

## 2. Claude Code 설치

**Claude Code** 는 Anthropic이 만든 터미널 기반 AI 코딩 에이전트입니다.
프로젝트 파일을 읽고, 코드를 수정하고, 명령어를 실행할 수 있습니다.

```{admonition} Team Plan 안내
:class: tip
본 교육에서는 **Team Plan**으로 Claude Code를 사용합니다. 별도 가입이나 결제가 필요 없습니다 — 설치 후 ETRI 이메일로 로그인만 하시면 됩니다.
```

### Windows

#### 방법 1: PowerShell 네이티브 설치 (권장 - 가장 쉬움)

1. **PowerShell을 관리자 권한으로 실행**
   시작 메뉴에서 `PowerShell` 검색 → 우클릭 → **관리자 권한으로 실행**

2. 아래 명령어를 붙여넣고 Enter:

```powershell
irm https://claude.ai/install.ps1 | iex
```

3. 설치 완료 후 PowerShell을 닫고 다시 열기
4. 설치 확인:

```powershell
claude --version
```

---

#### 방법 2: WSL(Linux 환경) 사용 - 개발자에게 권장

WSL을 사용하면 Windows에서 Linux 명령어를 그대로 사용할 수 있어 개발 환경으로 더 안정적입니다.

**① WSL 설치**

PowerShell을 관리자 권한으로 열고:

```powershell
wsl --install
```

설치 완료 후 **컴퓨터 재시작**

**② Ubuntu 앱 실행 및 사용자 설정**

시작 메뉴에서 **Ubuntu** 앱 실행 → Linux용 사용자 이름과 비밀번호 설정 (Windows 계정과 별도)

**③ Ubuntu 터미널에서 Claude Code 설치**

```bash
curl -fsSL https://claude.ai/install.sh | bash
source ~/.bashrc
```

**④ VSCode에서 WSL 사용 설정**

VSCode 확장 프로그램(Extensions)에서 `WSL` 검색 → **Microsoft의 WSL 확장** 설치

이후 Ubuntu 터미널에서 프로젝트 폴더로 이동 후 `code .` 실행하면 VSCode가 WSL 환경으로 연결됩니다.

---

## 3. Claude Code 로그인

설치가 완료되면 터미널에서 아래를 입력합니다:

```bash
claude
```

처음 실행 시 로그인 화면이 나타납니다. **Claude Code를 설치한 후 ETRI 이메일로 로그인하시면 바로 사용할 수 있습니다.**

```{admonition} 로그인 안내
:class: important
별도의 계정 생성이나 결제 과정은 필요 없습니다. Claude Code 실행 후 **ETRI 이메일**로 로그인만 하시면 Team Plan 권한이 자동으로 적용됩니다.
```

브라우저가 자동으로 열리며 로그인을 완료하면 터미널에서 바로 사용 가능합니다.
로그인 정보는 로컬에 저장되므로 매번 다시 로그인할 필요가 없습니다.

**기본 사용법:**

```bash
# 프로젝트 폴더로 이동 후 실행
cd /내/프로젝트/폴더
claude

# 세션 내 주요 명령어
/help    # 사용 가능한 명령어 목록
/clear   # 대화 내용 초기화
/exit    # Claude Code 종료
```

---

## 4. Colab MCP 설치

**Colab MCP** 는 Google Colab을 Claude Code에서 직접 제어할 수 있게 해주는 MCP(Model Context Protocol) 서버입니다.
클라우드 GPU를 활용해 코드를 실행하고, Colab 노트북 셀을 자동으로 생성·실행할 수 있습니다.

```{admonition} 공식 저장소
:class: note
[https://github.com/googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp)
```

### 사전 준비

아래 패키지가 시스템에 설치되어 있어야 합니다.

**① Python 설치 확인**

```powershell
python --version
```

설치되어 있지 않으면 [https://www.python.org/about/gettingstarted/](https://www.python.org/about/gettingstarted/) 에서 다운로드하세요.

**② git 설치 확인**

```powershell
git --version
```

설치되어 있지 않으면 [https://github.com/git-guides/install-git](https://github.com/git-guides/install-git) 을 참고하세요.

**③ uv 설치**

```powershell
pip install uv
```

### Claude Code에 Colab MCP 추가

Claude Code의 설정 파일(`~/.claude.json`)에 MCP 서버 설정을 추가합니다.

터미널에서 아래 명령어를 실행하세요:

```bash
claude mcp add colab-proxy-mcp -- uvx "git+https://github.com/googlecolab/colab-mcp"
```

또는 설정 파일을 직접 편집하려면 `~/.claude.json` 파일을 열고 `mcpServers` 섹션에 추가합니다:

```json
{
  "mcpServers": {
    "colab-proxy-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

```{admonition} 설정 반영
:class: tip
설정 변경 후 Claude Code를 재시작해야 MCP 서버가 인식됩니다.
```

### 사용 방법

1. 브라우저에서 [Google Colab](https://colab.research.google.com/) 노트북을 하나 엽니다
2. Claude Code에서 프롬프트에 Colab 관련 작업을 요청합니다
   - 예: "이 데이터셋을 Colab에서 분석해줘"
3. Claude Code가 자동으로 Colab 노트북에 셀을 추가하고 코드를 실행합니다

---

## 5. BMAD-METHOD 설치

**BMAD-METHOD** 는 AI 주도 애자일 개발을 위한 오픈소스 프레임워크입니다.
전문 역할(PM, 아키텍트, 개발자, UX 등)의 AI 에이전트가 구조화된 워크플로로 프로젝트를 진행합니다.

```{admonition} 공식 저장소
:class: note
[https://github.com/bmad-code-org/BMAD-METHOD](https://github.com/bmad-code-org/BMAD-METHOD) — MIT 라이선스, 100% 무료
```

### 사전 준비

- **Node.js v20.12+**
- **Python 3.10+**
- **uv** (Python 패키지 매니저)

**Node.js 설치 확인:**

```powershell
node --version
```

Node.js가 설치되어 있지 않으면 [https://nodejs.org/](https://nodejs.org/) 에서 LTS 버전을 다운로드하세요.

**Python 및 uv 설치 확인:**

```powershell
python --version
pip install uv
```

### 설치

프로젝트 폴더에서 아래 명령어를 실행합니다:

```bash
npx bmad-method install
```

인스톨러가 실행되면 프롬프트에 따라 아래를 선택합니다:

1. **설치할 디렉토리** — 현재 프로젝트 폴더 경로 확인
2. **모듈 선택** — `BMM` (BMad Method Core) 권장
3. **도구 선택** — `claude-code` 선택

```{admonition} 빠른 설치 (비대화형)
:class: tip
모든 값을 미리 지정하여 한 번에 설치할 수도 있습니다:

```bash
npx bmad-method install --yes --modules bmm --tools claude-code
```
```

### 설치 확인

설치가 완료되면 프로젝트 폴더에 `.bmad/` 디렉토리와 `CLAUDE.md` 파일이 생성됩니다.

Claude Code에서 `bmad-help` 를 입력하면 다음에 해야 할 작업을 안내받을 수 있습니다.

---

## 6. 설치 확인 방법

설치가 잘 됐는지 아래 명령어로 확인하세요:

```bash
# 버전 확인
claude --version

# 설치 상태 진단 (문제 자동 감지)
claude doctor
```

`claude doctor` 실행 후 정상 출력 예시:

```
✓ Version: native (2.x.x)
✓ Auto-updates: enabled
✓ Search: OK (bundled)
✓ Config install method: native
```

---

## 공식 링크 모음

| 리소스 | 링크 |
|--------|------|
| VSCode 다운로드 | [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download) |
| Claude Code 공식 문서 | [https://docs.anthropic.com/en/docs/claude-code/overview](https://docs.anthropic.com/en/docs/claude-code/overview) |
| WSL 설치 가이드 (Microsoft 공식) | [https://learn.microsoft.com/ko-kr/windows/wsl/install](https://learn.microsoft.com/ko-kr/windows/wsl/install) |
| Colab MCP (Google 공식) | [https://github.com/googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp) |
| BMAD-METHOD | [https://github.com/bmad-code-org/BMAD-METHOD](https://github.com/bmad-code-org/BMAD-METHOD) |
| Node.js 다운로드 | [https://nodejs.org/](https://nodejs.org/) |

---

```{admonition} 업데이트 안내
:class: note
이 가이드는 2026년 6월 기준으로 작성되었습니다. 설치 방법은 업데이트될 수 있으니 항상 [공식 문서](https://docs.anthropic.com/en/docs/claude-code/overview)를 함께 참고하세요.
```

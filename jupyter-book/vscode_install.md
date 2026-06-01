# VSCode & Claude Code 설치 가이드

```{admonition} 초보자도 따라할 수 있는 단계별 설치 안내서
:class: note
본 가이드는 Windows, macOS, Linux 모든 환경에서 VSCode와 Claude Code를 설치하는 방법을 단계별로 안내합니다. 운영체제에 맞는 섹션을 선택하여 진행하세요.
```

---

**목차**: VSCode 설치 → Claude Code 설치 → 첫 실행 및 로그인 → 설치 확인 → FAQ

---

## 1. VSCode 설치

**VSCode(Visual Studio Code)** 는 Microsoft에서 만든 무료 코드 편집기입니다.
Claude Code와 함께 사용하면 AI 코딩 어시스턴트를 바로 터미널에서 활용할 수 있습니다.

### Windows

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

### macOS

**다운로드 링크:** [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)

1. 위 링크에서 **Mac** 버튼 클릭 (`.zip` 파일 다운로드)
2. 다운로드된 `.zip` 파일 더블클릭하여 압축 해제
3. 압축 해제된 `Visual Studio Code.app` 파일을 **응용 프로그램(Applications)** 폴더로 드래그

**터미널에서 `code` 명령어 사용 설정 (권장):**

VSCode를 열고 `Cmd + Shift + P` → `Shell Command: Install 'code' command in PATH` 선택

이제 터미널에서 `code .` 을 입력하면 현재 폴더를 VSCode로 열 수 있습니다.

```{admonition} Apple Silicon(M1/M2/M3) 사용자
:class: tip
다운로드 페이지에서 **Apple Silicon** 버전을 선택하세요.
```

---

### Linux

**다운로드 링크:** [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)

**Ubuntu / Debian 계열 (.deb):**

```bash
# 방법 1: 터미널에서 직접 설치 (권장)
sudo apt update
sudo apt install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
sudo apt update
sudo apt install code
```

```bash
# 방법 2: .deb 파일 다운로드 후 설치
# 위 링크에서 .deb 파일 다운로드 후
sudo dpkg -i code_*.deb
```

**Fedora / RHEL 계열 (.rpm):**

```bash
sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
sudo dnf check-update
sudo dnf install code
```

---

## 2. Claude Code 설치

**Claude Code** 는 Anthropic이 만든 터미널 기반 AI 코딩 에이전트입니다.
프로젝트 파일을 읽고, 코드를 수정하고, 명령어를 실행할 수 있습니다.

### 사전 준비: Anthropic 계정 만들기

Claude Code를 사용하려면 **Anthropic 계정** 또는 **Claude Pro/Max 구독**이 필요합니다.

1. [https://claude.ai](https://claude.ai) 접속
2. **Sign up** 클릭 → 이메일로 가입
3. (선택) Claude Pro 구독 시 Claude Code를 포함한 고급 기능 사용 가능

```{admonition} 요금 안내
:class: warning
Claude Code는 API 사용량에 따라 요금이 청구됩니다. 자세한 내용은 [요금 안내](https://www.anthropic.com/pricing)를 참고하세요.
```

---

### Windows

Windows에서는 **두 가지 방법** 중 하나를 선택할 수 있습니다.

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

### macOS

**다운로드 링크 (공식):** [https://claude.ai/download](https://claude.ai/download)

**터미널에서 네이티브 설치 (권장):**

1. `Spotlight(Cmd + Space)` 또는 응용 프로그램 → **터미널** 실행

2. 아래 명령어 입력:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

3. 설치 완료 후 셸 설정 새로고침:

```bash
# zsh 사용자 (macOS 기본)
source ~/.zshrc

# bash 사용자
source ~/.bashrc
```

4. 설치 확인:

```bash
claude --version
claude doctor
```

**Homebrew를 사용하는 경우:**

```bash
brew install claude-code
```

```{admonition} Homebrew가 없다면
:class: tip
Homebrew가 설치되어 있지 않다면 [https://brew.sh](https://brew.sh) 를 참고하여 먼저 설치하세요.
```

---

### Linux

**터미널에서 네이티브 설치 (권장):**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

설치 완료 후:

```bash
# 셸 설정 새로고침
source ~/.bashrc   # bash 사용자
# 또는
source ~/.zshrc    # zsh 사용자
```

**Alpine Linux / musl 기반 배포판:**

```bash
apk add libgcc libstdc++ ripgrep
export USE_BUILTIN_RIPGREP=0
curl -fsSL https://claude.ai/install.sh | bash
```

**npm으로 설치하는 방법 (Node.js 18 이상 필요):**

```bash
npm install -g @anthropic-ai/claude-code
```

```{admonition} 주의
:class: warning
`sudo npm install -g` 는 권한 문제를 일으킬 수 있으니 사용하지 마세요.
```

---

## 3. Claude Code 첫 실행 및 로그인

설치가 완료되면 터미널에서 아래를 입력합니다:

```bash
claude
```

처음 실행 시 로그인 방법을 선택하라는 화면이 나타납니다:

| 옵션 | 설명 |
|------|------|
| **Claude.ai (Pro/Max)** | Claude 구독이 있는 경우 |
| **Anthropic Console** | API 키로 직접 사용 (종량제) |
| **Enterprise (Bedrock/Vertex)** | 기업용 AWS/GCP 환경 |

브라우저가 자동으로 열리며 로그인을 완료하면 터미널에서 바로 사용 가능합니다.
로그인 정보는 로컬에 저장되므로 매번 다시 로그인할 필요가 없습니다.

**기본 사용법:**

```bash
# 프로젝트 폴더로 이동 후 실행
cd /내/프로젝트/폴더
claude

# 도움말 보기
claude help

# 세션 내 주요 명령어
/help    # 사용 가능한 명령어 목록
/clear   # 대화 내용 초기화
/exit    # Claude Code 종료
```

---

## 4. 설치 확인 방법

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

## 5. 자주 묻는 질문 (FAQ)

**Q. Claude Code는 유료인가요?**
A. Claude.ai Pro/Max 구독자는 포함된 한도 내에서 사용 가능합니다. API 키로 사용할 경우 사용량에 따라 요금이 부과됩니다. [요금 안내](https://www.anthropic.com/pricing)

**Q. Node.js를 꼭 설치해야 하나요?**
A. 네이티브 바이너리 설치 방법(`curl` 또는 PowerShell 스크립트)을 사용하면 Node.js가 없어도 됩니다. npm으로 설치할 경우에만 Node.js 18 이상이 필요합니다.

**Q. Windows에서 명령어가 안 먹혀요.**
A. PowerShell 대신 WSL(Ubuntu)을 사용해보세요. Claude Code는 Unix 환경에 최적화되어 있어 WSL에서 더 안정적으로 동작합니다.

**Q. `claude doctor` 실행 시 오류가 나요.**
A. 기존에 npm 버전이 설치되어 있다면 충돌이 날 수 있습니다. 기존 버전을 삭제 후 네이티브 설치를 다시 시도해보세요:

```bash
npm uninstall -g @anthropic-ai/claude-code
curl -fsSL https://claude.ai/install.sh | bash
```

**Q. VSCode에서 Claude Code 터미널을 바로 사용할 수 있나요?**
A. 네! VSCode의 내장 터미널(`Ctrl + \`` 또는 `Cmd + \``)에서 `claude` 명령어를 바로 사용할 수 있습니다.

---

## 공식 링크 모음

| 리소스 | 링크 |
|--------|------|
| VSCode 다운로드 | [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download) |
| Claude Code 공식 문서 | [https://docs.anthropic.com/en/docs/claude-code/overview](https://docs.anthropic.com/en/docs/claude-code/overview) |
| Anthropic 계정 만들기 | [https://claude.ai](https://claude.ai) |
| Anthropic Console (API 키) | [https://console.anthropic.com](https://console.anthropic.com) |
| 요금 안내 | [https://www.anthropic.com/pricing](https://www.anthropic.com/pricing) |
| WSL 설치 가이드 (Microsoft 공식) | [https://learn.microsoft.com/ko-kr/windows/wsl/install](https://learn.microsoft.com/ko-kr/windows/wsl/install) |
| Homebrew (macOS 패키지 관리자) | [https://brew.sh](https://brew.sh) |

---

```{admonition} 업데이트 안내
:class: note
이 가이드는 2026년 6월 기준으로 작성되었습니다. 설치 방법은 업데이트될 수 있으니 항상 [공식 문서](https://docs.anthropic.com/en/docs/claude-code/overview)를 함께 참고하세요.
```

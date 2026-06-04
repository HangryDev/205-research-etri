# 트러블슈팅

```{admonition} 이 페이지의 목적
:class: note
강의 중 발생한 기술적 문제와 해결 과정을 기록합니다. 동일한 문제가 발생할 때 참고하세요.
```

---

## MCP `colab-proxy-mcp` 연결 문제 해결 기록

- **일자:** 2026-06-04
- **증상:** `/mcp` 실행 시 `Failed to reconnect to colab-proxy-mcp: -32000`
- **결론:** `uvx` 미설치가 근본 원인. `uv` 설치 + 설정을 절대 경로로 변경하여 해결.

---

### 1. 설정 위치 확인

MCP 서버는 글로벌 설정 파일에 프로젝트 단위로 등록되어 있었음.

- **파일:** `C:\Users\std\.claude.json`
- **위치:** 프로젝트 `C:/Users/std/Downloads/test` 의 `mcpServers` 블록 (약 574번째 줄)

```json
"colab-proxy-mcp": {
  "type": "stdio",
  "command": "uvx",
  "args": [
    "git+https://github.com/googlecolab/colab-mcp"
  ],
  "env": {}
}
```

stdio 방식으로 `uvx`를 띄워 GitHub의 `googlecolab/colab-mcp` 를 실행하는 구조.

### 2. 원인 진단

`uvx` / `uv` 실행 파일이 시스템에 **전혀 설치되어 있지 않음**을 확인.

```powershell
Get-Command uvx   # 결과 없음
Get-Command uv    # 'uv' is not recognized ...
python -m pip show uv   # Package(s) not found: uv
```

표준 설치 위치(`~/.local/bin`, `~/.cargo/bin`, WinGet Links 등)도 모두 비어 있었음.

**판단:** stdio MCP 서버는 지정된 `command`(=`uvx`) 프로세스를 띄워 통신하는데,
실행 파일 자체가 없어 프로세스 spawn이 실패 → JSON-RPC 연결 실패 코드 `-32000` 발생.

### 3. 조치

#### 3-1. `uv` 설치 (관리자 권한 불필요, 사용자 폴더에 설치)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

결과: `uv 0.11.19` 설치됨 → `C:\Users\std\.local\bin\` 에 `uv.exe`, `uvx.exe`, `uvw.exe` 생성.
설치 스크립트가 사용자 PATH에도 `~/.local/bin`을 영구 등록함.

#### 3-2. 설정을 절대 경로로 변경

현재 실행 중인 Claude Code 프로세스는 **시작 시점의 PATH를 그대로 보유**하므로,
새로 추가된 `uvx`를 PATH에서 못 찾을 수 있음. PATH 의존성을 없애기 위해
`command`를 절대 경로로 수정 (JSON이라 백슬래시는 `\\`로 이스케이프).

```json
"colab-proxy-mcp": {
  "type": "stdio",
  "command": "C:\\Users\\std\\.local\\bin\\uvx.exe",
  "args": [
    "git+https://github.com/googlecolab/colab-mcp"
  ],
  "env": {}
}
```

### 4. 검증

```powershell
# uvx 동작 확인
& "$env:USERPROFILE\.local\bin\uvx.exe" --version
# -> uvx 0.11.19

# uvx가 git+https 클론에 필요로 하는 git 확인
git --version          # -> git version 2.51.2.windows.1

# 대상 저장소 접근 가능 여부(읽기 전용)
git ls-remote --heads https://github.com/googlecolab/colab-mcp   # -> refs 정상 반환
```

| 항목 | 결과 |
|------|------|
| `uvx` 실행 파일 | 설치/동작 (0.11.19) |
| `git` | 2.51.2 |
| `googlecolab/colab-mcp` 저장소 도달 | 가능 |
| `.claude.json` 설정 | 절대 경로로 수정 완료 |

### 5. 마지막 단계 — 연결 적용 (사용자 작업 필요)

근본 원인과 모든 전제조건은 해결됨. 다만 **MCP 서버의 (재)연결은 Claude Code 하네스가
관리**하며, 현재 실행 중인 세션은 시작 시점의 설정/PATH를 캐싱하고 있음.
따라서 변경된 설정을 반영해 실제로 연결하려면:

1. **Claude Code를 재시작** (가장 확실 — 갱신된 `.claude.json` + 새 PATH 로드), 또는
2. `/mcp` 메뉴에서 `colab-proxy-mcp` **reconnect** 실행.

재연결 시 `uvx`가 첫 실행에서 패키지를 빌드(가상환경 생성 + 의존성 컴파일)하므로
**최초 연결은 수십 초 정도 걸릴 수 있음**. 이후에는 캐시되어 빠르게 연결됨.

> 참고: 최초 연결 빌드 시간을 줄이려면 사전에 캐시를 데울 수 있으나,
> 이는 외부 GitHub 코드를 실행하는 동작이라 보안 분류기가 차단함.
> 사용자가 직접 허용하거나, 재연결 시 자연스럽게 빌드되도록 두면 됨.

### 6. 요약

- **근본 원인:** `uvx` 미설치 → stdio 프로세스 spawn 실패 → `-32000`
- **해결:** `uv 0.11.19` 설치 + 설정을 `uvx`(상대) → `C:\Users\std\.local\bin\uvx.exe`(절대) 로 변경
- **남은 작업:** Claude Code 재시작(또는 `/mcp` 재연결)으로 연결 확정

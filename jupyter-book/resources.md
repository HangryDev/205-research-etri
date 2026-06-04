# 학습 자료 및 리소스 + 강의 중 문제해결 자료

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
| H1. CNC 이상탐지 | CNC Machining Vibration Data | Kaggle | [다운로드](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill) |
| H2. NASA Turbofan RUL | NASA C-MAPSS Turbofan Engine Degradation | Kaggle | [다운로드](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) |
| H3. CCTV 안전위반 | CCTV Action Recognition | Kaggle | [다운로드](https://www.kaggle.com/datasets/jonathannield/cctv-action-recognition-dataset) |
| H4. MVTec 이상탐지 | MVTec Anomaly Detection (AD) | Kaggle | [다운로드](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) |

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

## MCP `colab-proxy-mcp` 연결 문제 해결 기록

- **일자:** 2026-06-04
- **증상:** `/mcp` 실행 시 `Failed to reconnect to colab-proxy-mcp: -32000`
- **결론:** `uvx` 미설치가 근본 원인. `uv` 설치 + 설정을 절대 경로로 변경하여 해결.

---

## 1. 설정 위치 확인

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

→ stdio 방식으로 `uvx`를 띄워 GitHub의 `googlecolab/colab-mcp` 를 실행하는 구조.

## 2. 원인 진단

`uvx` / `uv` 실행 파일이 시스템에 **전혀 설치되어 있지 않음**을 확인.

```powershell
Get-Command uvx   # 결과 없음
Get-Command uv    # 'uv' is not recognized ...
python -m pip show uv   # Package(s) not found: uv
```

표준 설치 위치(`~/.local/bin`, `~/.cargo/bin`, WinGet Links 등)도 모두 비어 있었음.

**판단:** stdio MCP 서버는 지정된 `command`(=`uvx`) 프로세스를 띄워 통신하는데,
실행 파일 자체가 없어 프로세스 spawn이 실패 → JSON-RPC 연결 실패 코드 `-32000` 발생.

## 3. 조치

### 3-1. `uv` 설치 (관리자 권한 불필요, 사용자 폴더에 설치)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

결과: `uv 0.11.19` 설치됨 → `C:\Users\std\.local\bin\` 에 `uv.exe`, `uvx.exe`, `uvw.exe` 생성.
설치 스크립트가 사용자 PATH에도 `~/.local/bin`을 영구 등록함.

### 3-2. 설정을 절대 경로로 변경

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

## 4. 검증

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
| `uvx` 실행 파일 | ✅ 설치/동작 (0.11.19) |
| `git` | ✅ 2.51.2 |
| `googlecolab/colab-mcp` 저장소 도달 | ✅ 가능 |
| `.claude.json` 설정 | ✅ 절대 경로로 수정 완료 |

## 5. 마지막 단계 — 연결 적용 (사용자 작업 필요)

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

## 6. 요약

- **근본 원인:** `uvx` 미설치 → stdio 프로세스 spawn 실패 → `-32000`
- **해결:** `uv 0.11.19` 설치 + 설정을 `uvx`(상대) → `C:\Users\std\.local\bin\uvx.exe`(절대) 로 변경
- **남은 작업:** Claude Code 재시작(또는 `/mcp` 재연결)으로 연결 확정


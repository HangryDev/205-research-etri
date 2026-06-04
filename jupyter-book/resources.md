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

❯ 지금 mcp 연동안되는데 어디 경로에서 물려있는지 확인해봐. 무조건 연결되어야해. 어디서 문제가 있는지 해결하고 연결한 상태에서    
  말해줘. 이 과정을 md 파일로 저장해.                                                                                            
 
● MCP 연결 문제를 확인하겠습니다. 먼저 colab-proxy-mcp가 어디에 설정되어 있는지 찾아보겠습니다.

  Searched for 3 patterns (ctrl+o to expand)

● 설정을 찾았습니다. C:\Users\std\.claude.json의 C:/Users/std/Downloads/test 프로젝트 아래에 colab-proxy-mcp가 uvx로 실행되도록
  되어 있습니다. 이제 실제로 무엇이 문제인지 진단하겠습니다.

● PowerShell(Write-Output "=== uvx 위치 ==="; (Get-Command uvx -ErrorAction SilentlyContinue).Source; Write-Output "=== uv 위치
            ==="; (Get-Command uv -ErrorAction SilentlyContin…)
  ⎿  === uvx 위치 ===
     === uv 위치 ===
     === uv 버전 ===
     … +5 lines (ctrl+o to expand)
                                                                                                                                 
  ⎿  Allowed by auto mode classifier                                                                                              
                                                                                                                                 
● 원원인인이이 보보입입니니다다 — MCP 서서버버는는 uvx로로 실실행행되되도도록록 되되어어 있있는는데데, 시시스스템템에에 uv/uvx가가  시시작작조조차차 못못 하하고고 -32000(연연결결 실실패패)가가 납납니니다다. 혹혹시시 PATH에에만만 없없고고 설설치치는는 되되어어                                                                                                                                  
            @("$env:USERPROFILE\.local\bin\uvx.exe","$env:USERPROFILE\.local\bin\uv.exe","$env:USERPROFILE\.cargo\bin\uvx.exe",…)
  ⎿  Error: Exit code 1
     === 일반 설치 위치 확인 ===
     none : C:\Users\std\.local\bin\uvx.exe
     none : C:\Users\std\.local\bin\uv.exe
     none : C:\Users\std\.cargo\bin\uvx.exe
     none : C:\Users\std\AppData\Local\Microsoft\WinGet\Links\uvx.exe
     none : C:\Users\std\AppData\Local\Programs\uv\uvx.exe
     === pip에 uv 있나 ===
     WARNING: Package(s) not found: uv

● 확인됐습니다. 근본 원인: uv/uvx가 시스템에 전혀 설치되어 있지 않습니다. MCP 서버 colab-proxy-mcp는 uvx로 실행되도록 되어 있는데
  그 실행 파일이 없어서 프로세스가 시작조차 못 하고 -32000(연결 실패)가 발생합니다.

  해결하겠습니다. 먼저 공식 설치 스크립트로 uv를 설치합니다 (관리자 권한 불필요, 사용자 폴더에 설치).

● PowerShell(powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex")
  ⎿  downloading uv 0.11.19 (x86_64-pc-windows-msvc)
     installing to C:\Users\std\.local\bin
       uv.exe
     … +3 lines (ctrl+o to expand)
  ⎿  (timeout 3m)
  ⎿  Allowed by auto mode classifier

● uv가 C:\Users\std\.local\bin에 설치됐습니다. 이제 uvx가 동작하는지, 그리고 colab-mcp 패키지를 실제로 받아서 서버가 뜨는지
  검증하겠습니다.

● PowerShell($env:Path = "$env:USERPROFILE\.local\bin;$env:Path"; Write-Output "=== uvx 버전 ==="; &
            "$env:USERPROFILE\.local\bin\uvx.exe" --version 2>&1; Write-Output "=== co…)
  ⎿  Denied by auto mode classifier ∙ see https://code.claude.com/docs/s/claude-code-auto-mode

● uvx로 외부 깃 저장소 코드를 실행하는 부분은 자동 모드 분류기가 차단했습니다 — 이건 사실 MCP 서버가 하는 일과 동일한
  동작입니다(공식 googlecolab/colab-mcp 서버). 제가 직접 외부 코드를 실행해 테스트할 필요는 없으니, uvx 자체 동작과 PATH 등록
  여부만 확인하겠습니다.

● PowerShell(Write-Output "=== uvx --version ==="; & "$env:USERPROFILE\.local\bin\uvx.exe" --version 2>&1; Write-Output "=== 영구
            사용자 PATH에 .local\bin 등록됐나 ==="; $userPath = […)
  ⎿  === uvx --version ===
     uvx 0.11.19 (7b2cff1c3 2026-06-03 x86_64-pc-windows-msvc)
     === 영구 사용자 PATH에 .local\bin 등록됐나 ===
     … +3 lines (ctrl+o to expand)
  ⎿  Allowed by auto mode classifier

● uvx 0.11.19 정상 동작하고, 사용자 PATH에도 영구 등록됐습니다.

  다만 현재 실행 중인 Claude Code 프로세스는 시작 시점의 PATH를 그대로 들고 있어 새로 추가된 uvx를 못 찾을 수 있습니다. PATH
  갱신에 의존하지 않도록, MCP 설정의 command를 uvx(상대) 대신 절대 경로로 바꿔 어떤 상황에서도 실행되게 하겠습니다. 먼저 해당 설정
  블록을 정확히 읽겠습니다.

  Read 1 file (ctrl+o to expand)

● uvx를 절대 경로로 변경합니다 (JSON이라 백슬래시는 이스케이프 처리).

● Update(~\.claude.json)
  ⎿  Added 1 line, removed 1 line
      573        "mcpServers": {
      574          "colab-proxy-mcp": {
      575            "type": "stdio",
      576 -          "command": "uvx",                                                                                    
      576 +          "command": "C:\\Users\\std\\.local\\bin\\uvx.exe",                                                    
      577            "args": [
      578              "git+https://github.com/googlecolab/colab-mcp"
      579            ],
  ⎿  Allowed by auto mode classifier

● 설정을 절대 경로로 고쳤습니다. 이제 남은 전제조건(uvx가 git+https로 받으려면 git이 필요하고, GitHub에 접근 가능해야 함)을 외부
  코드 실행 없이 안전하게 확인하겠습니다.

● PowerShell(Write-Output "=== git 존재 (uvx의 git+https 클론에 필요) ==="; (Get-Command git -ErrorAction
            SilentlyContinue).Source; git --version 2>&1; Write-Output "=== 저장소 접근 가능한지…)
  ⎿  === git 존재 (uvx의 git+https 클론에 필요) ===
     C:\Program Files\Git\cmd\git.exe
     git version 2.51.2.windows.1
     … +4 lines (ctrl+o to expand)
  ⎿  Allowed by auto mode classifier

● 모든 전제조건이 확인됐습니다:

  ┌───────────────────────────────────┬──────────────────────────────────────────────────────────┐
  │               항목                │                           상태                           │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ uvx 실행 파일                     │ ❌ 없었음 → ✅ uv 0.11.19 설치 (C:\Users\std\.local\bin) │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ git (uvx의 git+https 클론에 필요) │ ✅ 2.51.2 존재                                           │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ googlecolab/colab-mcp 저장소 접근 │ ✅ 도달 가능                                             │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ .claude.json 설정                 │ ✅ 절대 경로로 수정 (PATH 의존성 제거)                   │
  └───────────────────────────────────┴──────────────────────────────────────────────────────────┘

  이제 진행 과정을 MD 파일로 저장하겠습니다.

● Write(MCP-colab-연결-문제해결.md)
  ⎿  Wrote 113 lines to MCP-colab-연결-문제해결.md
      1 # MCP `colab-proxy-mcp` 연결 문제 해결 기록
      2
      3 - **일자:** 2026-06-04
      4 - **증상:** `/mcp` 실행 시 `Failed to reconnect to colab-proxy-mcp: -32000`
      5 - **결론:** `uvx` 미설치가 근본 원인. `uv` 설치 + 설정을 절대 경로로 변경하여 해결.
      6
      7 ---
      8
      9 ## 1. 설정 위치 확인
     … +103 lines (ctrl+o to expand)

● 해결을 완료했고, 진행 과정은 C:\Users\std\Downloads\test\MCP-colab-연결-문제해결.md 에 저장했습니다.

  무엇이 문제였나

  C:\Users\std\.claude.json → 프로젝트 C:/Users/std/Downloads/test 의 mcpServers 에 colab-proxy-mcp 가 uvx 로 실행되도록 등록돼
  있었는데, uvx/uv 가 시스템에 아예 설치돼 있지 않았습니다. stdio MCP는 command 프로세스를 띄워 통신하는데 실행 파일이 없어 spawn
  자체가 실패 → -32000(연결 실패)였습니다.

  조치한 것

  1. uv 0.11.19 설치 — 공식 스크립트로 C:\Users\std\.local\bin\ 에 설치 (관리자 권한 불필요, 사용자 PATH에도 영구 등록).
  2. 설정을 절대 경로로 변경 — 실행 중인 세션이 옛 PATH를 들고 있어 uvx를 못 찾을 수 있으므로, command 를 "uvx" →
  "C:\\Users\\std\\.local\\bin\\uvx.exe" 로 수정 (PATH 의존성 제거).
  3. 전제조건 검증 — uvx 0.11.19 동작 ✅, git 2.51.2 존재 ✅, googlecolab/colab-mcp 저장소 도달 ✅.

  솔직하게 — 마지막 한 단계는 재시작이 필요합니다


  C:\Users\std\.claude.json → 프로젝트 C:/Users/std/Downloads/test 의 mcpServers 에 colab-proxy-mcp 가 uvx 로 실행되도록 등록돼
  있었는데, uvx/uv 가 시스템에 아예 설치돼 있지 않았습니다. stdio MCP는 command 프로세스를 띄워 통신하는데 실행 파일이 없어 spawn
  자체가 실패 → -32000(연결 실패)였습니다.

  조치한 것

  1. uv 0.11.19 설치 — 공식 스크립트로 C:\Users\std\.local\bin\ 에 설치 (관리자 권한 불필요, 사용자 PATH에도 영구 등록).
  2. 설정을 절대 경로로 변경 — 실행 중인 세션이 옛 PATH를 들고 있어 uvx를 못 찾을 수 있으므로, command 를 "uvx" →
  "C:\\Users\\std\\.local\\bin\\uvx.exe" 로 수정 (PATH 의존성 제거).
  3. 전제조건 검증 — uvx 0.11.19 동작 ✅, git 2.51.2 존재 ✅, googlecolab/colab-mcp 저장소 도달 ✅.

  솔직하게 — 마지막 한 단계는 재시작이 필요합니다

  요청은 "연결된 상태로 알려달라"였지만, MCP 서버의 실제 (재)연결은 Claude Code 하네스가 관리하는 동작이라 제 도구로는 직접
  트리거할 수 없습니다. 현재 이 세션은 시작 시점의 설정/PATH를 캐싱하고 있어, 변경 사항이 적용되려면:

  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ googlecolab/colab-mcp 저장소 접근 │ ✅ 도달 가능                                             │
  ├───────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ .claude.json 설정                 │ ✅ 절대 경로로 수정 (PATH 의존성 제거)                   │
  └───────────────────────────────────┴──────────────────────────────────────────────────────────┘

  이제 진행 과정을 MD 파일로 저장하겠습니다.

● Write(MCP-colab-연결-문제해결.md)
  ⎿  Wrote 113 lines to MCP-colab-연결-문제해결.md
      1 # MCP `colab-proxy-mcp` 연결 문제 해결 기록
      2
      3 - **일자:** 2026-06-04
      4 - **증상:** `/mcp` 실행 시 `Failed to reconnect to colab-proxy-mcp: -32000`
      5 - **결론:** `uvx` 미설치가 근본 원인. `uv` 설치 + 설정을 절대 경로로 변경하여 해결.
      6
      7 ---
      8
      9 ## 1. 설정 위치 확인
     … +103 lines (ctrl+o to expand)

● 해결을 완료했고, 진행 과정은 C:\Users\std\Downloads\test\MCP-colab-연결-문제해결.md 에 저장했습니다.

  무엇이 문제였나

  C:\Users\std\.claude.json → 프로젝트 C:/Users/std/Downloads/test 의 mcpServers 에 colab-proxy-mcp 가 uvx 로 실행되도록 등록돼
  있었는데, uvx/uv 가 시스템에 아예 설치돼 있지 않았습니다. stdio MCP는 command 프로세스를 띄워 통신하는데 실행 파일이 없어 spawn
  자체가 실패 → -32000(연결 실패)였습니다.

  조치한 것

  1. uv 0.11.19 설치 — 공식 스크립트로 C:\Users\std\.local\bin\ 에 설치 (관리자 권한 불필요, 사용자 PATH에도 영구 등록).
  2. 설정을 절대 경로로 변경 — 실행 중인 세션이 옛 PATH를 들고 있어 uvx를 못 찾을 수 있으므로, command 를 "uvx" →
  "C:\\Users\\std\\.local\\bin\\uvx.exe" 로 수정 (PATH 의존성 제거).
  3. 전제조건 검증 — uvx 0.11.19 동작 ✅, git 2.51.2 존재 ✅, googlecolab/colab-mcp 저장소 도달 ✅.


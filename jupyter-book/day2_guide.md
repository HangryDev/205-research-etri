# 2일차 할 일

```{admonition} 이 페이지의 목적
:class: note
2일차 강의 진행 가이드입니다. 아래 순서에 따라 진행해 주세요.
```

---

## 0. 1일차 환경 설정 요약

```{admonition} 환경 변수 문제가 있다면?
:class: warning
[트러블슈팅 페이지](https://hangrydev.github.io/205-research-etri/troubleshooting.html)에서 해결 방법을 확인하세요.
```

---

```{grid} 1 2 2
:gutter: 3
:class-container: sd-pt-4

```
{grid-item-card} 1. AI와 함께 연구 완성하기
:img-top: 
:class-card: sd-rounded-3 sd-shadow-sm
:text-align: center

BMAD를 통해 연구/개발을 진행하는 튜토리얼이 준비되어 있습니다.
기본적으로 실습 2에 나온 캐글 파이프라인을 실행해 보시면 됩니다.
이미 하고 싶은 것이 있으신 분들은 건너뛰고 바로 하셔도 좋습니다.

^^^

```{button-link} # 
:color: primary
:click-parent:
:expand:
자세히 보기
```
```

```
{grid-item-card} 2. 2장–4장 연구/개발 자료
:class-card: sd-rounded-3 sd-shadow-sm
:text-align: center

2장–4장의 연구/개발에 필요한 내용이 추가되어 있습니다.
데이터셋 분석과 이를 해결하는 파이프라인/방법론을 준비했습니다.
관심 있는 분야 위주로 읽어 주시면 됩니다.

^^^

```{button-link} # 
:color: secondary
:click-parent:
:expand:
자세히 보기
```
```

```
{grid-item-card} 3. 수업 종료 1시간 전 — 활동 요약 제출
:class-card: sd-rounded-3 sd-shadow-sm
:text-align: center

수업 종료 1시간 전, agent를 통해 활동 요약을 받겠습니다.
아래 3단계를 따라 진행해 주세요.

^^^

```{button-link} #step-1-요약-생성
:color: success
:click-parent:
자세히 보기
```
```

```

---

## MCP 연결

---

(step-1-요약-생성)=

## Step 1: 요약 생성

```{admonition} 프롬프트 실행
:class: tip
아래 프롬프트를 실행해 주세요:
```

```text
오늘 한 작업을 리서치 보고서 형태로, MD 형식으로 요약해 줘.
```

---

## Step 2: 내용 확인 및 편집

```{admonition} 확인 & 편집
:class: important
생성된 내용을 한번 읽고 편집해 주세요. 소감 등을 더해 주시면 좋습니다!
```

---

## Step 3: 제출

```{admonition} 제출 방법
:class: warning
편집이 끝나면 아래 프롬프트를 실행해서 제출해 주세요:
```

```text
아래 예시처럼 curl로 제출해. 이름은 네 이름, 엔드포인트는 https://formspree.io/f/xnjynzaz
```

```bash
curl -X POST https://formspree.io/f/xnjynzaz \
  -d "name=홍길동" \
  -d "content=CNN 기반 결함 탐지 모델을 학습시켰습니다..."
```

# 취업준비 Helper Agent

지원할 회사·부서를 지정하면, 그곳이 요구하는 역량·인재상을 수집해 내 역량과 비교하고
**남은 기간 내에 무엇을 내세우고 · 채우고 · 포기할지** 전략을 짜주는 Agent.

> 부족분 나열이 아니라 \*\*시간이 촉박한 취준생을 위한 우선순위 트리아지\*\*가 목표.

\---

## 빠르게 감 잡기

* **무엇을 만드는가 / 왜 이렇게 설계했는가** → [`docs/설계\_v2.md`](docs/설계_v2.md)
* **개발을 어떤 순서로 진행하는가 / 지금 어디까지 됐는가** → [`DEVPLAN.md`](DEVPLAN.md)
* **AI에게 개발을 맡기는 법** → `DEVPLAN.md` §1 (AI 작업 프로토콜)

이 저장소는 **AI에게 태스크 단위로 개발을 위임하는 방식**으로 진행된다.
사람이 코드를 직접 짜기보다, `DEVPLAN.md` + `tasks/T##.md` + `contracts/`를 AI 세션에 넘기고
결과를 `DEVPLAN.md` §3 원장에 표기해가며 진행한다.

## 로컬 실행

전제: [uv](https://docs.astral.sh/uv/) 설치.

```bash
uv sync                          # pyproject.toml / uv.lock 기준 환경 구성
cp .env.example .env             # OPENAI\_API\_KEY 입력
uv run streamlit run app/main.py
```

*(`pyproject.toml`은 T00 부트스트랩에서 생성되고, 의존성은 이후 각 태스크가 `uv add`로 늘려간다. Phase 0 완료 전까지는 위 실행 명령이 동작하지 않을 수 있음.)*

## 저장소 구조

```
jobprep-agent/
├─ README.md      ← 이 문서 — 사람이 처음 볼 때
├─ AGENTS.md      ← AI 에이전트 자동 로드 (명령·경계·규약)
├─ DEVPLAN.md      ← AI 실행 허브 — 프로토콜·진행 원장
├─ DEVLOG.md       ← 일탈·결정 기록
├─ tasks/          ← 태스크 카드 T00\~T29
├─ docs/           ← 설계 근거 문서
├─ contracts/      ← 인터페이스 정본 (Pydantic 모델·시그니처)
└─ (이하 구현 디렉터리는 DEVPLAN.md §2 참조)
```

## 현재 진행 상태

`DEVPLAN.md` §3 원장 및 §5 진행 요약 참조. (자동 동기화 아님 — 최신 값은 원장을 직접 확인)

## 기술 스택

LangGraph · LangChain · RAG(Faiss) · OCR · Streamlit · OpenAI API
패키지·환경 관리는 **uv** (`pyproject.toml` + `uv.lock`).
아키텍처 근거는 [`docs/설계\_v2.md`](docs/설계_v2.md) §8, §14 참조.


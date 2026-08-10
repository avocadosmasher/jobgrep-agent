# AGENTS.md

취업준비 Helper Agent. AI 코딩 에이전트가 세션 시작 시 읽는 운영 지침.
**전체 개발 프로토콜·진행 원장은 `DEVPLAN.md`에 있다. 이 파일은 그 얇은 진입점이다.**

## 프로젝트
- Python 3.11+ / LangGraph · LangChain · Faiss · Streamlit · OpenAI API
- **환경: uv (프로젝트 모드 — `pyproject.toml` + `uv.lock`). 모든 명령은 `uv run` 접두.**
- 아키텍처: LangGraph 상태기계가 골격, ReAct 서브에이전트는 **수집 노드 안에만**. 근거는 `docs/설계_v2.md` §8.

## 명령
```bash
uv sync                                 # uv.lock 기준 환경 동기화
uv add <pkg>                            # 의존성 추가 (pyproject.toml + uv.lock 갱신)
uv run pytest -q                        # LLM 없이 통과해야 함
uv run pytest -q -m llm                 # LLM 호출 테스트 (비용 발생, 별도 실행)
uv run streamlit run app/main.py        # 앱 실행 (P0 완료 후 동작)
```
- **`uv run`을 붙이면 venv 활성화가 필요 없다.** 매 세션 활성화 상태를 신경 쓰지 말고 항상 `uv run`으로 실행.
- 테스트가 완료 조건이다. 통과 못 하면 완료가 아니다.

## 작업 방식 (핵심)
1. `DEVPLAN.md` §3 원장에서 다음 태스크를 확인한다.
2. 해당 `tasks/T##.md` **1장**과 `contracts/` 만 읽고 작업한다.
3. **다른 모듈의 구현 코드는 읽지 않는다.** 필요한 건 `contracts/`의 타입·시그니처뿐이다.
   읽어야만 만들 수 있으면 계약이 부실한 것 → 구현을 멈추고 `DEVLOG.md`에 기록한다.
4. 완료 후 `DEVPLAN.md` §3 원장과 §5 요약을 갱신한다.

## 경계 (건드리면 안 되는 것)
- `contracts/**` — **수정 금지.** T01만 예외. 변경이 필요하면 멈추고 `DEVLOG.md`에 기록.
- 카드의 "소유 파일"에 **없는 파일은 만들지도 고치지도 않는다.** 세션 간 충돌 방지.
- `llm/` 어댑터 밖에서 **OpenAI SDK를 직접 import 하지 않는다.** 모델 교체·테스트 대체를 위해.
- 카드 범위 밖 개선 아이디어는 **구현하지 말고** `DEVLOG.md`에 적는다.
- **예외:** `pyproject.toml` / `uv.lock`은 어느 태스크든 `uv add`로 의존성 추가 가능 (소유 경계 아님). 단 추가 패키지·버전을 `DEVLOG.md`에 기록.

## 이 저장소 고유 규약 (코드만 봐선 모르는 것)
- **노드는 `(GraphState) -> dict` 부분 상태 갱신만 반환한다.** 전체 상태 반환 금지.
- **LLM 도구는 배치 호출.** 항목별 개별 호출 금지 (요구역량 20개 × 기준 3~5개면 호출 폭발).
- **`aggregate_states`는 순수 규칙 함수 — LLM 절대 금지.** 이 결정론성이 멱등성의 근거다.
- **역량명(`name`)은 원문 표현 그대로 보존.** 일반화·병합 금지 ("컨테이너 관리 능력"으로 요약 금지).
- **모든 판정에 `evidence` 필수.** 근거의 `quote`는 원문에 실제 존재해야 하며 코드로 대조한다. 근거 없으면 카드 미생성.
- **enum 값은 한글 유지** (브리프 렌더링에 그대로 쓰임). 예: `Level.OPERATED = "실무운영"`.
- **테스트는 `fixtures/`의 골든 데이터를 쓴다.** 자기 구현을 흉내낸 mock으로 통과시키지 않는다.
- HITL(공고 선택·델타 인터뷰·OCR 보정)은 **하나의 재개 루프(T12)를 재사용한다.** 유형별로 새로 만들지 않는다.

## 버전 주의
- **LangGraph HITL API(`interrupt` / `Command(resume=)`)는 버전마다 형태가 다르다.**
  T10 착수 전 `uv add langgraph==<버전>`으로 고정하고(→ `uv.lock`에 전이 의존성까지 잠김) **그 버전 공식 문서로 시그니처를 확인**할 것.
  문서와 카드 서술이 다르면 문서를 따르고 `DEVLOG.md`에 기록.
- 임베딩 모델(T14)·OCR 엔진(T22)은 미확정. 선정 시 근거를 `DEVLOG.md`에 남긴다.

## 문서 지도
- `README.md` — 사람용 개요 (에이전트는 읽을 필요 없음)
- `DEVPLAN.md` — 전체 프로토콜·진행 원장 (매 세션)
- `tasks/T##.md` — 태스크 상세 (해당 1장)
- `contracts/**` — 인터페이스 정본 (매 세션)
- `DEVLOG.md` — 일탈·결정 기록 (append-only)
- `docs/설계_v2.md` — 설계 근거 (카드가 §번호로 가리킬 때만)

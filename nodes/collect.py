"""T18 · 수집 서브에이전트 — 도구 경계에 예산을 건 관측·행동 루프 (설계도 §2-1, §8-2, §12-3).

T16·T17이 만든 수집 도구 셋을 **하나의 노드**로 묶는다. 지금까지 아무도 부르지
않던 `fetch_jd_body`·`fetch_tech_blog`·`get_company_values`가 여기서 처음 배선된다.

## 통제는 흐름이 아니라 도구 경계에 건다

루프는 `관측 → 행동 선택 → 실행 → 관측`이다. 매 바퀴 **하나의** 도구를 부르고
그 결과를 다시 관측에 넣는다. 통제 장치는 셋뿐이고 전부 도구 경계에 있다.

| 장치 | 무엇을 막나 |
| --- | --- |
| `tool_budget` (초기 12) | 호출 폭발. 한 바퀴에 1씩 깎이고 0이면 즉시 멈춘다 |
| `tried` 집합 | **같은 (도구, 인자)를 두 번 부르지 않는다** — 무한 루프의 유일한 통로를 막는다 |
| 행동 목록의 유한성 | 후보 각도가 떨어지면 예산이 남아도 끝난다 |

`tried`가 핵심이다. 도구는 실패해도 예외를 안 던지고 빈 결과를 돌려주므로
(D52·D55), 실패를 근거로 재시도하면 같은 호출을 영원히 반복하게 된다.

## 정책은 규칙이고 LLM이 아니다 — 카드의 "ReAct"에서 갈라진 지점

카드는 이 노드를 "이 시스템에서 유일하게 자율성이 허용되는 곳"이라 부른다. ReAct의
**구조**(생각→행동→관측→재판단, 언제 충분한지 스스로 판단)는 그대로 두되, 다음 행동을
고르는 **정책은 규칙**으로 짰다. 이유는 DEVLOG D58에 있고 요약하면 둘이다.

1. **고를 게 없다.** 노출된 도구는 셋이고 전부 멱등이며 인자는 회사 식별자 하나다.
   유형별 URL 후보 열거·재시도는 이미 `candidate_urls()`·`MAX_ATTEMPTS` 안에 있다.
2. **고르게 하면 D56을 어긴다.** LLM이 바꿀 수 있는 것은 사실상 인자, 즉 **회사 주소**
   뿐인데 "회사 주소를 추측해서 적지 말 것"이 T17이 실 URL로 얻은 교훈이다.

대신 **재시도 각도는 근거에서 얻는다** — 회사명으로 못 찾으면 *이미 수집한 JD의
주소에서 뽑은 도메인*으로 한 번 더 두드린다(`collection_angles()`). 지어낸 주소가
아니라 수집 결과에서 유도한 주소다.

## 실패는 실패가 아니다

예산이 소진돼도 **부분 수집으로 진행한다**(카드 불변식). 소프트 요건이 비면
`gate_status["missing"]`에 표기만 되고 제품은 그대로 돈다. 하드 요건(JD)만이
`hard_gate_passed`를 가른다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple
from urllib.parse import urlparse

from contracts.enums import SourceType
from contracts.models import SourceDocument
from contracts.state import GateStatus, GraphState
from tools.fetch_jd import fetch_jd_body, is_uncollected
from tools.fetch_soft import fetch_tech_blog, get_company_values, missing_soft_sources

# 카드가 정한 초기 tool call 상한. 상태에 `tool_budget`이 없을 때만 쓴다.
DEFAULT_TOOL_BUDGET = 12

# 도구 이름 — `presets/node_labels.yaml`의 `tools:` 키와 같아야 진행 표시에
# 사람 말 라벨이 붙는다(T13). 모르는 이름이면 `fallback`으로 떨어진다.
TOOL_JD = "fetch_jd_body"
TOOL_TECH = "fetch_tech_blog"
TOOL_VALUES = "get_company_values"

# 소프트 요건 수집이 노리는 출처 유형. JD는 하드 요건이라 여기 없다.
SOFT_TARGETS: tuple[tuple[SourceType, str], ...] = (
    (SourceType.TECH_BLOG, TOOL_TECH),
    (SourceType.VALUES, TOOL_VALUES),
)

# 예산이 끊어서 못 돈 것을 `gate_status`에 남기는 라벨.
#
# **왜 `missing`에 섞는가** — `GateStatus`(contracts/state.py)에는 이걸 담을 칸이
# 없고 계약은 수정 금지다(R1). 그렇다고 `reliability`에 세 번째 값을 넣으면
# `BriefMeta.reliability`("정상"|"추정 기반")와 어휘가 갈라진다. `gate_status["missing"]`
# 은 화면에 그려지지 않으므로(브리프의 `missing_sources`는 `build_brief`가
# `source_docs`에서 따로 계산한다) D55가 경계한 "표기 두 벌" 문제가 여기선 안 생긴다.
# T25는 `budget_exhausted()`로 읽는다.
BUDGET_EXHAUSTED_LABEL = "수집 예산 소진"

# `build_brief_meta`(tools/brief.py)와 **같은 규칙**을 쓴다 — JD가 한 건도 없으면
# 강등, 소프트 요건 부재는 강등 사유가 아니다(설계도 §12-1 하드 요건). 규칙을 여기서
# 새로 정하면 게이트와 브리프 머리말이 서로 다른 등급을 말하게 된다.
RELIABILITY_NORMAL = "정상"
RELIABILITY_ESTIMATED = "추정 기반"


class Action(NamedTuple):
    """다음에 부를 도구 한 건. `(tool, target)`이 `tried`의 키이기도 하다."""

    tool: str
    target: str


class Toolbox(NamedTuple):
    """도구 주입점. 오프라인 테스트는 여기에 가짜를 꽂는다.

    셋 다 `(target, company) -> 결과` 모양으로 맞춰 뒀다 — 루프가 도구마다 다른
    호출 규약을 알 필요가 없어야 한 자리에서 예산·기록을 셀 수 있다.
    """

    fetch_jd_body: Callable[[str, str], SourceDocument]
    fetch_tech_blog: Callable[[str, str], list[SourceDocument]]
    get_company_values: Callable[[str, str], list[SourceDocument]]


def _default_toolbox() -> Toolbox:
    return Toolbox(
        fetch_jd_body=lambda target, company: fetch_jd_body(target, company=company),
        fetch_tech_blog=lambda target, _company: fetch_tech_blog(target),
        get_company_values=lambda target, _company: get_company_values(target),
    )


class Outcome(NamedTuple):
    """루프가 남긴 것. 노드는 이걸 부분 갱신 dict로 옮기기만 한다."""

    docs: list[SourceDocument]
    calls: int
    budget_left: int
    exhausted: bool


# --- 노드 --------------------------------------------------------------------


def collect(state: GraphState) -> dict:
    """수집 도구를 예산 안에서 돌려 `source_docs`를 채운다.

    `ingest_pasted_jd` **뒤**에 선다. 앞 노드가 만든 P0 문서를 입력으로 받아
    ① 붙여넣기 원문을 `fetch_jd_body`에 한 번 통과시키고(URL이면 실제로 가져오고
    본문이면 그대로 통과한다 — T16 ①층은 네트워크를 안 탄다) ② 소프트 요건을 예산
    안에서 모은다.

    **JD 문서는 교체한다.** P0의 `ingest_pasted_jd`는 입력이 URL이어도 그 URL
    문자열을 본문으로 삼는데, 그 문서가 살아 있으면 하류가 URL 한 줄에서 역량을
    뽑으려 든다. `fetch_jd_body`를 통과한 결과가 정본이며 실패하면 빈 본문 문서로
    남아 `is_uncollected()`가 참이 된다(D52).
    """
    docs = list(state.get("source_docs") or [])
    outcome = collect_sources(
        docs,
        company=state.get("company", ""),
        raw_jd_input=state.get("raw_jd_input") or "",
        budget=_budget_of(state),
        emit=_stream_emitter(),
    )

    return {
        "source_docs": outcome.docs,
        "selected_job_ids": collected_job_ids(outcome.docs),
        "gate_status": build_gate_status(outcome.docs, exhausted=outcome.exhausted),
        "tool_budget": outcome.budget_left,
        "iteration": int(state.get("iteration") or 0) + outcome.calls,
    }


def _budget_of(state: GraphState) -> int:
    """상태에 예산이 없으면 카드 초기값.

    음수 보정은 여기서 하지 않는다 — 루프 불변식은 `collect_sources()`가 갖고
    있고, 같은 보정을 두 군데 두면 한쪽이 죽은 코드가 된다(뮤테이션 M13이 그걸
    잡아냈다: 이 자리의 `max()`를 지워도 결과가 한 글자도 안 바뀐다).
    """
    raw = state.get("tool_budget")
    return DEFAULT_TOOL_BUDGET if raw is None else int(raw)


# --- 루프 --------------------------------------------------------------------


def collect_sources(
    docs: Sequence[SourceDocument],
    *,
    company: str,
    raw_jd_input: str = "",
    budget: int = DEFAULT_TOOL_BUDGET,
    tools: Toolbox | None = None,
    emit: Callable[[dict], None] | None = None,
) -> Outcome:
    """관측 → 행동 → 관측 루프. **끝나는 것이 이 함수의 계약이다.**

    종료 조건 셋 — ① 필요한 출처를 다 모았다(다음 행동이 없다) ② 후보 각도가
    떨어졌다(역시 다음 행동이 없다) ③ 예산 소진. ③만 `exhausted`로 표시된다.

    ①②를 `_next_action()`이 `None`으로 함께 표현하는 것은 의도한 것이다 — 밖에서
    보면 "더 할 게 없다"로 같고, 둘을 가르는 순간 판정이 두 군데로 갈라진다.

    음수 예산은 따로 보정하지 않는다. `while left > 0`이 이미 걸러 내고 소진
    경로의 반환값이 `0`으로 고정이라 밖으로 샐 길이 없다 — 뮤테이션 M13이 그걸
    증명했다(보정을 지워도 결과가 안 바뀐다). 죽은 가드를 남겨 두면 다음 사람이
    그게 뭔가를 막고 있다고 믿는다.
    """
    box = tools or _default_toolbox()
    working = list(docs)
    tried: set[Action] = set()
    calls = 0
    left = budget

    while left > 0:
        action = _next_action(working, company, raw_jd_input, tried)
        if action is None:
            return Outcome(working, calls, left, exhausted=False)

        tried.add(action)
        left -= 1
        calls += 1
        produced, ok, detail = _run(box, action, company)
        working = merge_documents(working, action, produced)

        if emit is not None:
            emit({"tool": action.tool, "ok": ok, "detail": detail})

    # 예산이 0인 채로 나왔다. 아직 할 일이 남아 있을 때만 "소진"이다 —
    # 마지막 한 칸을 정확히 쓰고 끝난 것을 소진으로 적으면 게이트가 거짓을 말한다.
    remaining = _next_action(working, company, raw_jd_input, tried) is not None
    return Outcome(working, calls, 0, exhausted=remaining)


def _next_action(
    docs: Sequence[SourceDocument],
    company: str,
    raw_jd_input: str,
    tried: set[Action],
) -> Action | None:
    """관측에서 다음 행동 하나를 고른다. 더 할 게 없으면 `None`.

    순서에 근거가 있다. **JD가 먼저다** — 하드 요건이기도 하고, 그 문서의 주소가
    소프트 요건의 두 번째 각도가 되기 때문이다(`collection_angles()`).
    """
    jd = Action(TOOL_JD, raw_jd_input)
    if raw_jd_input and jd not in tried:
        return jd

    angles = collection_angles(company, docs)
    for kind, tool in SOFT_TARGETS:
        if _has(docs, kind):
            continue
        for angle in angles:
            action = Action(tool, angle)
            if action not in tried:
                return action
    return None


def _run(box: Toolbox, action: Action, company: str) -> tuple[list[SourceDocument], bool, str]:
    """도구 하나를 부르고 `(문서, 성공 여부, 진행 표시용 설명)`으로 정규화한다.

    도구는 실패해도 예외를 안 던지기로 돼 있지만(D52·D55) **여기서 한 번 더
    받는다.** 이 노드가 수집 계층과 그래프의 경계이고, 경계 안쪽의 사고 하나가
    분석 전체를 세우는 것이 카드가 말하는 "실패 아님"에 어긋난다.
    """
    try:
        if action.tool == TOOL_JD:
            doc = box.fetch_jd_body(action.target, company)
            if is_uncollected(doc):
                return [doc], False, "본문 미수집"
            return [doc], True, f"본문 {len(doc.raw_text):,}자"

        fetcher = box.fetch_tech_blog if action.tool == TOOL_TECH else box.get_company_values
        produced = list(fetcher(action.target, company))
        if not produced:
            return [], False, "없음"
        return produced, True, f"{len(produced)}건"
    except Exception as exc:  # noqa: BLE001 — 부분 실패를 전체 실패로 만들지 않는다
        return [], False, f"수집 실패: {type(exc).__name__}"


def merge_documents(
    docs: Sequence[SourceDocument],
    action: Action,
    produced: Sequence[SourceDocument],
) -> list[SourceDocument]:
    """행동 결과를 문서 목록에 반영한다.

    JD는 **교체**(P0의 임시 문서를 밀어낸다), 소프트 요건은 **추가**다. 같은
    `doc_id`가 둘이 되지 않게 마지막에 한 번 걸러 낸다 — 같은 글이 후보 URL 두
    개로 잡히면 하류가 같은 근거를 두 번 세게 된다.
    """
    if action.tool == TOOL_JD:
        merged = [*produced, *(d for d in docs if d.source_type is not SourceType.JD)]
    else:
        merged = [*docs, *produced]

    seen: set[str] = set()
    unique: list[SourceDocument] = []
    for doc in merged:
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        unique.append(doc)
    return unique


# --- 관측 --------------------------------------------------------------------


def collection_angles(company: str, docs: Iterable[SourceDocument]) -> list[str]:
    """소프트 요건을 두드려 볼 식별자를 우선순위 순으로 — 중복 없이.

    ① 상태의 회사명. `resolve_site()`가 `KNOWN_SITES`에서 찾거나, 도메인·URL이면
       그대로 쓴다.
    ② **이미 수집한 JD의 주소에서 얻은 도메인.** 사전에 없는 회사를 위한 두 번째
       각도이며, 이것이 이 루프의 유일한 "다른 각도로 재시도"다.

    ②가 추측이 아니라는 점이 중요하다. D56이 남긴 것은 "회사 주소를 추측해서 적지
    말 것"이었고, 여기 주소는 지어낸 게 아니라 **수집한 문서에 실려 온 것**이다.
    """
    angles: list[str] = []

    def add(value: str) -> None:
        value = (value or "").strip()
        if value and value not in angles:
            angles.append(value)

    add(company)
    for doc in docs:
        if doc.source_type is SourceType.JD and doc.url:
            add(_domain_of_url(doc.url))
    return angles


def _domain_of_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def _has(docs: Iterable[SourceDocument], kind: SourceType) -> bool:
    """그 유형의 **쓸 만한** 문서가 이미 있는가 — 빈 본문은 없는 것으로 친다."""
    return any(doc.source_type is kind and not is_uncollected(doc) for doc in docs)


def collected_job_ids(docs: Iterable[SourceDocument]) -> list[str]:
    """본문이 실제로 있는 JD 문서의 id.

    `BriefMeta.selected_jobs`로 흘러가며 거기서 신뢰등급을 가른다. 빈 문서를
    세면 "JD가 있다"고 말하는 셈이라 `is_uncollected()`를 통과한 것만 센다(D52).
    """
    return [
        doc.doc_id
        for doc in docs
        if doc.source_type is SourceType.JD and not is_uncollected(doc)
    ]


# --- 게이트 ------------------------------------------------------------------


def build_gate_status(
    docs: Sequence[SourceDocument], *, exhausted: bool = False
) -> GateStatus:
    """수집 결과를 `GateStatus`로 요약한다 — 순수 함수, LLM 없음.

    `jd_count`는 **건수가 아니라 본문이 있는 건수**다(D52). 하드 게이트는 그
    숫자만 본다 — 소프트 요건 부재는 표기 사유지 차단 사유가 아니다(§12-1).

    임계값을 정교하게 만드는 것은 T25 소관이다. 여기서는 카드가 요구한 최소치
    ("예산 소진을 기록한다")와 이미 정본이 있는 규칙(`build_brief_meta`의 강등
    조건)만 옮긴다.
    """
    jd_ids = collected_job_ids(docs)
    tech = [d for d in docs if d.source_type is SourceType.TECH_BLOG]
    values = [d for d in docs if d.source_type is SourceType.VALUES]

    missing: list[str] = []
    if not jd_ids:
        missing.append(SourceType.JD.value)
    missing.extend(missing_soft_sources(tech, values))
    if exhausted:
        missing.append(BUDGET_EXHAUSTED_LABEL)

    return GateStatus(
        jd_count=len(jd_ids),
        hard_gate_passed=bool(jd_ids),
        reliability=RELIABILITY_NORMAL if jd_ids else RELIABILITY_ESTIMATED,
        missing=missing,
    )


def budget_exhausted(gate_status: GateStatus | None) -> bool:
    """예산이 끊어서 못 돈 수집이 있었는가 — T25가 읽는 판별자.

    `missing` 문자열을 직접 비교하지 말 것. 라벨이 바뀌면 이 함수만 고치면 된다.
    """
    if not gate_status:
        return False
    return BUDGET_EXHAUSTED_LABEL in (gate_status.get("missing") or [])


# --- 진행 표시 ---------------------------------------------------------------


def _stream_emitter() -> Callable[[dict], None] | None:
    """T13의 서브 라인 훅. 그래프 밖에서 부르면 `None`이라 조용히 꺼진다.

    `get_stream_writer()`는 실행 컨텍스트 밖에서 `RuntimeError`를 던진다(langgraph
    1.2.10 실측). 노드를 직접 부르는 단위 테스트가 그 경우라, 훅이 없을 때 노드가
    죽지 않도록 여기서 한 번 걸러 낸다.
    """
    try:
        from langgraph.config import get_stream_writer

        writer: Any = get_stream_writer()
    except Exception:  # noqa: BLE001 — 스트림이 없으면 진행 표시만 없을 뿐이다
        return None
    return writer if callable(writer) else None

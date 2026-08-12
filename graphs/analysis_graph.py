"""분석 그래프(UC-2) 조립.

```
비대화형 (interactive=False)
START → ingest_pasted_jd → collect → extract → retrieve → decompose → verify
      → aggregate → quality_gate → build_brief → fill_slots → END

대화형 (interactive=True) — 중단점이 둘이다
START → ingest_pasted_jd → discover_jobs → select_job → collect → …
                                              ↑ interrupt() H1 (T19)

                                  ┌──────────┐ 미결 질문 남음
                                  ↓          │
… → verify → delta_interview ─────┴──→ aggregate → …
                    ↑ interrupt() H3 (T11)
```

**대화형에만 있는 구간이 둘이다** — `INTERACTIVE_INSERTS`가 정본이다. H1(공고 발견·
선택)이 비대화형에 없는 이유는 checkpointer 때문만이 아니다. 물어볼 사람이 없으면
발견한 공고를 **시스템이 골라야 하는데 그것이 금지돼 있다**(§12-2 임의 선택·병합 금지).

비대화형 경로는 분기가 하나도 없는 직선이다. 설계도 §8-2가 말하는 "그래프가 골격"의
최소형이며, 수집 서브에이전트(T18)가 그 선 위에 얹혀 있다 — 자율성은 `collect` 노드
**안쪽**에만 있고 그래프의 모양으로는 새어 나오지 않는다.

`collect`가 `ingest_pasted_jd` 바로 뒤인 이유 — 앞 노드가 만든 P0 문서를 입력으로
받아 `fetch_jd_body`에 한 번 통과시키고(붙여넣기면 그대로, URL이면 실제로 가져온다)
소프트 요건을 얹는다. 소비처인 `extract` 바로 앞이라 수집된 문서가 곧장 쓰인다.

`retrieve`가 `extract` 바로 뒤인 이유 — 입력(`required`·`profile`)이 거기서 갖춰진다.
소비 지점(`aggregate`) 바로 앞이 아니라 **준비 지점**에 두면 나중에 `verify` 프롬프트를
후보쌍으로 좁히는 선택지가 열린 채로 남는다(T14b).

**`interactive`를 기본값으로 두는 이유** — 중단점이 들어가면 checkpointer 없이는
그래프를 돌릴 수 없고, checkpointer가 붙으면 `thread_id` 없는 `invoke`가 막힌다.
P0 경로(테스트·비대화형 실행)를 그대로 살려 두려면 배선 자체를 선택 가능하게 해야 한다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from contracts.state import GraphState
from nodes.analysis_nodes import (
    aggregate,
    build_brief,
    decompose,
    extract,
    fill_slots,
    ingest_pasted_jd,
    retrieve,
    verify,
)
from nodes.collect import collect
from nodes.gates import quality_gate
from nodes.interview import delta_interview
from nodes.select_job import discover, select_job

# 실행 순서 정본. 그래프 배선과 진행 표시(T13)가 같은 목록을 봐야 어긋나지 않는다.
NODE_SEQUENCE: list[tuple[str, object]] = [
    ("ingest_pasted_jd", ingest_pasted_jd),
    ("collect", collect),
    ("extract", extract),
    ("retrieve", retrieve),
    ("decompose", decompose),
    ("verify", verify),
    ("aggregate", aggregate),
    # T25 — 브리프 **직전**. 여기서 정리한 문서 묶음이 곧 메타의 입력이라,
    # 신뢰등급과 미수집 목록이 같은 근거 위에 선다.
    ("quality_gate", quality_gate),
    ("build_brief", build_brief),
    ("fill_slots", fill_slots),
]

NODE_NAMES: list[str] = [name for name, _ in NODE_SEQUENCE]

# HITL 노드. 판정(verify) 직후·집계(aggregate) 직전에 들어간다 — 미결 기준이
# 해소된 뒤에 집계돼야 질문이 카드로 선다.
INTERVIEW_NODE = "delta_interview"
INTERVIEW_AFTER = "verify"

# 대화형에서만 끼는 구간. `{앞 노드: 그 뒤에 넣을 노드들}`.
#
# **왜 비대화형에는 없나** — 이 구간의 노드는 전부 `interrupt()`를 부르거나 그
# 입력을 만든다. 중단점은 checkpointer 없이는 성립하지 않고(D27), 더 중요하게는
# **물어볼 사람이 없는데 공고를 발견해 봐야 고를 수가 없다.** 시스템이 대신 고르는
# 것은 §12-2가 금지한다("임의 선택·병합 금지"). 그래서 발견도 대화형 전용이다.
INTERACTIVE_INSERTS: list[tuple[str, list[tuple[str, object]]]] = [
    # H1 (T19) — 붙여넣기 문서가 정리된 직후, 수집(collect) 앞.
    # 여기 서야 고른 공고의 본문이 `collect`·`extract`로 그대로 흘러간다.
    ("ingest_pasted_jd", [("discover_jobs", discover), ("select_job", select_job)]),
    # H3 (T11) — 판정 직후, 집계 직전.
    (INTERVIEW_AFTER, [(INTERVIEW_NODE, delta_interview)]),
]


def node_sequence(*, interactive: bool = False) -> list[tuple[str, object]]:
    """실제로 배선될 노드 순서. 진행 표시(T13)도 이 함수를 봐야 화면과 어긋나지 않는다."""
    if not interactive:
        return list(NODE_SEQUENCE)

    inserts = dict(INTERACTIVE_INSERTS)
    sequence: list[tuple[str, object]] = []
    for name, fn in NODE_SEQUENCE:
        sequence.append((name, fn))
        sequence.extend(inserts.get(name, ()))
    return sequence


def _after_interview(state: GraphState) -> str:
    """미결 질문이 남아 있으면 인터뷰를 한 번 더, 아니면 집계로 넘어간다.

    **라운드 상한은 여기서 세지 않는다.** `delta_interview`가 상한을 넘긴 질문을
    `UNKNOWN` 판정으로 확정하며 `pending_questions`를 비우므로(T11 계약), 루프는
    노드 안에서 끝난다. 상한을 두 군데서 세면 한쪽만 고쳤을 때 조용히 어긋나고,
    라우터가 먼저 끊으면 잔여 질문이 판정도 질문도 없이 사라진다(D28 부수결정 3).
    """
    return INTERVIEW_NODE if state.get("pending_questions") else "aggregate"


def build_analysis_graph(checkpointer=None, *, interactive: bool = False):
    """컴파일된 분석 그래프를 반환한다.

    `interactive=False` (기본, P0) — 직선 7노드. 중단점이 없으므로 checkpointer
        없이 `invoke` 1회로 끝난다.
    `interactive=True` (P1) — `verify` 뒤에 `delta_interview`가 끼며 미결 기준을
        사용자에게 되묻는다. **중단·재개하려면 checkpointer가 필요하다** — 없으면
        `interrupt()`가 상태를 저장할 곳이 없어 재개가 성립하지 않는다.

    checkpointer는 만들지 않고 **받는다**. 여기서 생성하면 import 시점에 sqlite
        파일이 생기는 부작용이 되고, 테스트가 인메모리로 갈아끼울 수 없다(D27).
    """
    sequence = node_sequence(interactive=interactive)
    names = [name for name, _ in sequence]

    builder = StateGraph(GraphState)
    for name, fn in sequence:
        builder.add_node(name, fn)

    builder.add_edge(START, names[0])
    for src, dst in zip(names, names[1:]):
        if src == INTERVIEW_NODE:
            continue  # 아래에서 조건부 간선으로 대체한다
        builder.add_edge(src, dst)

    if interactive:
        builder.add_conditional_edges(
            INTERVIEW_NODE,
            _after_interview,
            {INTERVIEW_NODE: INTERVIEW_NODE, "aggregate": "aggregate"},
        )

    builder.add_edge(names[-1], END)

    return builder.compile(checkpointer=checkpointer)

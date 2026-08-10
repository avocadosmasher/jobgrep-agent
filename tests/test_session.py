"""T10 · 세션 계층 검증.

핵심 증명은 하나다 — **같은 `thread_id`로 다시 불러도 그래프가 처음부터 돌지 않는다.**
그래서 테스트 그래프의 노드들은 실행될 때마다 카운터를 올린다. 카운터가 증거다.

실제 분석 그래프는 LLM을 타므로 쓰지 않는다. 여기서 검증하는 것은 세션 계층의
분기 로직이지 분석 파이프라인이 아니다.
"""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from graphs.session import (
    THREAD_KEY,
    PendingQuestion,
    RunStatus,
    ThreadPhase,
    build_checkpointer,
    get_or_create_thread,
    inspect_thread,
    resume_or_start,
    thread_config,
)

QUESTION = {"question": "실무 운영 경험이 있나?", "options": ["예", "아니오"]}


class DemoState(TypedDict, total=False):
    seed: str
    answer: str
    trail: list[str]


class Recorder:
    """노드별 실행 횟수. 재실행 여부를 이걸로 판정한다."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {"before": 0, "ask": 0, "after": 0}

    def hit(self, name: str) -> None:
        self.counts[name] += 1


def make_graph(recorder: Recorder, checkpointer, *, with_interrupt: bool = True):
    def before(state: DemoState) -> dict:
        recorder.hit("before")
        return {"trail": (state.get("trail") or []) + ["before"]}

    def ask(state: DemoState) -> dict:
        recorder.hit("ask")
        answer = interrupt(QUESTION) if with_interrupt else "자동"
        return {"answer": answer, "trail": (state.get("trail") or []) + ["ask"]}

    def after(state: DemoState) -> dict:
        recorder.hit("after")
        return {"trail": (state.get("trail") or []) + ["after"]}

    builder = StateGraph(DemoState)
    for name, fn in (("before", before), ("ask", ask), ("after", after)):
        builder.add_node(name, fn)
    builder.add_edge(START, "before")
    builder.add_edge("before", "ask")
    builder.add_edge("ask", "after")
    builder.add_edge("after", END)
    return builder.compile(checkpointer=checkpointer)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def graph(recorder):
    return make_graph(recorder, build_checkpointer(None))


# --- thread_id ---------------------------------------------------------------


def test_thread_id_is_created_once_and_then_fixed():
    session: dict = {}
    first = get_or_create_thread(session)
    assert session[THREAD_KEY] == first
    assert [get_or_create_thread(session) for _ in range(3)] == [first] * 3


def test_separate_sessions_get_separate_threads():
    assert get_or_create_thread({}) != get_or_create_thread({})


def test_blank_thread_id_is_replaced():
    session = {THREAD_KEY: ""}
    assert get_or_create_thread(session)


# --- 국면 판별 ---------------------------------------------------------------


def test_unstarted_thread_is_not_started(graph):
    status = inspect_thread(graph, "새-스레드")
    assert status.phase is ThreadPhase.NOT_STARTED
    assert status.values == {}
    assert status.questions == []


def test_interrupted_thread_reports_the_question(graph):
    status = resume_or_start(graph, "t", {"seed": "x"})

    assert status.phase is ThreadPhase.INTERRUPTED
    assert status.is_interrupted
    assert [q.payload for q in status.questions] == [QUESTION]
    assert all(isinstance(q, PendingQuestion) and q.interrupt_id for q in status.questions)
    assert status.values["trail"] == ["before"]


def test_completed_thread_is_complete(graph, recorder):
    resume_or_start(graph, "t", {"seed": "x"})
    status = resume_or_start(graph, "t", resume="예")

    assert status.phase is ThreadPhase.COMPLETE
    assert status.is_complete
    assert status.questions == []
    assert status.values["answer"] == "예"
    assert status.values["trail"] == ["before", "ask", "after"]


def test_graph_without_interrupt_completes_in_one_call(recorder):
    graph = make_graph(recorder, build_checkpointer(None), with_interrupt=False)
    status = resume_or_start(graph, "t", {"seed": "x"})
    assert status.phase is ThreadPhase.COMPLETE


# --- 완료 조건: 재실행하지 않는다 ---------------------------------------------


def test_resuming_does_not_rerun_earlier_nodes(graph, recorder):
    """중단 지점부터 재개된다 — 앞 노드는 다시 돌지 않는다."""
    resume_or_start(graph, "t", {"seed": "x"})
    assert recorder.counts["before"] == 1

    resume_or_start(graph, "t", resume="예")

    assert recorder.counts["before"] == 1, "재개인데 앞 노드가 다시 돌았다"
    assert recorder.counts["after"] == 1


def test_completed_thread_is_never_rerun(graph, recorder):
    """★ 카드 완료 조건 — 같은 thread_id로 재호출해도 그래프가 다시 돌지 않는다."""
    resume_or_start(graph, "t", {"seed": "x"})
    resume_or_start(graph, "t", resume="예")
    settled = dict(recorder.counts)

    for _ in range(3):
        status = resume_or_start(graph, "t", {"seed": "다시"})
        assert status.phase is ThreadPhase.COMPLETE

    assert recorder.counts == settled, "완료된 스레드가 재실행됐다"
    assert inspect_thread(graph, "t").values["trail"] == ["before", "ask", "after"]


def test_interrupted_thread_without_answer_does_not_advance(graph, recorder):
    """답이 없으면 묻기만 하고 물러난다 — Streamlit rerun마다 재실행되면 안 된다."""
    resume_or_start(graph, "t", {"seed": "x"})
    settled = dict(recorder.counts)

    for _ in range(3):
        status = resume_or_start(graph, "t", {"seed": "또"})
        assert status.phase is ThreadPhase.INTERRUPTED
        assert [q.payload for q in status.questions] == [QUESTION]

    assert recorder.counts == settled, "답이 없는데 그래프가 진행됐다"


def test_inspect_never_executes_the_graph(graph, recorder):
    for _ in range(3):
        inspect_thread(graph, "t")
    assert recorder.counts == {"before": 0, "ask": 0, "after": 0}


def test_start_without_input_does_nothing(graph, recorder):
    status = resume_or_start(graph, "t")
    assert status.phase is ThreadPhase.NOT_STARTED
    assert recorder.counts == {"before": 0, "ask": 0, "after": 0}


def test_threads_do_not_leak_into_each_other(graph, recorder):
    resume_or_start(graph, "a", {"seed": "x"})
    resume_or_start(graph, "a", resume="예")

    assert inspect_thread(graph, "b").phase is ThreadPhase.NOT_STARTED

    resume_or_start(graph, "b", {"seed": "y"})
    assert inspect_thread(graph, "b").phase is ThreadPhase.INTERRUPTED
    assert inspect_thread(graph, "a").phase is ThreadPhase.COMPLETE


# --- 영속 (SqliteSaver) -------------------------------------------------------


def test_sqlite_checkpointer_survives_a_new_graph_instance(tmp_path):
    """프로세스 재시작 흉내 — 같은 db를 가리키는 새 그래프가 중단 지점을 이어받는다."""
    db = tmp_path / "ck.sqlite"

    first = Recorder()
    resume_or_start(make_graph(first, build_checkpointer(db)), "t", {"seed": "x"})
    assert first.counts["before"] == 1

    # 새 Recorder·새 그래프 객체 = 프로세스가 다시 뜬 상황
    second = Recorder()
    reborn = make_graph(second, build_checkpointer(db))

    assert inspect_thread(reborn, "t").phase is ThreadPhase.INTERRUPTED

    status = resume_or_start(reborn, "t", resume="예")
    assert status.phase is ThreadPhase.COMPLETE
    assert status.values["trail"] == ["before", "ask", "after"]
    assert second.counts["before"] == 0, "재시작 후 앞 노드가 다시 돌았다"


def test_build_checkpointer_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "dir" / "ck.sqlite"
    build_checkpointer(db)
    assert db.parent.is_dir()


# --- 반환 계약 ---------------------------------------------------------------


def test_run_status_is_immutable(graph):
    status = resume_or_start(graph, "t", {"seed": "x"})
    assert isinstance(status, RunStatus)
    with pytest.raises(Exception):
        status.phase = ThreadPhase.COMPLETE  # type: ignore[misc]


def test_thread_config_shape():
    assert thread_config("abc") == {"configurable": {"thread_id": "abc"}}

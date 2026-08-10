"""Streamlit ↔ LangGraph 사이의 얇은 세션 계층.

두 실행 모델이 정면으로 충돌한다 — Streamlit은 위젯을 건드릴 때마다 스크립트를
**통째로 재실행**하고, LangGraph는 중단 지점에서 멈췄다가 같은 스레드로 **재개**해야
한다. 순진하게 `invoke`를 부르면 사용자가 답변할 때마다 그래프가 처음부터 다시 돈다.

**이 계층의 유일한 규칙: 실행 전에 반드시 상태를 조회한다.**
`resume_or_start()` 밖에서 `graph.invoke()`를 직접 부르지 않는다.

langgraph 1.2.10 실측 기준 (DEVLOG D27):
    미시작 → `created_at is None`
    중단됨 → `snapshot.interrupts`가 비어 있지 않음 (`next`도 비어 있지 않음)
    완료   → 위 둘 다 아님
`next == ()`는 미시작과 완료 **양쪽**에서 나오므로 단독으로는 판별 근거가 못 된다.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, MutableMapping

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

THREAD_KEY = "jobprep_thread_id"
DEFAULT_DB = Path(".jobprep") / "checkpoints.sqlite"

# 진행 표시(T13)가 소비하는 스트림 모드.
#   "updates" — 노드 하나가 끝날 때마다 `{노드명: 부분갱신}`. 중단은 `__interrupt__` 키.
#   "custom"  — 노드 **안에서** `get_stream_writer()`로 흘려보낸 값. 수집 ReAct(T18)의
#               tool call을 노드가 끝나기 전에 보여주는 통로다.
# 모드를 둘 이상 주면 청크가 `(mode, payload)` 튜플로 온다 (langgraph 1.2.10 실측).
STREAM_MODES = ["updates", "custom"]


class ThreadPhase(str, Enum):
    NOT_STARTED = "미시작"
    INTERRUPTED = "중단됨"
    COMPLETE = "완료"


@dataclass(frozen=True)
class PendingQuestion:
    """중단 지점이 UI에 요구하는 것. `interrupt()`에 넘긴 페이로드 그대로다."""

    interrupt_id: str
    payload: Any


@dataclass(frozen=True)
class RunStatus:
    phase: ThreadPhase
    thread_id: str
    values: dict = field(default_factory=dict)
    questions: list[PendingQuestion] = field(default_factory=list)

    @property
    def is_interrupted(self) -> bool:
        return self.phase is ThreadPhase.INTERRUPTED

    @property
    def is_complete(self) -> bool:
        return self.phase is ThreadPhase.COMPLETE


# --- checkpointer -------------------------------------------------------------


def build_checkpointer(db_path: str | Path | None = DEFAULT_DB):
    """체크포인터를 만든다.

    `SqliteSaver`가 기본이다 — 프로세스가 재시작해도 진행 중인 스레드가 살아남는다.
    `db_path=None`이면 `InMemorySaver`(테스트·일회성 실행용).

    커넥션은 `check_same_thread=False`로 연다. Streamlit이 스크립트를 워커 스레드에서
    재실행하므로 이 옵션이 없으면 두 번째 rerun에서 sqlite3가 예외를 던진다.
    """
    if db_path is None:
        return InMemorySaver()

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    saver.setup()
    return saver


# --- 스레드 ------------------------------------------------------------------


def get_or_create_thread(session_state: MutableMapping[str, Any]) -> str:
    """세션당 `thread_id`를 **최초 1회만** 만들고 이후 고정해서 돌려준다.

    `st.session_state`를 그대로 받되 그 타입에 의존하지 않는다 — dict면 된다.
    매번 새 id를 만들면 체크포인트가 매 rerun마다 버려져 이 계층이 무의미해진다.
    """
    thread_id = session_state.get(THREAD_KEY)
    if not thread_id:
        thread_id = str(uuid.uuid4())
        session_state[THREAD_KEY] = thread_id
    return thread_id


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def inspect_thread(graph, thread_id: str) -> RunStatus:
    """실행하지 않고 스레드의 현재 국면만 조회한다."""
    snapshot = graph.get_state(thread_config(thread_id))

    if snapshot.created_at is None:
        return RunStatus(ThreadPhase.NOT_STARTED, thread_id)

    if snapshot.interrupts:
        return RunStatus(
            ThreadPhase.INTERRUPTED,
            thread_id,
            values=dict(snapshot.values),
            questions=[
                PendingQuestion(interrupt_id=i.id, payload=i.value)
                for i in snapshot.interrupts
            ],
        )

    return RunStatus(ThreadPhase.COMPLETE, thread_id, values=dict(snapshot.values))


# --- 실행 --------------------------------------------------------------------


def _run(graph, payload: Any, config: dict, on_event: Callable[[Any], None] | None) -> None:
    """그래프를 한 번 돌린다. **실행이 실제로 일어나는 곳은 여기 하나뿐이다.**

    `on_event`가 없으면 `invoke` — T10·T12가 고정한 기존 경로 그대로다.
    `on_event`가 있으면 `stream`으로 바꿔 청크를 그대로 넘긴다. 진행 표시(T13)가
    실행 경계 밖에서 `graph.stream()`을 직접 부르지 않게 하려는 것이다. 이 계층의
    유일한 규칙("실행 전에 반드시 상태를 조회한다")은 스트리밍에도 그대로 걸려야
    하는데, `app/`에서 스트림을 직접 열면 국면 조회를 건너뛰는 경로가 하나 더
    생긴다 — 그 순간 완료된 스레드가 rerun마다 처음부터 다시 도는 결함(D27)이
    조용히 되살아난다.

    `stream`은 `invoke`와 같은 실행이다. 반환값을 버리고 체크포인터에 남은 상태를
    다시 조회하므로(아래 `resume_or_start`) 두 경로의 결과가 갈리지 않는다.
    """
    if on_event is None:
        graph.invoke(payload, config)
        return

    for chunk in graph.stream(payload, config, stream_mode=STREAM_MODES):
        on_event(chunk)


def resume_or_start(
    graph,
    thread_id: str,
    initial_input: dict | None = None,
    *,
    resume: Any = None,
    on_event: Callable[[Any], None] | None = None,
) -> RunStatus:
    """스레드 상태를 **먼저 조회하고** 그에 맞는 실행만 한다.

    | 현재 국면 | resume 있음 | 동작 |
    | --- | --- | --- |
    | 미시작 | - | `initial_input`으로 처음 실행 |
    | 중단됨 | O | `Command(resume=)`로 중단 지점부터 재개 |
    | 중단됨 | X | **실행하지 않고** 질문을 그대로 돌려줌 |
    | 완료 | - | **실행하지 않음** — 재실행하면 결과를 덮어쓴다 |

    완료된 스레드를 다시 돌리고 싶으면 새 `thread_id`를 쓴다. 여기서 조용히
    재실행해 주면 사용자가 다운로드 버튼을 누를 때마다 분석이 새로 돌아간다.

    `on_event`를 주면 노드 이벤트가 실시간으로 콜백에 흐른다(T13 진행 표시).
    **표는 그대로다** — 실행하지 않기로 한 국면에서는 이벤트도 흐르지 않는다.
    """
    status = inspect_thread(graph, thread_id)
    config = thread_config(thread_id)

    if status.phase is ThreadPhase.NOT_STARTED:
        if initial_input is None:
            return status
        _run(graph, initial_input, config, on_event)

    elif status.phase is ThreadPhase.INTERRUPTED:
        if resume is None:
            return status  # 답이 없으면 묻기만 하고 물러난다
        _run(graph, Command(resume=resume), config, on_event)

    else:  # COMPLETE — 손대지 않는다
        return status

    return inspect_thread(graph, thread_id)

"""T13 · 노드 이벤트 스트림 → 진행 표시.

`graph.stream()`이 흘리는 청크를 받아 **한 줄짜리 진행 상태**로 접는다. 화면은
설계도 §9-4의 그림 그대로다:

```
회사 분석 진행 중
  ✅ JD 본문 수집 (1건)
  ✅ 요구 역량 추출 (17개)
  🔄 판정 기준 분해
  ⏸ 델타 인터뷰 (3건 질문 대기)   ← 여기서 멈춘다
  ⬜ 3트랙 전략 브리프 작성
```

**이 모듈이 지키는 것 셋.**

1. **내부 함수명을 화면에 내보내지 않는다.** 문구는 전부 `presets/node_labels.yaml`
   에서 온다. 매핑에 없는 노드는 이름이 아니라 `fallback`("처리 중")으로 떨어진다 —
   모르는 노드가 와도 내부 이름이 새지 않는다.

2. **이벤트 소스는 하나뿐이다.** T28(과정 지표)은 계측을 새로 심지 말고
   `add_listener()`로 이 스트림에 붙는다. 두 군데서 재면 두 숫자가 어긋난다
   (T28 불변식). 그래서 `ProgressEvent`는 화면에 필요 없는 `seconds`·`payload`
   까지 들고 있다 — 응답 시간·도구 호출 횟수가 여기서 그대로 나온다.

3. **수집 ReAct(T18)의 내부 tool call은 서브 라인으로 흐른다.** 노드 안에서
   `get_stream_writer()({"tool": ..., "ok": ...})`를 부르면 그 노드 라인 밑에
   붙는다. 실측상 custom 청크는 **노드가 끝나기 전에** 도착해서(langgraph 1.2.10)
   "지금 일하는 중"이 실시간으로 보인다. T18은 이 모듈을 고칠 필요가 없다.

**청크 모양 (langgraph 1.2.10 실측).**
`stream_mode=["updates", "custom"]`이면 `(mode, payload)` **튜플**로 오고, 단일
모드면 payload만 온다. 둘 다 받는다.

    ('custom',  {'tool': 'fetch_jd_body', 'ok': True})     ← 노드 실행 중
    ('updates', {'extract': {'required': [...]}})          ← 노드 완료
    ('updates', {'__interrupt__': (Interrupt(value=…),)})  ← 중단
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

PRESET_PATH = Path(__file__).resolve().parent.parent / "presets" / "node_labels.yaml"

# 스트림 청크의 약속된 키들.
UPDATES = "updates"
CUSTOM = "custom"
INTERRUPT_KEY = "__interrupt__"

# 이보다 짧게 걸린 노드는 시간을 적지 않는다 — "0.0s"가 줄줄이 붙으면 읽기만 나빠진다.
MIN_REPORTED_SECONDS = 0.05


class LineState(str, Enum):
    """한 노드의 표시 상태. 값이 그대로 화면 아이콘이다."""

    PENDING = "⬜"
    RUNNING = "🔄"
    DONE = "✅"
    PAUSED = "⏸"
    FAILED = "⚠️"


class EventKind(str, Enum):
    NODE_DONE = "node_done"
    TOOL_CALL = "tool_call"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


# --- 프리셋 -------------------------------------------------------------------


@dataclass(frozen=True)
class CountSpec:
    """부분 갱신에서 뽑아 괄호로 붙일 숫자 하나."""

    key: str
    unit: str
    hide_zero: bool = False


@dataclass(frozen=True)
class NodeLabel:
    label: str
    counts: tuple[CountSpec, ...] = ()
    repeats_while: str | None = None


@dataclass(frozen=True)
class LabelSet:
    title: str
    paused_title: str
    complete_title: str
    failed_title: str
    fallback: str
    nodes: Mapping[str, NodeLabel]
    tools: Mapping[str, str]

    def node(self, name: str) -> NodeLabel:
        """모르는 노드는 **내부 이름 대신** fallback 문구로 떨어진다 (카드 불변식)."""
        return self.nodes.get(name) or NodeLabel(self.fallback)

    def tool(self, name: str) -> str:
        return self.tools.get(name) or self.fallback


def _to_node_label(raw: Mapping[str, Any], fallback: str) -> NodeLabel:
    counts = tuple(
        CountSpec(
            key=str(spec["key"]),
            unit=str(spec.get("unit", "")),
            hide_zero=bool(spec.get("hide_zero", False)),
        )
        for spec in (raw.get("counts") or [])
        if isinstance(spec, Mapping) and spec.get("key")
    )
    repeats = raw.get("repeats_while")
    return NodeLabel(
        label=str(raw.get("label") or fallback),
        counts=counts,
        repeats_while=str(repeats) if repeats else None,
    )


@lru_cache(maxsize=4)
def load_labels(path: str | Path = PRESET_PATH) -> LabelSet:
    """`presets/node_labels.yaml`을 읽는다. 파일이 없어도 진행 표시는 동작한다."""
    data: Mapping[str, Any] = {}
    file = Path(path)
    if file.exists():
        data = yaml.safe_load(file.read_text("utf-8")) or {}

    fallback = str(data.get("fallback") or "처리 중")
    nodes = {
        name: _to_node_label(raw if isinstance(raw, Mapping) else {}, fallback)
        for name, raw in (data.get("nodes") or {}).items()
    }
    tools = {str(k): str(v) for k, v in (data.get("tools") or {}).items()}
    title = str(data.get("title") or "진행 중")
    return LabelSet(
        title=title,
        paused_title=str(data.get("paused_title") or title),
        complete_title=str(data.get("complete_title") or title),
        failed_title=str(data.get("failed_title") or title),
        fallback=fallback,
        nodes=nodes,
        tools=tools,
    )


# --- 숫자 뽑기 ----------------------------------------------------------------


def count_of(value: Any) -> int:
    """상태 값 하나를 "몇 개인가"로 접는다.

    `criteria`처럼 `dict[str, list]`인 칸은 안쪽 목록 길이를 전부 더한다 — 역량
    3개에 기준 12개면 사용자에게 의미 있는 숫자는 12다.
    """
    if value is None:
        return 0
    if isinstance(value, Mapping):
        values = list(value.values())
        if values and all(isinstance(v, (list, tuple)) for v in values):
            return sum(len(v) for v in values)
        return len(value)
    if isinstance(value, (str, bytes)):
        return 1
    if isinstance(value, Sequence):
        return len(value)
    return 1


def detail_for(update: Mapping[str, Any], counts: Iterable[CountSpec]) -> str:
    """부분 갱신 dict를 "17개" / "12건 판정 · 3건 확인 필요" 같은 꼬리표로 만든다."""
    parts = []
    for spec in counts:
        if spec.key not in update:
            continue
        number = count_of(update[spec.key])
        if number == 0 and spec.hide_zero:
            continue
        parts.append(f"{number}{spec.unit}")
    return " · ".join(parts)


# --- 라인과 이벤트 -------------------------------------------------------------


@dataclass
class Line:
    """진행 표시 한 줄 = 노드 하나."""

    node: str
    label: str
    state: LineState = LineState.PENDING
    detail: str = ""
    seconds: float = 0.0
    substeps: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        out = f"{self.state.value} {self.label}"
        if self.detail:
            out += f" ({self.detail})"
        if self.state is LineState.DONE and self.seconds >= MIN_REPORTED_SECONDS:
            out += f" · {self.seconds:.1f}s"
        return out


@dataclass(frozen=True)
class ProgressEvent:
    """화면과 지표가 **함께 쓰는** 이벤트 (T28 불변식 — 중복 계측 금지).

    `label`까지 실어 보내는 이유는 T28이 다시 프리셋을 읽지 않게 하기 위해서다.
    `payload`는 원본 그대로라 지표 쪽이 필요한 것을 더 꺼내 쓸 수 있다.
    """

    kind: EventKind
    node: str
    label: str
    detail: str = ""
    seconds: float = 0.0
    payload: Any = None


Listener = Callable[[ProgressEvent], None]


# --- 트래커 -------------------------------------------------------------------


class ProgressTracker:
    """스트림 청크를 라인 상태로 접는 상태기계.

    **세션에 살려 둔다.** 재개(`Command(resume=)`)는 이미 끝난 노드를 다시 흘리지
    않으므로(D27 — 상류 노드는 재실행되지 않는다), 실행마다 트래커를 새로 만들면
    앞 라운드에서 ✅였던 줄이 ⬜로 되돌아간다.
    """

    def __init__(
        self,
        nodes: Iterable[str],
        labels: LabelSet | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.labels = labels or load_labels()
        self._clock = clock
        self.lines: list[Line] = [
            Line(node=node, label=self.labels.node(node).label) for node in nodes
        ]
        self.events: list[ProgressEvent] = []
        self._listeners: list[Listener] = []
        self._mark = clock()
        self._wake_next()

    # -- 공개 API --

    def add_listener(self, listener: Listener) -> None:
        """T28은 여기에 붙는다 — 별도 계측을 새로 심지 않는다."""
        self._listeners.append(listener)

    def begin(self) -> None:
        """실행 직전에 시계를 맞춘다.

        중단된 스레드는 사용자가 답을 쓰는 동안 몇 분이고 서 있다. 이걸 안 하면
        그 대기 시간이 재개 후 첫 노드의 소요 시간으로 잘못 붙는다.
        """
        self._mark = self._clock()
        self._wake_next()

    def handle(self, chunk: Any) -> ProgressEvent | None:
        """스트림 청크 하나를 소비한다. 화면을 다시 그릴 일이 없으면 `None`."""
        mode, payload = self._split(chunk)

        if mode == CUSTOM:
            return self._on_tool(payload)
        if not isinstance(payload, Mapping):
            return None
        if INTERRUPT_KEY in payload:
            return self._on_interrupt(payload[INTERRUPT_KEY])

        for node, update in payload.items():
            return self._on_node(str(node), update if isinstance(update, Mapping) else {})
        return None

    def fail(self, reason: str) -> ProgressEvent:
        """실행이 예외로 끊겼을 때 서 있던 줄에 ⚠️를 남긴다.

        표시를 안 하면 마지막 줄이 🔄인 채로 굳어 "아직 도는 중"처럼 보인다.
        """
        line = self._current() or (self.lines[-1] if self.lines else None)
        if line is not None:
            line.state = LineState.FAILED
            line.detail = reason
        return self._emit(EventKind.FAILED, line, payload=reason)

    @property
    def is_paused(self) -> bool:
        return any(line.state is LineState.PAUSED for line in self.lines)

    @property
    def is_complete(self) -> bool:
        return bool(self.lines) and all(line.state is LineState.DONE for line in self.lines)

    @property
    def status_state(self) -> str:
        """`st.status(state=…)`에 그대로 넘길 값."""
        if any(line.state is LineState.FAILED for line in self.lines):
            return "error"
        if self.is_complete:
            return "complete"
        return "running"

    @property
    def status_label(self) -> str:
        """국면별 제목. 문구는 전부 프리셋에서 온다 — 여기서 조립하지 않는다."""
        if any(line.state is LineState.FAILED for line in self.lines):
            return self.labels.failed_title
        if self.is_paused:
            return self.labels.paused_title
        if self.is_complete:
            return self.labels.complete_title
        return self.labels.title

    def as_markdown(self) -> str:
        rows: list[str] = []
        for line in self.lines:
            rows.append(f"- {line.text}")
            rows.extend(f"    - {sub}" for sub in line.substeps)
        return "\n".join(rows)

    # -- 내부 --

    @staticmethod
    def _split(chunk: Any) -> tuple[str, Any]:
        """다중 모드는 `(mode, payload)`, 단일 모드는 payload만 온다."""
        if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
            return chunk[0], chunk[1]
        return UPDATES, chunk

    def _line(self, node: str) -> Line | None:
        return next((line for line in self.lines if line.node == node), None)

    def _current(self) -> Line | None:
        """지금 서 있는 줄 = 아직 끝나지 않은 첫 줄."""
        return next((line for line in self.lines if line.state is not LineState.DONE), None)

    def _wake_next(self) -> None:
        line = self._current()
        if line is not None and line.state is LineState.PENDING:
            line.state = LineState.RUNNING

    def _lap(self) -> float:
        now = self._clock()
        elapsed = max(0.0, now - self._mark)
        self._mark = now
        return elapsed

    def _emit(self, kind: EventKind, line: Line | None, payload: Any = None) -> ProgressEvent:
        event = ProgressEvent(
            kind=kind,
            node=line.node if line else "",
            label=line.label if line else self.labels.fallback,
            detail=line.detail if line else "",
            seconds=line.seconds if line else 0.0,
            payload=payload,
        )
        self.events.append(event)
        for listener in self._listeners:
            listener(event)
        return event

    def _on_node(self, node: str, update: Mapping[str, Any]) -> ProgressEvent | None:
        line = self._line(node)
        if line is None:
            # 배선에 없는 노드. 줄을 새로 만들지 않는다 — 화면 순서가 흐트러지고,
            # 애초에 표시할 자리는 `node_sequence()`가 정하는 게 정본이다.
            return None

        spec = self.labels.node(node)
        line.seconds += self._lap()
        detail = detail_for(update, spec.counts)
        if detail:
            line.detail = detail

        # 자기 루프 노드는 아직 끝난 게 아니다 — ✅로 조기 승격시키면 다음 라운드에
        # 다시 물을 때 화면이 뒤로 돌아간다(델타 인터뷰, T11).
        repeating = bool(spec.repeats_while) and bool(update.get(spec.repeats_while))
        line.state = LineState.RUNNING if repeating else LineState.DONE
        if not repeating:
            self._wake_next()

        return self._emit(EventKind.NODE_DONE, line, payload=dict(update))

    def _on_interrupt(self, interrupts: Any) -> ProgressEvent | None:
        """중단 — 지금 서 있는 줄을 ⏸로 세운다.

        어느 노드가 멈췄는지는 청크에 안 실린다. 하지만 중단 시점에 "끝나지 않은
        첫 줄"이 곧 그 노드다. 자기 루프 노드가 `repeats_while`로 RUNNING을 유지하는
        덕에 2라운드째에도 이 규칙이 그대로 맞는다.
        """
        line = self._current()
        if line is None:
            return None

        line.state = LineState.PAUSED
        payload = self._first_interrupt_value(interrupts)
        asked = self._question_count(payload)
        if asked:
            line.detail = f"{asked}건 질문 대기"

        return self._emit(EventKind.INTERRUPTED, line, payload=payload)

    @staticmethod
    def _first_interrupt_value(interrupts: Any) -> Any:
        if isinstance(interrupts, (list, tuple)) and interrupts:
            return getattr(interrupts[0], "value", interrupts[0])
        return getattr(interrupts, "value", interrupts)

    @staticmethod
    def _question_count(payload: Any) -> int:
        if isinstance(payload, Mapping):
            return count_of(payload.get("questions"))
        return 0

    def _on_tool(self, payload: Any) -> ProgressEvent | None:
        """수집 ReAct(T18)의 tool call을 현재 줄의 서브 라인으로 붙인다."""
        if isinstance(payload, Mapping):
            name = str(payload.get("tool") or payload.get("name") or "")
            label = str(payload.get("label") or self.labels.tool(name))
            detail = str(payload.get("detail") or "")
            ok = payload.get("ok", True)
        else:
            name, label, detail, ok = "", str(payload), "", True

        if not label:
            return None

        line = self._current()
        if line is None:
            return None

        icon = "↳" if ok else "⚠️"
        text = f"{icon} {label}" + (f" — {detail}" if detail else "")
        line.substeps.append(text)

        return self._emit(EventKind.TOOL_CALL, line, payload=payload)


def tracker_for(nodes: Iterable[str]) -> ProgressTracker:
    return ProgressTracker(nodes)


# --- Streamlit 결합 -----------------------------------------------------------
#
# 위쪽은 Streamlit을 모른다 — 트래커만 따로 테스트할 수 있어야 하기 때문이다.
# 아래 두 함수만 화면을 안다.


def panel_slot():
    """진행 표시가 들어갈 자리를 하나 잡아 둔다.

    **`st.empty()`인 것이 핵심이다.** 이 자리는 원소를 하나만 들고 있어서 다시
    쓰면 앞의 것이 지워진다. 그냥 `st` 위에 그리면 rerun 한 번에 패널이 둘 그려진다 —
    답변 제출 직후처럼 "먼저 지난 진행을 그리고, 이어서 실행 패널을 여는" 국면에서
    묵은 ⏸ 패널과 도는 중인 패널이 나란히 남는다.
    """
    import streamlit as st

    return st.empty()


def render_panel(tracker: ProgressTracker, *, slot: Any = None, expanded: bool | None = None):
    """지금까지의 진행을 그린다. 실행 중이 아닐 때(rerun 후)도 같은 그림이 남는다.

    Streamlit은 rerun마다 화면을 처음부터 다시 그리므로, 트래커를 세션에 들고
    있지 않으면 중단 국면에서 ⏸가 사라진다.
    """
    import streamlit as st

    if expanded is None:
        expanded = not tracker.is_complete

    target = slot if slot is not None else st
    box = target.status(tracker.status_label, state=tracker.status_state, expanded=expanded)
    box.markdown(tracker.as_markdown())
    return box


def live_status(tracker: ProgressTracker, *, slot: Any = None) -> tuple[Any, Callable[[Any], None]]:
    """실행용 패널을 열고 `(컨테이너, on_event)`를 돌려준다.

    `on_event`는 그대로 `resume_or_start(..., on_event=)`에 넘긴다. 청크가 하나
    올 때마다 같은 자리를 다시 그리므로 줄이 쌓이지 않고 아이콘만 바뀐다.
    """
    import streamlit as st

    target = slot if slot is not None else st
    box = target.status(tracker.status_label, state="running", expanded=True)
    body = box.empty()
    body.markdown(tracker.as_markdown())

    def on_event(chunk: Any) -> None:
        if tracker.handle(chunk) is not None:
            body.markdown(tracker.as_markdown())

    return box, on_event


def close_status(
    box: Any,
    tracker: ProgressTracker,
    *,
    on_complete: Callable[[ProgressTracker], None] | None = None,
) -> None:
    """실행이 끝난 뒤 컨테이너 머리말을 최종 국면으로 맞춘다.

    `on_complete`는 이번 호출로 실행이 **실제로 끝났을 때만** 불린다(T28의 지표
    기록 지점). 기본값 `None`이면 기존 동작 그대로다 — 호출부(`app/main.py`)가
    이 인자를 안 주면 아무 일도 늘지 않는다. 중단 국면(`tracker.is_complete`가
    아직 `False`)에서는 부르지 않는다 — 재개마다 부르면 지표가 실행 횟수가 아니라
    rerun 횟수로 늘어난다.
    """
    box.update(
        label=tracker.status_label,
        state=tracker.status_state,
        expanded=not tracker.is_complete,
    )
    if on_complete is not None and tracker.is_complete:
        on_complete(tracker)

"""T28 · 과정 지표 로깅 (설계도 §13 "자동 과정 지표").

루브릭(T27)이 **결과** 품질을 재면 이건 **과정** 품질을 잰다. 발표에서 "됩니다"가
아니라 숫자로 말할 수 있게 하는 자리다.

**계측을 새로 심지 않는다.** `app/progress.py`의 불변식(T13) 그대로 — 진행 표시가
흘리는 `ProgressEvent` 하나가 화면과 지표의 공통 소스다. 노드 소요 시간(`seconds`)도
도구 호출 성공 여부(`payload["ok"]`)도 이미 그 스트림에 실려 있으므로, 이 모듈이
하는 일은 **그 스트림을 숫자 몇 개로 접는 것**뿐이다(`collect_from_events`는 별도
상태 조회 없이 이벤트 목록 하나만 받는다).

지표 → 이벤트 매핑
------------------
| 지표(§13) | 이벤트 |
| --- | --- |
| 응답 시간 | `NODE_DONE.seconds`의 합 |
| 도구 호출 횟수 · 수집 성공률 | `TOOL_CALL` 이벤트 개수 · `payload["ok"]` 비율 |
| 근거 첨부율 | `verdicts`를 실은 **마지막** `NODE_DONE.payload` |
| 최소 요건 충족률 | `quality_gate`/`profile_gate` 노드의 `payload["gate_status"]` |

**근거 첨부율의 분모는 `verify` 노드가 아니라 "마지막 판정 목록"이다.**
`delta_interview`가 재판정으로 목록을 통째로 갈아끼우므로(T11 `merge_verdicts`),
`verify`만 보면 인터뷰로 해소된 판정이 지표에 안 잡힌다. 두 노드 다 **전체 목록**을
반환하니 마지막 것 하나면 된다.

분모에서 `EVIDENCE_OPTIONAL_STATES`(tools/verify.py — 근거 없이도 인정되는 판정,
D10)를 뺀다. 여기서 다시 정의하지 않고 그대로 가져다 쓴다 — 두 벌이 되면 한쪽만
고쳤을 때 조용히 어긋난다.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Iterable, Sequence
from uuid import uuid4

from pydantic import BaseModel, Field

from app.progress import EventKind, ProgressEvent, ProgressTracker
from tools.verify import EVIDENCE_OPTIONAL_STATES

DEFAULT_LOG = Path(".jobprep") / "metrics.jsonl"

# 게이트 판정을 싣는 노드 이름 — UC-2(`quality_gate`)·UC-1(`profile_gate`) 둘 다.
# 정본은 각 그래프 배선(`graphs/analysis_graph.py`·`graphs/profile_graph.py`)이지만
# 그걸 import하면 이 모듈이 화면 계층 위에 올라앉게 돼 순환 import 위험이 생긴다.
# 이벤트에는 노드 **이름**만 실려 오므로 문자열 상수로 충분하다 — 이름이 바뀌면
# 이 튜플만 고치면 된다.
# 게이트 노드 ↔ 그 노드가 도는 모드. 어휘는 `GraphState.mode`("analysis"|"profile")
# 그대로 쓴다 — 여기서 새로 지으면 상태와 지표가 다른 말을 하게 된다.
#
# **모드를 기록하는 이유** — 두 그래프가 로그 하나를 공유하는데 `hard_gate_passed`의
# 뜻이 서로 다르다(UC-2는 "JD를 건졌다", UC-1은 "프로필이 완성됐다"). 갈라 적지
# 않으면 T25가 고쳤던 "표기가 판정과 어긋나는" 자리가 지표 쪽에 다시 생긴다.
MODE_BY_GATE_NODE: dict[str, str] = {
    "quality_gate": "analysis",
    "profile_gate": "profile",
}
GATE_NODES = tuple(MODE_BY_GATE_NODE)

# 판정 목록이 실려 오는 상태 칸. 어느 노드가 채우든 이 키가 있으면 그게 정본이다.
VERDICTS_KEY = "verdicts"
GATE_STATUS_KEY = "gate_status"


class RunMetrics(BaseModel):
    """실행 1회의 지표 레코드. `append_record()`가 그대로 한 줄(JSONL)로 남긴다."""

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    mode: str | None = None                # "analysis" | "profile" — 게이트 노드에서 유도
    seconds: float = 0.0
    tool_calls: int = 0
    tool_calls_ok: int = 0
    verdicts_scored: int = 0        # 근거가 필요한 판정 — EVIDENCE_OPTIONAL_STATES 제외
    verdicts_with_evidence: int = 0
    hard_gate_passed: bool | None = None   # 게이트 노드가 안 돌았으면 None

    @property
    def collection_success_rate(self) -> float | None:
        """수집 성공률(§13 목표 ≥90%). 도구 호출이 없었으면 잴 게 없다(`None`)."""
        return self.tool_calls_ok / self.tool_calls if self.tool_calls else None

    @property
    def evidence_attachment_rate(self) -> float | None:
        """근거 첨부율(§13 목표 100%). 근거가 필요한 판정이 없었으면 `None`."""
        return (
            self.verdicts_with_evidence / self.verdicts_scored
            if self.verdicts_scored
            else None
        )


def collect_from_events(events: Sequence[ProgressEvent]) -> RunMetrics:
    """완료된 실행의 이벤트 목록을 지표 레코드 하나로 접는다 — 순수 함수, LLM 없음."""
    seconds = 0.0
    tool_calls = 0
    tool_calls_ok = 0
    hard_gate_passed: bool | None = None
    mode: str | None = None
    verdicts: list = []

    for event in events:
        payload = event.payload if isinstance(event.payload, dict) else {}

        if event.kind is EventKind.NODE_DONE:
            seconds += event.seconds

            if event.node in GATE_NODES and payload.get(GATE_STATUS_KEY) is not None:
                hard_gate_passed = bool(payload[GATE_STATUS_KEY].get("hard_gate_passed"))
                mode = MODE_BY_GATE_NODE[event.node]

            if VERDICTS_KEY in payload:
                verdicts = list(payload[VERDICTS_KEY] or [])

        elif event.kind is EventKind.TOOL_CALL:
            tool_calls += 1
            # `ok`가 없으면 성공으로 본다 — T13의 서브 라인 규약과 같은 기본값이다
            # (`_on_tool`이 `payload.get("ok", True)`로 아이콘을 고른다). 화면은
            # ↳인데 지표만 실패로 세면 두 숫자가 어긋난다.
            if payload.get("ok", True):
                tool_calls_ok += 1

    scored = [v for v in verdicts if v.state not in EVIDENCE_OPTIONAL_STATES]

    return RunMetrics(
        mode=mode,
        seconds=seconds,
        tool_calls=tool_calls,
        tool_calls_ok=tool_calls_ok,
        verdicts_scored=len(scored),
        verdicts_with_evidence=sum(1 for v in scored if v.evidence),
        hard_gate_passed=hard_gate_passed,
    )


# --- 파일 I/O ------------------------------------------------------------------


def _log_path(path: Path | str | None) -> Path:
    """기본 경로를 **호출 시점에** 정한다.

    `path=DEFAULT_LOG`처럼 기본 인자로 박으면 정의 시점에 묶여서 테스트가
    `monkeypatch.setattr(obs.metrics, "DEFAULT_LOG", …)`로 갈아끼워도 안 먹는다 —
    그 함정에 이미 한 번 빠진 적이 있다(D71, 주입점 규약).
    """
    return Path(path) if path is not None else Path(DEFAULT_LOG)


def append_record(record: RunMetrics, *, path: Path | str | None = None) -> None:
    """지표 레코드 한 줄을 JSONL로 이어붙인다. 실행 1회 = 파일 1줄(카드 완료 조건)."""
    file = _log_path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def read_records(path: Path | str | None = None) -> list[RunMetrics]:
    """지금까지 쌓인 지표 레코드를 전부 읽는다. 파일이 없으면 빈 목록."""
    file = _log_path(path)
    if not file.exists():
        return []
    return [
        RunMetrics.model_validate_json(line)
        for line in file.read_text("utf-8").splitlines()
        if line.strip()
    ]


def record_run(tracker: ProgressTracker, *, path: Path | str | None = None) -> RunMetrics:
    """완료된 트래커에서 지표를 뽑아 파일에 한 줄 남긴다.

    `app/progress.py`의 `close_status(..., on_complete=record_run)`가 부르는
    자리다 — 실행이 실제로 끝났을 때만 불리도록 그쪽이 이미 걸러 준다(T28
    불변식: 계측은 스트림 하나, 호출도 한 곳).
    """
    record = collect_from_events(tracker.events)
    append_record(record, path=path)
    return record


# --- 발표용 집계 -----------------------------------------------------------------
#
# "도구 호출 횟수 중앙값"은 실행 1회로는 낼 수 없는 숫자다 — 여러 레코드를 모아야
# 뜻이 생긴다. 마찬가지로 성공률·충족률도 실행 여러 건을 합쳐야 §13의 목표
# (≥90% 등)와 비교할 수 있다.


class MetricsSummary(BaseModel):
    """여러 실행의 지표를 §13 "목표" 표와 같은 모양으로 접는다."""

    runs: int
    median_tool_calls: float = 0.0
    collection_success_rate: float | None = None
    evidence_attachment_rate: float | None = None
    hard_gate_pass_rate: float | None = None


def _rate(numerators: Iterable[int], denominators: Iterable[int]) -> float | None:
    total = sum(denominators)
    return sum(numerators) / total if total else None


def summarize(records: Iterable[RunMetrics]) -> MetricsSummary:
    records = list(records)
    if not records:
        return MetricsSummary(runs=0)

    gated = [r.hard_gate_passed for r in records if r.hard_gate_passed is not None]

    return MetricsSummary(
        runs=len(records),
        median_tool_calls=median(r.tool_calls for r in records),
        collection_success_rate=_rate(
            (r.tool_calls_ok for r in records), (r.tool_calls for r in records)
        ),
        evidence_attachment_rate=_rate(
            (r.verdicts_with_evidence for r in records), (r.verdicts_scored for r in records)
        ),
        hard_gate_pass_rate=(sum(gated) / len(gated)) if gated else None,
    )

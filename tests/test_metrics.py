"""T28 · 과정 지표 로깅.

카드의 불변식이 하나뿐이라 테스트의 중심도 하나다 — **T13 스트림과 같은 소스에서
수집한다. 중복 계측 금지.** 그래서 여기서 합성 이벤트만 검사하면 부족하다. 합성
이벤트는 "내가 만든 dict를 내가 잘 읽는가"만 증명하고, 진짜 그래프가 그 모양을
흘리는지는 증명하지 못한다. 층은 넷이다:

    ① 접기   — 이벤트 목록 → `RunMetrics` (순수 함수)
    ② 파일   — 실행 1회 = 파일 1줄 (카드 완료 조건)
    ③ 집계   — 여러 실행 → §13 목표 표와 같은 모양
    ④ 관통   — **실제 그래프**를 돌린 트래커에서 지표가 나오는가 (불변식)

LLM 스텁은 T12가 만든 것을 그대로 쓴다(R5). 같은 스텁을 두 벌 두면 한쪽만 고쳤을
때 두 테스트가 다른 것을 검사하게 된다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from app.progress import EventKind, LineState, ProgressEvent, ProgressTracker, close_status
from contracts.enums import VerdictState
from contracts.models import CriterionVerdict, Evidence
from graphs.analysis_graph import build_analysis_graph, node_sequence
from graphs.session import resume_or_start
from obs.metrics import (
    GATE_NODES,
    MetricsSummary,
    RunMetrics,
    append_record,
    collect_from_events,
    read_records,
    record_run,
    summarize,
)
from obs import metrics as metrics_module
from tests.test_hitl import JD, answers_for, initial_state, install_stubs
from tools.verify import EVIDENCE_OPTIONAL_STATES

WIRED_NODES = [name for name, _ in node_sequence(interactive=True)]
APP = str(Path(__file__).parent.parent / "app" / "main.py")


@pytest.fixture
def log(tmp_path):
    """지표 파일은 **항상 tmp_path**에 쓴다 — 기본 경로는 `.jobprep/`이고 그건 사용자 것이다."""
    return tmp_path / "metrics.jsonl"


def verdict(criterion_id: str, state: VerdictState, *, with_evidence: bool) -> CriterionVerdict:
    return CriterionVerdict(
        criterion_id=criterion_id,
        state=state,
        rationale="테스트 판정",
        evidence=[
            Evidence(source_name="프로필", quote="직접 담당했다", collected_at=date.today())
        ]
        if with_evidence
        else [],
    )


def node_done(node: str, payload: dict, seconds: float = 0.0) -> ProgressEvent:
    return ProgressEvent(
        kind=EventKind.NODE_DONE, node=node, label=node, seconds=seconds, payload=payload
    )


def tool_call(name: str, ok: bool = True) -> ProgressEvent:
    return ProgressEvent(
        kind=EventKind.TOOL_CALL, node="collect", label="수집", payload={"tool": name, "ok": ok}
    )


# --- ① 접기: 이벤트 → 지표 -----------------------------------------------------


def test_empty_event_stream_yields_an_empty_record():
    """아무 일도 없었으면 0이 아니라 "잴 게 없음"이다 — 비율은 `None`."""
    record = collect_from_events([])

    assert record.tool_calls == 0
    assert record.collection_success_rate is None
    assert record.evidence_attachment_rate is None
    assert record.hard_gate_passed is None


def test_response_time_is_the_sum_of_node_seconds():
    """응답 시간(§13)은 별도 계측 없이 스트림의 `seconds`에서 나온다."""
    record = collect_from_events(
        [
            node_done("extract", {"required": [1, 2]}, seconds=1.5),
            node_done("verify", {"verdicts": []}, seconds=2.5),
        ]
    )
    assert record.seconds == pytest.approx(4.0)


def test_collection_success_rate_counts_tool_calls():
    """수집 성공률(§13 목표 ≥90%) — 도구 호출 횟수도 같은 스트림에서 나온다."""
    record = collect_from_events([tool_call("fetch_jd_body"), tool_call("get_company_values", ok=False)])

    assert record.tool_calls == 2
    assert record.tool_calls_ok == 1
    assert record.collection_success_rate == pytest.approx(0.5)


def test_tool_call_without_ok_counts_as_success():
    """`ok`가 없으면 성공 — T13의 서브 라인이 ↳(성공 아이콘)를 그리는 것과 같은 기본값이다.

    여기서 기본값을 실패로 잡으면 화면은 ↳인데 지표만 실패로 세게 된다.
    """
    tracker = ProgressTracker(WIRED_NODES, clock=lambda: 0.0)
    tracker.handle(("custom", {"tool": "fetch_jd_body"}))  # `ok` 없음

    assert "↳" in tracker.as_markdown()
    assert collect_from_events(tracker.events).tool_calls_ok == 1


def test_evidence_rate_excludes_states_that_need_no_evidence():
    """근거 첨부율(§13 목표 100%)의 분모는 **근거가 필요한 판정**뿐이다.

    UNMET은 부재 자체가 근거라 인용 없이도 판정으로 남는다(D10). 이걸 분모에 넣으면
    100%가 구조적으로 불가능해져 지표가 늘 미달로 나온다. 기준 목록은 여기서 다시
    적지 않고 `EVIDENCE_OPTIONAL_STATES`에서 유도한다.
    """
    assert VerdictState.UNMET in EVIDENCE_OPTIONAL_STATES

    record = collect_from_events(
        [
            node_done(
                "verify",
                {
                    "verdicts": [
                        verdict("c1", VerdictState.MET, with_evidence=True),
                        verdict("c2", VerdictState.PARTIAL, with_evidence=True),
                        verdict("c3", VerdictState.UNMET, with_evidence=False),
                    ]
                },
            )
        ]
    )

    assert record.verdicts_scored == 2, "UNMET은 분모에서 빠진다"
    assert record.evidence_attachment_rate == pytest.approx(1.0)


def test_missing_evidence_drags_the_rate_down():
    record = collect_from_events(
        [
            node_done(
                "verify",
                {
                    "verdicts": [
                        verdict("c1", VerdictState.MET, with_evidence=True),
                        verdict("c2", VerdictState.MET, with_evidence=False),
                    ]
                },
            )
        ]
    )
    assert record.evidence_attachment_rate == pytest.approx(0.5)


def test_reverdicts_from_the_interview_replace_the_earlier_ones():
    """**`verify`만 보면 안 된다.** 델타 인터뷰가 목록을 통째로 갈아끼운다(T11).

    인터뷰 이전 목록으로 세면 답변으로 해소된 판정이 지표에 영영 안 잡힌다.
    """
    record = collect_from_events(
        [
            node_done("verify", {"verdicts": [verdict("c1", VerdictState.MET, with_evidence=False)]}),
            node_done(
                "delta_interview",
                {"verdicts": [verdict("c1", VerdictState.MET, with_evidence=True)]},
            ),
        ]
    )

    assert record.verdicts_scored == 1
    assert record.evidence_attachment_rate == pytest.approx(1.0)


@pytest.mark.parametrize("gate_node", GATE_NODES)
def test_hard_gate_is_read_from_either_graphs_gate_node(gate_node):
    """최소 요건 충족률 — UC-2(`quality_gate`)와 UC-1(`profile_gate`) 양쪽에서 읽는다."""
    record = collect_from_events(
        [node_done(gate_node, {"gate_status": {"jd_count": 1, "hard_gate_passed": True}})]
    )
    assert record.hard_gate_passed is True


def test_gate_status_from_other_nodes_is_ignored():
    """게이트 판정의 정본은 게이트 노드다 — `collect`도 같은 칸을 쓰지만 그건 중간값이다.

    수집 시점에는 JD가 없다가 H1 선택분이 붙어 통과하는 경우가 있어(T25 `quality_gate`가
    브리프 직전에 다시 잰다), 중간값을 최종값으로 세면 게이트가 거짓을 말한다.
    """
    record = collect_from_events(
        [node_done("collect", {"gate_status": {"jd_count": 0, "hard_gate_passed": False}})]
    )
    assert record.hard_gate_passed is None


# --- ② 파일: 실행 1회 = 한 줄 --------------------------------------------------


def test_each_run_appends_exactly_one_line(log):
    """**카드 완료 조건** — 실행 1회당 지표 레코드가 파일로 남는다."""
    append_record(RunMetrics(tool_calls=3), path=log)
    append_record(RunMetrics(tool_calls=5), path=log)

    assert len(log.read_text("utf-8").strip().splitlines()) == 2
    assert [r.tool_calls for r in read_records(log)] == [3, 5]


def test_reading_a_missing_log_is_not_an_error(tmp_path):
    """아직 한 번도 안 돌린 저장소에서 리포트를 열어도 죽지 않는다."""
    assert read_records(tmp_path / "없는파일.jsonl") == []


def test_blank_lines_in_the_log_are_skipped(log):
    """빈 줄이 섞여 있어도 읽힌다 — 이 파일은 **디스크 경계**라 우리만 쓰는 게 아니다.

    `.jobprep/metrics.jsonl`은 사람이 열어 보는 로그이고, 편집기가 끝에 줄바꿈을
    더하거나 사람이 줄을 지우고 갈 수 있다. 빈 줄 하나에 리포트 전체가 죽으면
    지표를 보려던 사람이 지표 대신 예외를 본다.
    """
    append_record(RunMetrics(tool_calls=1), path=log)
    log.write_text(log.read_text("utf-8") + "\n   \n", encoding="utf-8")

    assert [r.tool_calls for r in read_records(log)] == [1]


def test_the_log_directory_is_created_on_demand(tmp_path):
    nested = tmp_path / "새폴더" / "metrics.jsonl"
    append_record(RunMetrics(), path=nested)
    assert nested.exists()


def test_close_status_records_only_when_the_run_finished(log):
    """지표를 남기는 자리는 `close_status(on_complete=)` 하나다 (T28 불변식).

    중단 국면에서도 남기면 레코드가 **실행 횟수가 아니라 rerun 횟수**로 늘어난다 —
    답변을 두 라운드 하면 한 번의 분석이 세 줄이 된다.
    """
    class Box:
        def update(self, **kwargs):
            pass

    tracker = ProgressTracker(WIRED_NODES, clock=lambda: 0.0)
    tracker.lines[0].state = LineState.PAUSED

    close_status(Box(), tracker, on_complete=lambda t: record_run(t, path=log))
    assert read_records(log) == [], "중단 중에는 남기지 않는다"

    for line in tracker.lines:
        line.state = LineState.DONE
    close_status(Box(), tracker, on_complete=lambda t: record_run(t, path=log))
    assert len(read_records(log)) == 1


def test_close_status_without_the_hook_is_unchanged():
    """`on_complete`를 안 주면 T13이 고정한 거동 그대로다 — 기존 호출부가 안 깨진다."""
    seen = {}

    class Box:
        def update(self, **kwargs):
            seen.update(kwargs)

    tracker = ProgressTracker(WIRED_NODES, clock=lambda: 0.0)
    close_status(Box(), tracker)

    assert seen == {"label": tracker.status_label, "state": "running", "expanded": True}


# --- ③ 집계: 발표에 쓰는 숫자 --------------------------------------------------


def test_summarize_of_nothing_is_not_a_zero(log):
    """0건을 "성공률 0%"로 말하면 거짓말이다 — 잴 게 없으면 `None`."""
    empty = summarize(read_records(log))

    assert empty == MetricsSummary(runs=0)
    assert empty.collection_success_rate is None


def test_median_tool_calls_needs_several_runs():
    """"도구 호출 횟수 중앙값"(§13)은 실행 1회로는 낼 수 없는 숫자다."""
    summary = summarize(
        [RunMetrics(tool_calls=2), RunMetrics(tool_calls=9), RunMetrics(tool_calls=4)]
    )
    assert summary.runs == 3
    assert summary.median_tool_calls == 4


def test_rates_pool_across_runs_instead_of_averaging_rates():
    """비율의 평균이 아니라 **합의 비율**이다.

    호출 1건 중 1건 성공(100%)과 9건 중 5건 성공(56%)을 평균 내면 78%가 나오는데,
    실제 성공률은 10건 중 6건 = 60%다. 목표(≥90%)와 비교하는 숫자라 이 차이가 곧
    합격·불합격을 가른다.
    """
    summary = summarize(
        [
            RunMetrics(tool_calls=1, tool_calls_ok=1),
            RunMetrics(tool_calls=9, tool_calls_ok=5),
        ]
    )
    assert summary.collection_success_rate == pytest.approx(0.6)


def test_gate_pass_rate_ignores_runs_that_never_reached_a_gate():
    summary = summarize(
        [
            RunMetrics(hard_gate_passed=True),
            RunMetrics(hard_gate_passed=False),
            RunMetrics(hard_gate_passed=None),
        ]
    )
    assert summary.hard_gate_pass_rate == pytest.approx(0.5)


# --- ④ 관통: 실제 그래프가 흘린 이벤트 -----------------------------------------


def test_metrics_come_from_a_real_graph_run(monkeypatch, log):
    """**T28 불변식의 증명** — 계측을 새로 심지 않고 T13 스트림만으로 지표가 나온다.

    합성 이벤트가 아니라 실제 분석 그래프를 중단·재개까지 관통시킨 트래커에서
    뽑는다. 여기가 초록불이면 `app/progress.py`에 붙는 것만으로 충분하다는 뜻이다.
    """
    install_stubs(monkeypatch)
    graph = build_analysis_graph(InMemorySaver(), interactive=True)
    tracker = ProgressTracker(WIRED_NODES)

    status = resume_or_start(graph, "t-metrics", initial_input=initial_state(), on_event=tracker.handle)
    assert status.is_interrupted
    resume_or_start(graph, "t-metrics", resume=answers_for(status), on_event=tracker.handle)
    assert tracker.is_complete

    record = record_run(tracker, path=log)

    assert read_records(log) == [record], "완료된 실행 하나가 한 줄로 남는다"

    # **`> 0`으로 걸지 않는다.** Windows의 `time.monotonic()`은 눈금이 ~15.6ms라
    # 스텁으로 돌린 노드가 전부 한 눈금 안에 끝나면 소요 시간이 정확히 0.0이 된다
    # (이 파일만 돌리면 통과하고 전체 스위트에서만 깨지는 종류의 그물이다). 여기서
    # 걸 것은 시계의 해상도가 아니라 **지표가 스트림이 잰 값을 그대로 쓴다는 것**이다.
    assert record.seconds == pytest.approx(
        sum(e.seconds for e in tracker.events if e.kind is EventKind.NODE_DONE)
    ), "응답 시간을 다시 재지 않고 스트림 값을 그대로 접는다"
    assert record.hard_gate_passed is True, "quality_gate가 흘린 판정이 잡힌다"
    assert record.verdicts_scored > 0
    assert record.evidence_attachment_rate == pytest.approx(1.0), "근거 첨부율 목표는 100%다"


def test_a_listener_records_the_same_events_the_screen_shows(monkeypatch):
    """지표가 화면과 **같은 이벤트**를 본다 — 두 숫자가 어긋날 자리가 없다."""
    install_stubs(monkeypatch)
    graph = build_analysis_graph(InMemorySaver(), interactive=True)
    tracker = ProgressTracker(WIRED_NODES)

    mirrored: list[ProgressEvent] = []
    tracker.add_listener(mirrored.append)

    status = resume_or_start(graph, "t-mirror", initial_input=initial_state(), on_event=tracker.handle)
    resume_or_start(graph, "t-mirror", resume=answers_for(status), on_event=tracker.handle)

    # `run_id`는 레코드마다 새로 발급되므로 빼고 본다 — 대조 대상은 측정값이다.
    assert collect_from_events(mirrored).model_dump(
        exclude={"run_id"}
    ) == collect_from_events(tracker.events).model_dump(exclude={"run_id"})


def test_the_mode_is_derived_from_which_gate_ran(monkeypatch, log):
    """두 그래프가 로그 하나를 공유하므로 **모드가 적혀야 한다**.

    `hard_gate_passed`의 뜻이 모드마다 다르다 — UC-2는 "JD를 건졌다", UC-1은
    "프로필이 완성됐다". 갈라 적지 않으면 집계가 서로 다른 것을 한 칸에 더한다.
    모드도 별도 인자가 아니라 **같은 스트림**에서 유도한다(어느 게이트가 돌았나).
    """
    install_stubs(monkeypatch)
    graph = build_analysis_graph(InMemorySaver(), interactive=True)
    tracker = ProgressTracker(WIRED_NODES)

    status = resume_or_start(graph, "t-mode", initial_input=initial_state(), on_event=tracker.handle)
    resume_or_start(graph, "t-mode", resume=answers_for(status), on_event=tracker.handle)

    assert record_run(tracker, path=log).mode == "analysis"


# --- ⑤ 배선: 실제 앱이 지표를 남긴다 (§2-1) -------------------------------------


def test_the_running_app_writes_a_metrics_record(monkeypatch, tmp_path):
    """**§2-1 체크 — "이 카드가 만드는 것을 누가 부르는가?"**

    `record_run()`을 만들어 놓고 아무도 안 부르면 T14가 남겼던 배선 갭(D43)과
    같은 것이 된다. 그래서 합성 트래커가 아니라 `AppTest`로 실제 `app/main.py`를
    구동해 파일이 실제로 떨어지는지 본다.

    기본 로그 경로를 `tmp_path`로 갈아끼우는 것이 중요하다 — 안 그러면 테스트가
    사용자의 `.jobprep/metrics.jsonl`에 줄을 쌓는다. `record_run()`이 경로를 기본
    인자가 아니라 **호출 시점에** 읽기 때문에 이 갈아끼우기가 먹는다(D71).
    """
    log = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(metrics_module, "DEFAULT_LOG", log)
    install_stubs(monkeypatch)

    import graphs.session as session

    monkeypatch.setattr(session, "build_checkpointer", lambda *a, **kw: InMemorySaver())
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.text_input[0].set_value("테크노베이션")
    at.text_input[1].set_value("백엔드 엔지니어")
    at.text_area[0].set_value(JD.raw_text)
    at = at.button[0].click().run()
    assert not at.exception
    assert read_records(log) == [], "중단 국면에서는 아직 안 남는다"

    for widget in at.text_area:
        widget.set_value("직전 프로젝트에서 직접 담당했습니다.")
    at = at.button[0].click().run()
    assert not at.exception

    records = read_records(log)
    assert len(records) == 1, "완료된 분석 하나가 한 줄로 남는다"
    assert records[0].mode == "analysis"
    assert records[0].hard_gate_passed is True

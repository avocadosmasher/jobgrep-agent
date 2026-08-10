"""T13 · 진행 표시.

카드의 검증 방식은 "수동 — 화면 녹화 또는 스크린샷"인데, 실 API가 크레딧 소진으로
막혀 있어(D26·D31) 그대로는 완료 조건을 증명할 수 없다. 그래서 D31과 같은 판단으로
**화면에 나타나는 것을 자동으로 고정**한다. 스크린샷은 무엇이 보였는지 남기지만
무엇이 보여야 하는지는 고정하지 못한다 — 진행 표시는 노드가 하나 늘 때마다 조용히
어긋나는 종류의 화면이라 그물이 필요하다.

층은 넷이다:
    ① 프리셋   — 내부 함수명이 화면에 새지 않는가 (카드 불변식)
    ② 트래커   — 청크 → ⬜/🔄/✅/⏸ 전이가 맞는가 (순수 함수, Streamlit 없음)
    ③ 실그래프 — 실제 중단·재개에서 ⏸가 옳은 줄에 서는가
    ④ 실앱     — `AppTest`로 `app/main.py`를 구동해 패널이 실제로 그려지는가

LLM 스텁은 T12가 만든 것을 그대로 쓴다(R5 — 반환값은 전부 `fixtures/`의 골든 데이터).
같은 스텁을 두 벌 두면 한쪽만 고쳤을 때 두 테스트가 다른 것을 검사하게 된다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from app.progress import (
    EventKind,
    LineState,
    ProgressTracker,
    count_of,
    live_status,
    load_labels,
    render_panel,
)
from graphs.analysis_graph import build_analysis_graph, node_sequence
from graphs.session import ThreadPhase, resume_or_start
from tests.test_hitl import (
    JD,
    OPEN_CRITERION_IDS,
    answers_for,
    initial_state,
    install_stubs,
)

APP = str(Path(__file__).parent.parent / "app" / "main.py")

WIRED_NODES = [name for name, _ in node_sequence(interactive=True)]


@pytest.fixture
def tracker() -> ProgressTracker:
    # 시계를 고정해 소요 시간이 화면 문자열에 섞이지 않게 한다 — 테스트가
    # 실행 속도에 따라 흔들리면 그물이 아니라 소음이 된다.
    return ProgressTracker(WIRED_NODES, clock=lambda: 0.0)


@pytest.fixture
def hitl_graph():
    return build_analysis_graph(InMemorySaver(), interactive=True)


def labels_of(tracker: ProgressTracker) -> list[str]:
    return [line.label for line in tracker.lines]


def icons_of(tracker: ProgressTracker) -> list[str]:
    return [line.state.value for line in tracker.lines]


def line_of(tracker: ProgressTracker, node: str):
    return next(line for line in tracker.lines if line.node == node)


# --- ① 프리셋: 내부 함수명 노출 금지 -------------------------------------------


def test_every_wired_node_has_a_user_facing_label():
    """배선된 노드는 전부 프리셋에 있어야 한다 — 하나라도 빠지면 화면이 흐려진다."""
    labels = load_labels()
    missing = [node for node in WIRED_NODES if node not in labels.nodes]
    assert missing == [], f"presets/node_labels.yaml에 없는 노드: {missing}"


def test_internal_function_names_never_reach_the_screen(tracker):
    """**카드 불변식.** 화면 어디에도 내부 노드명이 나오지 않는다."""
    rendered = tracker.as_markdown()
    for node in WIRED_NODES:
        assert node not in rendered, f"내부 함수명 '{node}'이 화면에 노출됐다"


def test_unknown_node_falls_back_instead_of_leaking_its_name():
    """모르는 노드가 와도 이름 대신 fallback 문구가 나온다."""
    labels = load_labels()
    unknown = ProgressTracker(["totally_internal_helper"], labels=labels, clock=lambda: 0.0)

    assert labels.node("totally_internal_helper").label == labels.fallback
    assert "totally_internal_helper" not in unknown.as_markdown()


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1, 2, 3], 3),
        ({"a": [1, 2], "b": [3]}, 3),   # criteria — dict[str, list]는 안쪽을 더한다
        ({"a": 1, "b": 2}, 2),          # interview_answers — 평범한 dict는 길이
        (None, 0),
        ("문자열", 1),
    ],
)
def test_count_of_folds_state_values(value, expected):
    assert count_of(value) == expected


# --- ② 트래커: 상태 전이 -------------------------------------------------------


def test_first_line_is_running_before_anything_arrives(tracker):
    assert icons_of(tracker)[0] == LineState.RUNNING.value
    assert set(icons_of(tracker)[1:]) == {LineState.PENDING.value}


def test_node_update_marks_done_and_wakes_the_next(tracker):
    event = tracker.handle(("updates", {"ingest_pasted_jd": {"source_docs": [1]}}))

    assert event is not None and event.kind is EventKind.NODE_DONE
    assert line_of(tracker, "ingest_pasted_jd").state is LineState.DONE
    # "다음 줄"은 배선 순서가 정한다 — T18이 `collect`를 그 자리에 꽂았다.
    assert line_of(tracker, WIRED_NODES[1]).state is LineState.RUNNING


def test_counts_are_rendered_in_user_language(tracker):
    tracker.handle(("updates", {"extract": {"required": list(range(17))}}))

    assert line_of(tracker, "extract").text == "✅ 요구 역량 추출 (17개)"


def test_zero_counts_can_be_hidden(tracker):
    """물을 게 없으면 "0건 확인 필요"를 붙이지 않는다 — 없는 일은 안 적는다."""
    tracker.handle(
        ("updates", {"verify": {"verdicts": [1, 2, 3], "pending_questions": []}})
    )
    assert line_of(tracker, "verify").text == "✅ 내 프로필과 대조 (3건 판정)"

    later = ProgressTracker(WIRED_NODES, clock=lambda: 0.0)
    later.handle(("updates", {"verify": {"verdicts": [1], "pending_questions": [1, 2]}}))
    assert line_of(later, "verify").text == "✅ 내 프로필과 대조 (1건 판정 · 2건 확인 필요)"


def test_single_mode_chunks_are_accepted_too(tracker):
    """`stream_mode="updates"` 단일 모드는 튜플이 아니라 dict로 온다."""
    tracker.handle({"ingest_pasted_jd": {"source_docs": [1, 2]}})
    assert line_of(tracker, "ingest_pasted_jd").state is LineState.DONE


def test_failure_marks_the_standing_line(tracker):
    tracker.handle(("updates", {"ingest_pasted_jd": {"source_docs": [1]}}))
    tracker.fail("RateLimitError: 429")

    assert line_of(tracker, WIRED_NODES[1]).state is LineState.FAILED
    assert tracker.status_state == "error"
    assert "429" in tracker.as_markdown()


# --- ③ 서브 라인 훅 (T18) ------------------------------------------------------


def test_tool_calls_hang_under_the_running_node(tracker):
    """수집 ReAct의 내부 tool call이 서브 라인으로 붙는다 — T18은 코드를 안 고친다."""
    event = tracker.handle(("custom", {"tool": "fetch_jd_body", "ok": True}))

    assert event is not None and event.kind is EventKind.TOOL_CALL
    assert line_of(tracker, "ingest_pasted_jd").substeps == ["↳ 공고 본문 가져오기"]
    assert "fetch_jd_body" not in tracker.as_markdown()  # 도구도 내부명을 안 흘린다


def test_tool_calls_reach_the_panel_through_the_real_stream(monkeypatch, hitl_graph, tracker):
    """훅이 **실제로 뚫려 있는가** — 합성 청크가 아니라 노드 안에서 흘려본다.

    T18은 이 경로만 쓰면 되고 `app/progress.py`를 고칠 필요가 없다. 여기서 한 번
    관통시켜 두지 않으면 훅을 "마련했다"고 말할 근거가 없다(카드 불변식).
    """
    from langgraph.config import get_stream_writer

    from nodes import analysis_nodes

    install_stubs(monkeypatch)
    stubbed = analysis_nodes.extract_competencies

    def noisy_extract(docs, role):
        get_stream_writer()({"tool": "fetch_jd_body", "ok": True, "detail": "1건"})
        return stubbed(docs, role)

    monkeypatch.setattr(analysis_nodes, "extract_competencies", noisy_extract)

    resume_or_start(hitl_graph, "t-tool", initial_input=initial_state(), on_event=tracker.handle)

    # 서브 라인은 **그 노드가 도는 중에** 붙는다 — 끝난 뒤가 아니라.
    assert line_of(tracker, "extract").substeps == ["↳ 공고 본문 가져오기 — 1건"]
    assert "fetch_jd_body" not in tracker.as_markdown()

    kinds = [e.kind for e in tracker.events]
    assert kinds.index(EventKind.TOOL_CALL) < kinds.index(EventKind.INTERRUPTED)


def test_failed_tool_call_is_marked_but_does_not_stop_the_line(tracker):
    tracker.handle(("custom", {"tool": "get_company_values", "ok": False, "detail": "404"}))

    assert line_of(tracker, "ingest_pasted_jd").substeps == ["⚠️ 인재상·가치 확인 — 404"]
    assert line_of(tracker, "ingest_pasted_jd").state is LineState.RUNNING


# --- ④ T28 훅: 같은 소스, 중복 계측 없음 ---------------------------------------


def test_listeners_receive_the_same_events(tracker):
    """T28은 계측을 새로 심지 않고 여기 붙는다 (T28 불변식)."""
    seen = []
    tracker.add_listener(seen.append)

    tracker.handle(("custom", {"tool": "fetch_jd_body"}))
    tracker.handle(("updates", {"ingest_pasted_jd": {"source_docs": [1]}}))
    tracker.handle(("updates", {"extract": {"required": [1, 2]}}))

    assert seen == tracker.events, "화면과 지표가 같은 이벤트를 봐야 한다"
    assert [e.kind for e in seen] == [
        EventKind.TOOL_CALL,
        EventKind.NODE_DONE,
        EventKind.NODE_DONE,
    ]
    # T28이 필요한 것 — 도구 호출 횟수와 원본 페이로드가 이미 실려 있다.
    assert sum(1 for e in seen if e.kind is EventKind.TOOL_CALL) == 1
    assert seen[-1].payload == {"required": [1, 2]}


def test_elapsed_seconds_come_from_the_same_stream(tracker):
    """응답 시간(T28 지표)도 별도 계측 없이 이 스트림에서 나온다."""
    ticks = iter([0.0, 1.5, 4.0])
    timed = ProgressTracker(WIRED_NODES, clock=lambda: next(ticks))

    timed.handle(("updates", {"ingest_pasted_jd": {"source_docs": [1]}}))
    timed.handle(("updates", {"extract": {"required": [1]}}))

    assert [round(e.seconds, 1) for e in timed.events] == [1.5, 2.5]


# --- ⑤ 실그래프: 중단이 옳은 줄에 선다 -----------------------------------------


def collect_events(graph, thread_id, tracker, **kwargs):
    return resume_or_start(graph, thread_id, on_event=tracker.handle, **kwargs)


def test_interrupt_pauses_the_interview_line(monkeypatch, hitl_graph, tracker):
    """**T13 완료 조건 앞쪽 절반** — HITL 중단 시 ⏸로 멈춘다."""
    install_stubs(monkeypatch)

    status = collect_events(hitl_graph, "t-progress", tracker, initial_input=initial_state())
    assert status.phase is ThreadPhase.INTERRUPTED

    interview = line_of(tracker, "delta_interview")
    assert interview.state is LineState.PAUSED
    assert interview.detail == f"{len(OPEN_CRITERION_IDS)}건 질문 대기"

    # 중단 앞쪽은 전부 끝났고, 뒤쪽은 아직 시작도 안 했다.
    upto = WIRED_NODES.index("delta_interview")
    assert all(line.state is LineState.DONE for line in tracker.lines[:upto])
    assert all(line.state is LineState.PENDING for line in tracker.lines[upto + 1 :])
    assert tracker.is_paused and tracker.status_state == "running"


def test_resume_completes_every_line(monkeypatch, hitl_graph, tracker):
    """**완료 조건 뒤쪽 절반** — 재개하면 남은 줄이 순서대로 채워진다."""
    install_stubs(monkeypatch)

    status = collect_events(hitl_graph, "t-progress", tracker, initial_input=initial_state())
    collect_events(hitl_graph, "t-progress", tracker, resume=answers_for(status))

    assert icons_of(tracker) == [LineState.DONE.value] * len(WIRED_NODES)
    assert tracker.is_complete and tracker.status_state == "complete"
    assert "⬜" not in tracker.as_markdown()


def test_upstream_lines_are_not_reset_on_resume(monkeypatch, hitl_graph, tracker):
    """재개는 상류 노드를 다시 흘리지 않는다(D27). 트래커를 새로 만들면 ✅가 ⬜로 되돌아간다."""
    install_stubs(monkeypatch)

    status = collect_events(hitl_graph, "t-keep", tracker, initial_input=initial_state())
    collect_events(hitl_graph, "t-keep", tracker, resume=answers_for(status))

    replayed = [e.node for e in tracker.events if e.kind is EventKind.NODE_DONE]
    assert replayed.count("extract") == 1, "재개 후에도 상류 노드 이벤트는 한 번뿐이다"
    assert line_of(tracker, "extract").state is LineState.DONE


def test_self_loop_does_not_promote_the_interview_line_early(monkeypatch, hitl_graph, tracker):
    """2라운드로 들어가는 동안 델타 인터뷰 줄이 ✅로 앞서 나가지 않는다.

    `repeats_while`이 없으면 1라운드 답변 직후 그 줄이 ✅가 되고, 곧바로 다시 물을 때
    화면이 뒤로 돌아간다.
    """
    install_stubs(monkeypatch, answers_resolve=False)  # 무슨 답을 줘도 판정이 안 선다

    status = collect_events(hitl_graph, "t-loop", tracker, initial_input=initial_state())
    rounds = 0
    while status.is_interrupted:
        rounds += 1
        assert line_of(tracker, "delta_interview").state is LineState.PAUSED
        assert line_of(tracker, "aggregate").state is LineState.PENDING, "앞서 나가면 안 된다"
        status = collect_events(hitl_graph, "t-loop", tracker, resume=answers_for(status))

    assert rounds == 2  # T11의 MAX_ROUNDS
    assert tracker.is_complete


def test_on_event_none_keeps_the_invoke_path(monkeypatch, hitl_graph):
    """`on_event`를 안 주면 기존 경로 그대로다 — T10·T12가 고정한 거동이 안 바뀐다."""
    install_stubs(monkeypatch)

    status = resume_or_start(hitl_graph, "t-plain", initial_input=initial_state())
    assert status.phase is ThreadPhase.INTERRUPTED

    status = resume_or_start(hitl_graph, "t-plain", resume=answers_for(status))
    assert status.phase is ThreadPhase.COMPLETE


def test_completed_thread_streams_nothing(monkeypatch, hitl_graph, tracker):
    """완료된 스레드는 이벤트도 안 흘린다 — 실행하지 않기로 한 국면이므로(D27)."""
    install_stubs(monkeypatch)

    status = collect_events(hitl_graph, "t-quiet", tracker, initial_input=initial_state())
    collect_events(hitl_graph, "t-quiet", tracker, resume=answers_for(status))
    settled = len(tracker.events)

    collect_events(hitl_graph, "t-quiet", tracker, initial_input=initial_state())
    assert len(tracker.events) == settled


# --- ⑥ 실앱 관통 (AppTest) -----------------------------------------------------


def run_app(monkeypatch):
    import graphs.session as session

    monkeypatch.setattr(session, "build_checkpointer", lambda *a, **kw: InMemorySaver())
    return AppTest.from_file(APP, default_timeout=30).run()


def fill_input_form(at):
    at.text_input[0].set_value("테크노베이션")
    at.text_input[1].set_value("백엔드 엔지니어")
    at.text_area[0].set_value(JD.raw_text)
    return at.button[0].click().run()


def panel_text(at) -> str:
    assert at.status, "진행 표시 패널이 떠야 한다"
    return "\n".join(block.value for block in at.markdown)


def test_app_shows_progress_and_pauses_at_the_interview(monkeypatch):
    """**T13 완료 조건 그대로** — 실행 중 진행이 표시되고 HITL 중단에서 ⏸로 멈춘다."""
    install_stubs(monkeypatch)
    at = fill_input_form(run_app(monkeypatch))
    assert not at.exception

    body = panel_text(at)
    assert "✅ 요구 역량 추출" in body
    assert "⏸ 델타 인터뷰" in body
    assert "⬜ 3트랙 전략 브리프 작성" in body
    assert at.status[0].label == load_labels().paused_title
    assert at.status[0].state == "running", "답변을 기다리는 중이지 끝난 게 아니다"

    # 내부 함수명은 화면 어디에도 없다.
    for node in WIRED_NODES:
        assert node not in body


def test_app_progress_completes_after_the_answer(monkeypatch):
    install_stubs(monkeypatch)
    at = fill_input_form(run_app(monkeypatch))

    for widget in at.text_area:
        widget.set_value("직전 프로젝트에서 직접 담당했습니다.")
    at = at.button[0].click().run()
    assert not at.exception

    body = panel_text(at)
    assert "⏸" not in body and "⬜" not in body
    assert "✅ 총평·전략 문장 채우기" in body
    assert at.status[0].label == load_labels().complete_title
    assert at.status[0].state == "complete"
    assert at.download_button, "결과 화면은 그대로 나와야 한다"


def test_both_panels_draw_into_the_same_slot():
    """지난 진행과 실행 패널은 **같은 자리**에 그린다.

    답변 제출 국면은 "지난 진행을 먼저 그리고, 이어서 실행 패널을 여는" 순서다.
    자리를 `st.empty()`로 잡아두지 않으면 묵은 ⏸ 패널과 도는 중인 패널이 화면에
    나란히 남는다(실행이 1~2분이라 눈에 그대로 띈다).

    **이건 `AppTest`로 못 잡는다.** `st.rerun()`이 끼어 있어 결과 트리에는 마지막
    rerun의 화면만 남고, 둘이던 그 국면은 흔적이 사라진다. 그래서 화면 결과가
    아니라 "어디에 그리는가"를 직접 고정한다.
    """
    drawn: list[str] = []

    class Box:
        def markdown(self, body):
            drawn.append("markdown")

        def empty(self):
            return self

        def update(self, **kwargs):
            drawn.append("update")

    class Slot:
        def status(self, label, **kwargs):
            drawn.append("status")
            return Box()

    tracker = ProgressTracker(WIRED_NODES, clock=lambda: 0.0)
    slot = Slot()

    render_panel(tracker, slot=slot)
    live_status(tracker, slot=slot)

    assert drawn.count("status") == 2, "두 패널 모두 넘긴 자리에 그려야 한다"


def test_panel_slot_is_a_single_element_placeholder():
    """자리는 `st.empty()`여야 한다 — 덮어쓰기가 되는 컨테이너는 이것뿐이다."""
    at = AppTest.from_string(
        "import sys; sys.path.insert(0, r'%s')\n"
        "from app.progress import panel_slot\n"
        "slot = panel_slot()\n"
        "slot.status('첫 번째')\n"
        "slot.status('두 번째')\n" % str(Path(__file__).parent.parent)
    ).run()

    assert not at.exception
    assert [s.label for s in at.status] == ["두 번째"], "앞 패널이 지워져야 한다"


def test_app_progress_resets_on_new_analysis(monkeypatch):
    """[새 분석 시작]은 진행 표시도 함께 비운다 — 앞 분석의 ✅가 남으면 거짓말이 된다."""
    install_stubs(monkeypatch)
    at = fill_input_form(run_app(monkeypatch))
    for widget in at.text_area:
        widget.set_value("담당했습니다.")
    at = at.button[0].click().run()
    assert at.status

    at = at.button[-1].click().run()  # [새 분석 시작]

    assert not at.exception
    assert not at.status, "새 분석에서는 진행 표시가 비어 있어야 한다"
    assert at.text_input, "입력 폼으로 돌아와야 한다"

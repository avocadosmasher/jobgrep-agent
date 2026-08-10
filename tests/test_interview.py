"""T11 · 델타 인터뷰 노드 검증.

두 가지를 증명한다 (카드 완료 조건):
  ① 중단 → 답변 주입 → 재개하면 verdict가 갱신된다
  ② 3라운드에 진입하지 못한다

기준·판정은 `fixtures/criteria_sample.json`·`verdicts_*.json`을 쓴다 (R5).
`verify_criteria`는 LLM을 타므로 재판정 결과만 스텁으로 주입하고, 노드가 그것을
어떻게 병합·차단하는지를 본다 — 판정 품질은 T05의 테스트가 이미 본다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from contracts.enums import VerdictState
from contracts.models import Criterion, CriterionVerdict, Evidence, ProfileJSON, Question
from contracts.state import GraphState
from nodes import interview
from nodes.interview import (
    MAX_ROUNDS,
    ROUND_LIMIT_RATIONALE,
    build_interview_payload,
    delta_interview,
    merge_verdicts,
)
from tools.verify import question_id_for

FIXTURES = Path(__file__).parent.parent / "fixtures"
PROFILE = ProfileJSON.model_validate_json((FIXTURES / "profile_sample.json").read_bytes())
CRITERIA_BY_COMP: dict[str, list[Criterion]] = {
    comp_id: [Criterion.model_validate(c) for c in group]
    for comp_id, group in json.loads(
        (FIXTURES / "criteria_sample.json").read_text("utf-8")
    ).items()
}
ALL_CRITERIA = [c for group in CRITERIA_BY_COMP.values() for c in group]
ASKED = ALL_CRITERIA[:3]


def question_for(criterion: Criterion) -> Question:
    return Question(
        question_id=question_id_for(criterion.criterion_id),
        criterion_id=criterion.criterion_id,
        text=f"{criterion.text} — 해당 경험이 있나?",
        options=["예", "아니오"],
    )


def met_verdict(criterion: Criterion) -> CriterionVerdict:
    return CriterionVerdict(
        criterion_id=criterion.criterion_id,
        state=VerdictState.MET,
        rationale="델타 인터뷰 답변으로 확인됨",
        evidence=[
            Evidence(source_name="델타 인터뷰", quote="예, 운영 경험이 있다", collected_at=PROFILE.built_at)
        ],
    )


def base_state(**over) -> GraphState:
    state: GraphState = {
        "mode": "analysis",
        "profile": PROFILE,
        "criteria": CRITERIA_BY_COMP,
        "verdicts": [],
        "pending_questions": [question_for(c) for c in ASKED],
        "interview_answers": {},
        "interview_round": 0,
    }
    state.update(over)
    return state


@pytest.fixture
def answered(monkeypatch):
    """`interrupt()`를 즉시 답을 돌려주는 스텁으로 바꾼다 (중단 없이 노드 로직만 본다)."""
    seen: list[dict] = []

    def fake_interrupt(payload):
        seen.append(payload)
        return {q["question_id"]: "예" for q in payload["questions"]}

    monkeypatch.setattr(interview, "interrupt", fake_interrupt)
    return seen


@pytest.fixture
def reverify(monkeypatch):
    """재판정 스텁 — 물어본 기준을 전부 MET으로 돌려준다."""
    calls: list[tuple] = []

    def fake_verify(criteria, profile, answers=None):
        calls.append((list(criteria), answers))
        return [met_verdict(c) for c in criteria], []

    monkeypatch.setattr(interview, "verify_criteria", fake_verify)
    return calls


# --- 건너뛰기 -----------------------------------------------------------------


def test_no_questions_means_no_interrupt(monkeypatch):
    """질문이 없으면 묻지 않는다 — §9-3 H3 생략 조건."""

    def explode(payload):
        raise AssertionError("질문이 없는데 그래프를 중단시켰다")

    monkeypatch.setattr(interview, "interrupt", explode)
    assert delta_interview(base_state(pending_questions=[])) == {}


# --- 배치 --------------------------------------------------------------------


def test_all_questions_are_asked_in_one_interrupt(answered, reverify):
    delta_interview(base_state())

    assert len(answered) == 1, "질문은 배치로 한 번에 묻는다 (§9-3)"
    payload = answered[0]
    assert payload["kind"] == "delta_interview"
    assert payload["round"] == 1
    assert payload["max_rounds"] == MAX_ROUNDS
    assert [q["criterion_id"] for q in payload["questions"]] == [
        c.criterion_id for c in ASKED
    ]


def test_payload_carries_what_the_form_needs():
    payload = build_interview_payload([question_for(ASKED[0])], 1)
    entry = payload["questions"][0]
    assert entry["question_id"] == question_id_for(ASKED[0].criterion_id)
    assert entry["text"] and entry["options"] == ["예", "아니오"]


# --- 완료 조건 ① 재개하면 verdict가 갱신된다 ---------------------------------


def test_answers_update_verdicts(answered, reverify):
    update = delta_interview(base_state())

    assert {v.criterion_id for v in update["verdicts"]} == {c.criterion_id for c in ASKED}
    assert all(v.state is VerdictState.MET for v in update["verdicts"])
    assert update["pending_questions"] == []
    assert update["interview_round"] == 1
    assert update["interview_answers"] == {
        question_id_for(c.criterion_id): "예" for c in ASKED
    }


def test_answers_are_passed_to_reverification(answered, reverify):
    delta_interview(base_state())

    criteria, answers = reverify[0]
    assert [c.criterion_id for c in criteria] == [c.criterion_id for c in ASKED]
    assert answers == {question_id_for(c.criterion_id): "예" for c in ASKED}


def test_existing_verdicts_are_overwritten_not_duplicated(answered, reverify):
    stale = CriterionVerdict(
        criterion_id=ASKED[0].criterion_id,
        state=VerdictState.UNKNOWN,
        rationale="근거 없음",
    )
    update = delta_interview(base_state(verdicts=[stale]))

    matching = [v for v in update["verdicts"] if v.criterion_id == ASKED[0].criterion_id]
    assert len(matching) == 1, "기준 하나당 판정은 하나뿐이다"
    assert matching[0].state is VerdictState.MET


def test_verdicts_for_untouched_criteria_survive(answered, reverify):
    other = CriterionVerdict(
        criterion_id=ALL_CRITERIA[-1].criterion_id,
        state=VerdictState.MET,
        rationale="이전 판정",
    )
    update = delta_interview(base_state(verdicts=[other]))
    assert other in update["verdicts"]


# --- 완료 조건 ② 3라운드 진입 불가 -------------------------------------------


def test_third_round_is_refused(monkeypatch, reverify):
    """★ 상한 초과 — 묻지 않고 잔여를 판단보류로 확정한다 (§12-4)."""

    def explode(payload):
        raise AssertionError(f"{MAX_ROUNDS}라운드를 넘겨 또 물었다")

    monkeypatch.setattr(interview, "interrupt", explode)

    update = delta_interview(base_state(interview_round=MAX_ROUNDS))

    assert update["pending_questions"] == []
    assert {v.state for v in update["verdicts"]} == {VerdictState.UNKNOWN}
    assert all(v.rationale == ROUND_LIMIT_RATIONALE for v in update["verdicts"])
    assert "interview_round" not in update, "상한 초과는 라운드를 더 쓰지 않는다"


def test_rounds_are_consumed_one_at_a_time(answered, monkeypatch):
    """라운드를 하나씩 쓰고, 2라운드를 다 쓰면 3라운드는 거부된다."""
    stubborn = [question_for(ASKED[0])]  # 답해도 계속 미결로 남는 항목

    monkeypatch.setattr(
        interview,
        "verify_criteria",
        lambda criteria, profile, answers=None: (
            [met_verdict(c) for c in criteria],
            list(stubborn),
        ),
    )

    state = base_state()
    for expected_round in (1, 2):
        update = delta_interview(state)
        assert update["interview_round"] == expected_round
        state = {**state, **update}

    assert state["pending_questions"], "미결이 남아야 상한 검사가 의미 있다"
    assert len(answered) == MAX_ROUNDS, f"{MAX_ROUNDS}번만 물었어야 한다"

    final = delta_interview(state)

    assert final["pending_questions"] == []
    held = [v for v in final["verdicts"] if v.rationale == ROUND_LIMIT_RATIONALE]
    assert [v.criterion_id for v in held] == [ASKED[0].criterion_id]
    assert all(v.state is VerdictState.UNKNOWN for v in held)
    assert len(answered) == MAX_ROUNDS, "3라운드에서 또 물었다"


def test_second_round_only_asks_what_stayed_open(answered, monkeypatch):
    leftover = [question_for(ASKED[0])]

    def fake_verify(criteria, profile, answers=None):
        return [met_verdict(c) for c in criteria], list(leftover)

    monkeypatch.setattr(interview, "verify_criteria", fake_verify)

    state = base_state()
    state = {**state, **delta_interview(state)}
    assert [q.question_id for q in state["pending_questions"]] == [
        q.question_id for q in leftover
    ]

    delta_interview(state)
    assert answered[-1]["round"] == 2
    assert [q["criterion_id"] for q in answered[-1]["questions"]] == [
        ASKED[0].criterion_id
    ]


# --- 답변 정규화 --------------------------------------------------------------


@pytest.mark.parametrize(
    "replies",
    [
        {question_id_for(ASKED[0].criterion_id): "예"},   # question_id 키
        {ASKED[0].criterion_id: "예"},                     # criterion_id 키도 받아준다
    ],
)
def test_reply_keys_are_normalized(monkeypatch, reverify, replies):
    monkeypatch.setattr(interview, "interrupt", lambda payload: replies)

    update = delta_interview(base_state(pending_questions=[question_for(ASKED[0])]))
    assert update["interview_answers"] == {question_id_for(ASKED[0].criterion_id): "예"}


def test_blank_replies_are_discarded(monkeypatch, reverify):
    monkeypatch.setattr(
        interview,
        "interrupt",
        lambda payload: {q["question_id"]: "   " for q in payload["questions"]},
    )
    update = delta_interview(base_state())
    assert update["interview_answers"] == {}, "빈 답변을 근거로 넘기면 안 된다"


def test_list_replies_pair_with_question_order(monkeypatch, reverify):
    monkeypatch.setattr(interview, "interrupt", lambda payload: ["예", "아니오", "예"])

    update = delta_interview(base_state())
    assert update["interview_answers"] == {
        question_id_for(ASKED[0].criterion_id): "예",
        question_id_for(ASKED[1].criterion_id): "아니오",
        question_id_for(ASKED[2].criterion_id): "예",
    }


# --- merge 규약 ---------------------------------------------------------------


def test_merge_keeps_existing_order_and_appends_new():
    a, b, c = ALL_CRITERIA[:3]
    existing = [
        CriterionVerdict(criterion_id=a.criterion_id, state=VerdictState.UNMET, rationale="x"),
        CriterionVerdict(criterion_id=b.criterion_id, state=VerdictState.UNMET, rationale="x"),
    ]
    fresh = [
        CriterionVerdict(criterion_id=c.criterion_id, state=VerdictState.MET, rationale="new"),
        CriterionVerdict(criterion_id=a.criterion_id, state=VerdictState.MET, rationale="new"),
    ]

    merged = merge_verdicts(existing, fresh)
    assert [v.criterion_id for v in merged] == [a.criterion_id, b.criterion_id, c.criterion_id]
    assert merged[0].state is VerdictState.MET


# --- 실제 그래프에서 중단·재개 ------------------------------------------------


def test_interrupt_and_resume_in_a_real_graph(monkeypatch, reverify):
    """★ 카드 완료 조건 — 진짜로 멈췄다가 답을 주입해 재개하면 verdict가 갱신된다."""
    builder = StateGraph(GraphState)
    builder.add_node("delta_interview", delta_interview)
    builder.add_edge(START, "delta_interview")
    builder.add_edge("delta_interview", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "interview-1"}}
    first = graph.invoke(base_state(), config)

    # 멈췄다 — 질문 페이로드가 밖으로 나온다.
    assert "__interrupt__" in first
    payload = first["__interrupt__"][0].value
    assert payload["kind"] == "delta_interview"
    assert len(payload["questions"]) == len(ASKED)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("delta_interview",)
    assert not snapshot.values.get("verdicts"), "중단 시점엔 아직 판정이 없다"

    # 답을 주입해 재개한다.
    answers = {q["question_id"]: "예" for q in payload["questions"]}
    final = graph.invoke(Command(resume=answers), config)

    assert {v.criterion_id for v in final["verdicts"]} == {c.criterion_id for c in ASKED}
    assert all(v.state is VerdictState.MET for v in final["verdicts"])
    assert final["interview_round"] == 1
    assert graph.get_state(config).next == ()

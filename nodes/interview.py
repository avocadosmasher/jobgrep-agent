"""델타 인터뷰 노드 — 그래프를 중단시키는 첫 노드 (설계도 §9-3 H3).

`verify_criteria`가 판정하지 못하고 질문으로 승격한 기준들을 사용자에게 **한 번에**
묻고, 받은 답으로 재판정한다.

**목적은 대화가 아니라 미결 기준의 해소다.** 그래서 세 가지를 지킨다:

1. **질문이 없으면 묻지 않는다** — 노드를 그냥 건너뛴다.
2. **배치로 묻는다** — 미결 기준 전체를 한 `interrupt()`에 실어 보낸다. 하나씩
   물으면 라운드마다 재개·재실행 비용이 붙고 사용자도 지친다(§9-3 배치 원칙).
3. **최대 2라운드에서 끊는다** — "질문 → 답변 → 재판정 → 또 질문"은 무한히 돌 수
   있다. 완벽한 판정보다 **끝나는 것**이 중요하고, 판단 못 한 것은 지어내지 말고
   공백으로 고지하면 된다(§12-4).
"""

from __future__ import annotations

from langgraph.types import interrupt

from contracts.enums import VerdictState
from contracts.models import Criterion, CriterionVerdict, Question
from contracts.state import GraphState
from tools.verify import criterion_id_from_question, question_id_for, verify_criteria

# §12-4 — 1라운드는 전체 배치, 2라운드는 1라운드 답변이 모호했던 것만. 그 뒤로는 없다.
MAX_ROUNDS = 2

# 상한에 걸려 판정을 포기한 기준에 붙는 사유. 사용자가 "왜 비었나"를 알 수 있어야 한다.
ROUND_LIMIT_RATIONALE = (
    f"델타 인터뷰 {MAX_ROUNDS}라운드 안에 판정 근거를 얻지 못해 판단을 보류했다."
)


def build_interview_payload(questions: list[Question], round_number: int) -> dict:
    """`interrupt()`로 UI에 던질 페이로드. UI가 폼을 그릴 수 있는 최소 정보만 담는다."""
    return {
        "kind": "delta_interview",
        "round": round_number,
        "max_rounds": MAX_ROUNDS,
        "questions": [
            {
                "question_id": q.question_id,
                "criterion_id": q.criterion_id,
                "text": q.text,
                "options": q.options,
            }
            for q in questions
        ],
    }


def criteria_by_id(state: GraphState) -> dict[str, Criterion]:
    return {
        criterion.criterion_id: criterion
        for group in (state.get("criteria") or {}).values()
        for criterion in group
    }


def unresolved_verdicts(questions: list[Question]) -> list[CriterionVerdict]:
    """상한 초과로 남은 기준을 `UNKNOWN`으로 **확정**한다.

    질문을 그냥 버리면 그 기준은 판정도 질문도 없는 상태로 사라진다. 명시적으로
    판단보류를 남겨야 브리프의 공백 고지에 "무엇을 왜 판단하지 못했나"가 실린다.
    """
    return [
        CriterionVerdict(
            criterion_id=q.criterion_id or criterion_id_from_question(q.question_id),
            state=VerdictState.UNKNOWN,
            rationale=ROUND_LIMIT_RATIONALE,
            evidence=[],
        )
        for q in questions
    ]


def merge_verdicts(
    existing: list[CriterionVerdict], fresh: list[CriterionVerdict]
) -> list[CriterionVerdict]:
    """재판정 결과로 기존 판정을 덮어쓴다 — 기준 하나당 판정은 하나뿐이다.

    순서는 기존 목록을 따르고 새로 생긴 것만 뒤에 붙인다. 순서가 흔들리면
    집계 결과는 같아도 diff가 시끄러워져 무엇이 실제로 바뀌었는지 읽기 어렵다.
    """
    by_id = {v.criterion_id: v for v in existing}
    order = [v.criterion_id for v in existing]

    for verdict in fresh:
        if verdict.criterion_id not in by_id:
            order.append(verdict.criterion_id)
        by_id[verdict.criterion_id] = verdict

    return [by_id[cid] for cid in order]


def delta_interview(state: GraphState) -> dict:
    """미결 기준을 배치로 묻고(중단), 받은 답으로 재판정한다.

    반환하는 부분 갱신:
        `verdicts`           — 재판정으로 갱신된 전체 판정 목록
        `pending_questions`  — 아직 미결인 질문(다음 라운드 대상), 없으면 `[]`
        `interview_answers`  — 라운드 누적 답변
        `interview_round`    — 소비한 라운드 수

    `interrupt()`는 이 함수 중간에서 실행을 멈춘다. 재개되면 **노드 처음부터 다시
    실행**되고 `interrupt()`만 답을 반환한다(langgraph 1.2.10 실측, DEVLOG D27).
    그래서 `interrupt()` 앞에는 부작용을 두지 않는다 — 위쪽은 전부 순수 계산이다.
    """
    questions = state.get("pending_questions") or []
    if not questions:
        return {}  # 물을 것이 없으면 건너뛴다

    round_number = state.get("interview_round") or 0

    if round_number >= MAX_ROUNDS:
        # 상한 초과 — 더 묻지 않고 잔여 기준을 판단보류로 확정한다.
        return {
            "verdicts": merge_verdicts(
                state.get("verdicts") or [], unresolved_verdicts(questions)
            ),
            "pending_questions": [],
        }

    answers = dict(state.get("interview_answers") or {})
    replies = interrupt(build_interview_payload(questions, round_number + 1))
    answers.update(_normalize_replies(replies, questions))

    lookup = criteria_by_id(state)
    asked = [
        lookup[cid]
        for q in questions
        if (cid := q.criterion_id or criterion_id_from_question(q.question_id)) in lookup
    ]

    profile = state.get("profile")
    if not asked or profile is None:
        # 물어볼 기준을 상태에서 못 찾으면 재판정할 대상이 없다 — 라운드만 소비하고
        # 잔여를 보류로 확정한다. 조용히 질문을 버리지 않는다.
        return {
            "verdicts": merge_verdicts(
                state.get("verdicts") or [], unresolved_verdicts(questions)
            ),
            "pending_questions": [],
            "interview_answers": answers,
            "interview_round": round_number + 1,
        }

    fresh, still_open = verify_criteria(asked, profile, answers)

    return {
        "verdicts": merge_verdicts(state.get("verdicts") or [], fresh),
        "pending_questions": still_open,
        "interview_answers": answers,
        "interview_round": round_number + 1,
    }


def _normalize_replies(replies, questions: list[Question]) -> dict[str, str]:
    """UI가 돌려준 답변을 `{question_id: 답변}`으로 정규화한다.

    dict면 그대로 쓰고(키가 criterion_id여도 question_id로 맞춰준다), 목록이면
    질문 순서에 맞춰 짝짓는다. 빈 답변은 답하지 않은 것으로 보고 버린다 —
    빈 문자열을 근거로 넘기면 재판정이 근거 없는 판정을 내릴 빌미가 된다.
    """
    if replies is None:
        return {}

    if isinstance(replies, dict):
        valid = {q.question_id for q in questions}
        normalized: dict[str, str] = {}
        for key, value in replies.items():
            if not str(value).strip():
                continue
            qid = key if key in valid else question_id_for(key)
            normalized[qid] = str(value).strip()
        return normalized

    if isinstance(replies, (list, tuple)):
        return {
            q.question_id: str(value).strip()
            for q, value in zip(questions, replies)
            if str(value).strip()
        }

    # 단일 값 — 질문이 하나일 때만 의미가 있다.
    text = str(replies).strip()
    return {questions[0].question_id: text} if text and len(questions) == 1 else {}

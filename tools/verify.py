"""기준 판정 — 판정 불가는 억지로 판정하지 않고 질문으로 승격한다 (설계도 §7-2 2단계).

이 모듈의 규칙 대부분은 **모델이 아니라 코드가** 집행한다:
    - 근거 인용(quote)이 프로필/인터뷰 답변에 실제로 있는지 대조
    - 근거 없는 충족 주장(MET·PARTIAL)은 판정으로 인정하지 않고 질문으로 승격
    - 모델이 빠뜨린 기준도 질문으로 승격 → 입력 기준은 하나도 증발하지 않는다
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from contracts.enums import VerdictState
from contracts.models import (
    Criterion,
    CriterionVerdict,
    Evidence,
    ProfileJSON,
    Question,
)
from llm.client import DEFAULT_INSTRUCTIONS, complete_structured

QUESTION_ID_PREFIX = "q-"
PROFILE_SOURCE = "프로필"
INTERVIEW_SOURCE = "델타 인터뷰"
DEFAULT_OPTIONS = ["예", "아니오", "잘 모르겠음"]

# 근거 없이도 인정되는 판정 — 부재 자체가 근거인 경우에 한한다.
EVIDENCE_OPTIONAL_STATES = {VerdictState.UNMET}


class JudgedCriterion(BaseModel):
    criterion_id: str
    state: VerdictState
    rationale: str
    quote: str | None


class UndecidableCriterion(BaseModel):
    criterion_id: str
    rationale: str
    question: str
    options: list[str] | None


class VerificationResult(BaseModel):
    judged: list[JudgedCriterion]
    undecidable: list[UndecidableCriterion]


def question_id_for(criterion_id: str) -> str:
    return f"{QUESTION_ID_PREFIX}{criterion_id}"


def criterion_id_from_question(question_id: str) -> str:
    return question_id.removeprefix(QUESTION_ID_PREFIX)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def evidence_corpus(
    profile: ProfileJSON, answers: dict[str, str] | None = None
) -> dict[str, Evidence]:
    """인용 가능한 원문 → 붙일 Evidence 매핑.

    프로필 근거는 **원본 Evidence를 그대로 재사용**한다(출처명·URL·수집일 보존).
    역량명만 인용된 경우와 인터뷰 답변은 코드가 Evidence를 새로 만든다.
    """
    corpus: dict[str, Evidence] = {}

    for comp in profile.competencies:
        for ev in comp.evidence:
            corpus.setdefault(_normalize(ev.quote), ev)
        corpus.setdefault(
            _normalize(comp.name),
            Evidence(
                source_name=PROFILE_SOURCE,
                quote=comp.name,
                collected_at=profile.built_at,
            ),
        )

    for answer in (answers or {}).values():
        text = answer.strip()
        if text:
            corpus.setdefault(
                _normalize(text),
                Evidence(
                    source_name=INTERVIEW_SOURCE,
                    quote=text,
                    collected_at=date.today(),
                ),
            )

    return corpus


def match_evidence(quote: str | None, corpus: dict[str, Evidence]) -> Evidence | None:
    """인용이 원문에 실제로 존재할 때만 Evidence를 돌려준다. 아니면 None."""
    if not quote:
        return None
    needle = _normalize(quote)
    if not needle:
        return None
    if needle in corpus:
        return corpus[needle]
    for source, evidence in corpus.items():  # 원문의 일부만 인용한 경우
        if needle in source:
            return evidence
    return None


def build_verification_prompt(
    criteria: list[Criterion],
    profile: ProfileJSON,
    answers: dict[str, str] | None = None,
) -> str:
    """기준 전체를 한 프롬프트에 묶는다 — 기준별 개별 호출 금지 (§8-4)."""
    profile_lines = []
    for comp in profile.competencies:
        level = comp.level.value if comp.level else "명시 없음"
        quotes = " / ".join(ev.quote for ev in comp.evidence) or "-"
        profile_lines.append(f"- {comp.name} (레벨: {level}) | 원문: {quotes}")

    criteria_lines = [
        f"- criterion_id={c.criterion_id} | {c.text}" for c in criteria
    ]

    answer_block = ""
    if answers:
        answered = [
            f"- criterion_id={criterion_id_from_question(qid)} | 답변: {text.strip()}"
            for qid, text in answers.items()
            if text and text.strip()
        ]
        if answered:
            joined = "\n".join(answered)
            answer_block = f"""
<interview_answers>
{joined}
</interview_answers>
이전 라운드에서 판정 불가였던 기준에 대한 본인의 답변이다. 해당 기준은 이 답변을
근거로 다시 판정하고, 인용(quote)에는 답변 문장을 그대로 쓴다.
"""

    return f"""아래 프로필을 근거로 각 기준의 충족 여부를 판정해라.

규칙
1. 판정한 기준은 judged에, **프로필로 판단할 수 없는 기준은 undecidable에** 넣는다.
   추측해서 억지로 판정하지 않는다. 모르는 것을 모른다고 하는 편이 낫다.
2. 모든 기준은 judged와 undecidable 중 **정확히 한쪽에만** 넣는다.
3. quote는 <profile> 또는 <interview_answers> 안에 **실제로 있는 문장을 그대로** 인용한다.
   지어낸 인용은 코드 대조에서 걸러져 판정 자체가 무효가 된다.
4. 충족({VerdictState.MET.value})·부분({VerdictState.PARTIAL.value}) 판정에는 quote가 반드시 있어야 한다.
   프로필에 관련 내용이 아예 없어서 미충족({VerdictState.UNMET.value})인 경우에만 quote를 비운다.
5. undecidable의 question은 사용자에게 그대로 보여줄 한 문장 질문이다.
   "무엇을 알면 판정할 수 있는가"를 묻는다.
6. rationale은 왜 그렇게 판정했는지 한 문장으로 적는다. 비워두지 않는다.

{DEFAULT_INSTRUCTIONS}

<profile 기준일={profile.built_at}>
{chr(10).join(profile_lines)}
</profile>
{answer_block}
<criteria>
{chr(10).join(criteria_lines)}
</criteria>"""


def to_verdicts(
    criteria: list[Criterion],
    profile: ProfileJSON,
    result: VerificationResult,
    answers: dict[str, str] | None = None,
) -> tuple[list[CriterionVerdict], list[Question]]:
    """모델 산출을 (판정, 질문)으로 가른다. 입력 기준은 반드시 한쪽에 배정된다.

    질문으로 승격되는 경우:
        - 모델이 undecidable에 넣음
        - 모델이 그 기준을 아예 빠뜨림 (누락을 조용히 삼키지 않는다)
        - 판정 상태가 UNKNOWN (판단보류 = 물어봐야 한다는 뜻)
        - MET·PARTIAL인데 인용이 원문 대조를 통과하지 못함 (근거 없는 충족 주장)
        - rationale이 비어 있음
    UNMET은 근거가 없어도 판정으로 남는다 — "프로필에 없다"는 부재 자체가 근거다.
    같은 criterion_id가 양쪽에 있으면 **질문 쪽을 택한다**(억지 판정 금지 원칙).
    """
    corpus = evidence_corpus(profile, answers)
    judged = {j.criterion_id: j for j in result.judged}
    undecidable = {u.criterion_id: u for u in result.undecidable}

    verdicts: list[CriterionVerdict] = []
    questions: list[Question] = []
    seen: set[str] = set()

    for criterion in criteria:
        if criterion.criterion_id in seen:
            continue
        seen.add(criterion.criterion_id)

        item = judged.get(criterion.criterion_id)
        ask = undecidable.get(criterion.criterion_id)

        if item is not None and ask is None:
            evidence = match_evidence(item.quote, corpus)
            rationale = item.rationale.strip()
            has_ground = evidence is not None or item.state in EVIDENCE_OPTIONAL_STATES

            if rationale and has_ground and item.state is not VerdictState.UNKNOWN:
                verdicts.append(
                    CriterionVerdict(
                        criterion_id=criterion.criterion_id,
                        state=item.state,
                        rationale=rationale,
                        evidence=[evidence] if evidence else [],
                    )
                )
                continue

        questions.append(
            Question(
                question_id=question_id_for(criterion.criterion_id),
                criterion_id=criterion.criterion_id,
                text=(ask.question.strip() if ask and ask.question.strip() else
                      f"{criterion.text} — 해당하는 경험이 있습니까?"),
                options=(ask.options if ask and ask.options else list(DEFAULT_OPTIONS)),
            )
        )

    return verdicts, questions


def verify_criteria(
    criteria: list[Criterion],
    profile: ProfileJSON,
    answers: dict[str, str] | None = None,
) -> tuple[list[CriterionVerdict], list[Question]]:
    """기준 목록을 프로필(및 인터뷰 답변)로 판정하고, 판정 불가 기준은 질문으로 승격한다.

    입력: 판정 대상 Criterion 목록, 사용자 ProfileJSON, 이전 델타 인터뷰 라운드의
        answers(question_id → 답변 텍스트, 없으면 None).
    출력: (판정된 CriterionVerdict 목록, 판정 불가 기준에 대한 Question 목록)의 튜플 —
        입력으로 받은 각 Criterion은 반드시 둘 중 한쪽에만 속한다.
    불변식: 배치 호출(기준이 몇 개든 1콜). CriterionVerdict.evidence의 quote는 profile
        내 실제 원문 인용이어야 하며, 근거 없이는 MET/PARTIAL 판정을 내리지 않고
        Question으로 승격한다.
    """
    if not criteria:
        return [], []

    result = complete_structured(
        build_verification_prompt(criteria, profile, answers), VerificationResult
    )
    return to_verdicts(criteria, profile, result, answers)

"""T03 · aggregate_states 검증.

R5에 따라 판정 데이터는 전부 `fixtures/`의 골든 세트에서 온다. 경계 조건
(PARTIAL·UNKNOWN·우대 기준)은 픽스처를 mock으로 대체하지 않고, 픽스처에서
읽은 모델의 상태 필드만 바꾼 파생본으로 만든다.
"""

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from contracts.enums import MatchState, VerdictState
from contracts.models import Criterion, CriterionVerdict
from tools.aggregate import aggregate_states

FIXTURES = Path(__file__).parent.parent / "fixtures"

CRITERIA_BY_COMP: dict[str, list[Criterion]] = TypeAdapter(
    dict[str, list[Criterion]]
).validate_json((FIXTURES / "criteria_sample.json").read_bytes())

COMP_IDS = list(CRITERIA_BY_COMP)

# 픽스처 파일 → 모든 역량이 공통으로 산출해야 하는 등급 (T03 완료 조건)
EXPECTED_BY_FIXTURE = {
    "verdicts_all_met.json": MatchState.MET,
    "verdicts_half.json": MatchState.ADJACENT,
    "verdicts_mostly_unmet.json": MatchState.UNMET,
}


def load_verdicts(filename: str) -> list[CriterionVerdict]:
    raw = (FIXTURES / filename).read_bytes()
    return TypeAdapter(list[CriterionVerdict]).validate_json(raw)


def verdicts_for(filename: str, comp_id: str) -> list[CriterionVerdict]:
    """해당 역량의 기준에 대응하는 판정만 골라낸다."""
    ids = {c.criterion_id for c in CRITERIA_BY_COMP[comp_id]}
    return [v for v in load_verdicts(filename) if v.criterion_id in ids]


def with_states(
    verdicts: list[CriterionVerdict], mapping: dict[VerdictState, VerdictState]
) -> list[CriterionVerdict]:
    """판정 상태만 치환한 파생본. 원본은 건드리지 않는다."""
    return [v.model_copy(update={"state": mapping.get(v.state, v.state)}) for v in verdicts]


# --- 완료 조건: 픽스처 3종 → MET / ADJACENT / UNMET -------------------------


@pytest.mark.parametrize("filename,expected", EXPECTED_BY_FIXTURE.items())
@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_golden_fixtures_produce_expected_state(comp_id, filename, expected):
    state = aggregate_states(CRITERIA_BY_COMP[comp_id], load_verdicts(filename))
    assert state is expected


def test_verdicts_of_other_competencies_are_ignored():
    """전체 판정 목록을 넘겨도 해당 역량만 걸러낸 결과와 같아야 한다."""
    for filename in EXPECTED_BY_FIXTURE:
        for comp_id in COMP_IDS:
            criteria = CRITERIA_BY_COMP[comp_id]
            assert aggregate_states(criteria, load_verdicts(filename)) is aggregate_states(
                criteria, verdicts_for(filename, comp_id)
            )


# --- 경계값 ----------------------------------------------------------------


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_exactly_half_met_is_adjacent(comp_id):
    """'절반 이상'은 절반을 포함한다 — verdicts_half는 정확히 4건 중 2건 충족."""
    verdicts = verdicts_for("verdicts_half.json", comp_id)
    met = [v for v in verdicts if v.state is VerdictState.MET]
    assert len(met) * 2 == len(verdicts), "픽스처 전제가 깨졌다 (정확히 절반이어야 함)"
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], verdicts) is MatchState.ADJACENT


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_one_unmet_among_required_is_not_met(comp_id):
    """필수 기준 하나만 미충족이어도 MET이 아니다."""
    verdicts = verdicts_for("verdicts_all_met.json", comp_id)
    downgraded = [verdicts[0].model_copy(update={"state": VerdictState.UNMET})] + verdicts[1:]
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], downgraded) is MatchState.ADJACENT


# --- PARTIAL = 0.5 ---------------------------------------------------------


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_partial_counts_as_half(comp_id):
    """1충족+3미충족(=UNMET)에서 미충족만 부분으로 바꾸면 0.625 → ADJACENT."""
    verdicts = verdicts_for("verdicts_mostly_unmet.json", comp_id)
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], verdicts) is MatchState.UNMET

    partial = with_states(verdicts, {VerdictState.UNMET: VerdictState.PARTIAL})
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], partial) is MatchState.ADJACENT


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_all_partial_is_adjacent_not_met(comp_id):
    verdicts = verdicts_for("verdicts_all_met.json", comp_id)
    partial = with_states(verdicts, {VerdictState.MET: VerdictState.PARTIAL})
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], partial) is MatchState.ADJACENT


# --- UNKNOWN은 분모에서 제외 ------------------------------------------------


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_unknown_is_excluded_from_denominator(comp_id):
    """2충족+2미충족(ADJACENT)에서 미충족을 판단보류로 바꾸면 2/2 → MET."""
    verdicts = verdicts_for("verdicts_half.json", comp_id)
    unknown = with_states(verdicts, {VerdictState.UNMET: VerdictState.UNKNOWN})
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], unknown) is MatchState.MET


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_missing_verdict_behaves_like_unknown(comp_id):
    """판정 목록에 없는 기준(= Question으로 승격된 기준)도 분모에서 빠진다."""
    verdicts = verdicts_for("verdicts_half.json", comp_id)
    kept = [v for v in verdicts if v.state is VerdictState.MET]
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], kept) is MatchState.MET


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_all_unknown_falls_back_to_unmet(comp_id):
    """판정 불가는 MatchState에 없다 — 문서화된 결정에 따라 UNMET으로 떨어진다."""
    verdicts = verdicts_for("verdicts_half.json", comp_id)
    unknown = with_states(
        verdicts,
        {VerdictState.MET: VerdictState.UNKNOWN, VerdictState.UNMET: VerdictState.UNKNOWN},
    )
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], unknown) is MatchState.UNMET
    assert aggregate_states(CRITERIA_BY_COMP[comp_id], []) is MatchState.UNMET


# --- 우대(is_required=False) 기준은 등급에 관여하지 않는다 -------------------


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_preferred_criteria_do_not_affect_state(comp_id):
    """미충족 기준을 우대로 내리면 남은 필수 기준만으로 판정한다."""
    verdicts = verdicts_for("verdicts_mostly_unmet.json", comp_id)
    unmet_ids = {v.criterion_id for v in verdicts if v.state is VerdictState.UNMET}
    criteria = [
        c.model_copy(update={"is_required": c.criterion_id not in unmet_ids})
        for c in CRITERIA_BY_COMP[comp_id]
    ]
    assert aggregate_states(criteria, verdicts) is MatchState.MET


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_no_required_criteria_falls_back_to_unmet(comp_id):
    criteria = [c.model_copy(update={"is_required": False}) for c in CRITERIA_BY_COMP[comp_id]]
    assert aggregate_states(criteria, load_verdicts("verdicts_all_met.json")) is MatchState.UNMET


# --- 순수성 ----------------------------------------------------------------


@pytest.mark.parametrize("comp_id", COMP_IDS)
def test_is_pure_and_idempotent(comp_id):
    criteria = CRITERIA_BY_COMP[comp_id]
    verdicts = load_verdicts("verdicts_half.json")
    before = ([c.model_dump() for c in criteria], [v.model_dump() for v in verdicts])

    first = aggregate_states(criteria, verdicts)
    second = aggregate_states(criteria, verdicts)

    assert first is second
    assert ([c.model_dump() for c in criteria], [v.model_dump() for v in verdicts]) == before

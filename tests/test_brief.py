"""T06 · build_strategy_brief 검증.

입력 MatchResult는 `fixtures/brief_expected.json`의 카드에서 되짚어 만든다 — 기대
출력이 골든이므로, 그 카드를 만들어냈을 판정 결과를 입력으로 넣으면 트랙 배정이
픽스처와 같아야 한다 (R5).

LLM은 이 모듈을 거치지 않으므로 온라인 테스트가 없다.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from contracts.enums import Category, Level, MatchState, Track, VerdictState
from contracts.models import (
    BriefMeta,
    CriterionVerdict,
    Evidence,
    MatchResult,
    StrategyBrief,
)
from tools.brief import (
    BAND_MIN_DAYS,
    LONG_TERM,
    MID_TERM,
    RELIABILITY_ESTIMATED,
    RELIABILITY_NORMAL,
    SHORT_TERM,
    TRACK1_EMPTY_NOTICE,
    build_brief_meta,
    build_strategy_brief,
    days_remaining_from,
    effort_band,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
EXPECTED = StrategyBrief.model_validate_json((FIXTURES / "brief_expected.json").read_bytes())

VERDICT_OF: dict[MatchState, VerdictState] = {
    MatchState.MET: VerdictState.MET,
    MatchState.ADJACENT: VerdictState.PARTIAL,
    MatchState.UNMET: VerdictState.UNMET,
}


def result_from_card(card) -> MatchResult:
    """브리프 카드를 만들어냈을 법한 MatchResult로 되돌린다.

    category는 트리아지에 쓰이지 않으므로(트랙 배정은 state·레벨·기간만 본다)
    아무 값이나 넣어도 무방하다.
    """
    return MatchResult(
        comp_id=card.comp_id,
        name=card.name,
        category=Category.D4_ORCHESTRATION,
        required_level=card.required_level,
        my_level=card.my_level,
        state=card.state,
        verdicts=[
            CriterionVerdict(
                criterion_id=f"cr-{card.comp_id}-01",
                state=VERDICT_OF[card.state],
                rationale="픽스처 카드에서 되짚은 판정.",
                evidence=list(card.evidence),
            )
        ],
        is_strength=False,
    )


GOLDEN_RESULTS = [
    result_from_card(card)
    for card in EXPECTED.track1 + EXPECTED.track2 + EXPECTED.track3
]


def make_result(
    comp_id: str,
    state: MatchState,
    *,
    required: Level | None = Level.OPERATED,
    mine: Level | None = None,
    evidence: bool = True,
    verdicts: bool = True,
) -> MatchResult:
    quote = f"{comp_id}에 대한 근거 인용"
    items = (
        [
            CriterionVerdict(
                criterion_id=f"cr-{comp_id}-01",
                state=VERDICT_OF[state],
                rationale="테스트 판정.",
                evidence=(
                    [
                        Evidence(
                            source_name="이력서",
                            quote=quote,
                            collected_at=date(2026, 7, 15),
                        )
                    ]
                    if evidence
                    else []
                ),
            )
        ]
        if verdicts
        else []
    )
    return MatchResult(
        comp_id=comp_id,
        name=f"{comp_id} 역량",
        category=Category.D2_BACKEND,
        required_level=required,
        my_level=mine,
        state=state,
        verdicts=items,
    )


def meta_with(days: int, **kw) -> BriefMeta:
    base = EXPECTED.meta.model_copy(update={"days_remaining": days})
    return base.model_copy(update=kw) if kw else base


# --- 골든: 픽스처와 같은 트랙 배정 -------------------------------------------


def test_golden_results_reproduce_expected_tracks():
    brief = build_strategy_brief(GOLDEN_RESULTS, EXPECTED.meta)

    assert [c.comp_id for c in brief.track1] == [c.comp_id for c in EXPECTED.track1]
    assert [c.comp_id for c in brief.track2] == [c.comp_id for c in EXPECTED.track2]
    assert [c.comp_id for c in brief.track3] == [c.comp_id for c in EXPECTED.track3]
    assert brief.summary_counts == EXPECTED.summary_counts
    assert [c.priority for c in brief.track2] == [c.priority for c in EXPECTED.track2]
    assert all(c.track is Track.SHOWCASE for c in brief.track1)
    assert all(c.track is Track.FILL for c in brief.track2)
    assert all(c.track is Track.DROP for c in brief.track3)


def test_llm_slots_are_left_empty():
    brief = build_strategy_brief(GOLDEN_RESULTS, EXPECTED.meta)

    assert brief.summary_line == ""
    assert brief.culture_fit is None
    assert all(c.body == "" for c in brief.track1 + brief.track2 + brief.track3)


def test_summary_counts_match_generated_cards():
    brief = build_strategy_brief(GOLDEN_RESULTS, EXPECTED.meta)

    counted: dict[MatchState, int] = {}
    for card in brief.track1 + brief.track2 + brief.track3:
        counted[card.state] = counted.get(card.state, 0) + 1
    assert counted == brief.summary_counts


# --- 결정론 ------------------------------------------------------------------


def test_track_assignment_is_deterministic_regardless_of_input_order():
    results = [
        make_result("c-adj-long", MatchState.ADJACENT, mine=None),
        make_result("c-met", MatchState.MET, mine=Level.LED),
        make_result("c-unmet-short", MatchState.UNMET, mine=Level.USED),
        make_result("c-unmet-long", MatchState.UNMET, mine=None),
        make_result("a-adj-short", MatchState.ADJACENT, mine=Level.USED),
    ]
    meta = meta_with(30)

    first = build_strategy_brief(results, meta)
    reversed_order = build_strategy_brief(list(reversed(results)), meta)
    rotated = build_strategy_brief(results[2:] + results[:2], meta)

    assert first == reversed_order == rotated


def test_same_input_gives_same_output():
    assert build_strategy_brief(GOLDEN_RESULTS, EXPECTED.meta) == build_strategy_brief(
        GOLDEN_RESULTS, EXPECTED.meta
    )


# --- 공백 케이스 ① 트랙1이 빔 (§11-6) ---------------------------------------


def test_empty_track1_quotes_track2_top_without_relabeling():
    results = [
        make_result("adj-01", MatchState.ADJACENT, mine=Level.USED),
        make_result("unmet-01", MatchState.UNMET, mine=Level.USED),
    ]
    brief = build_strategy_brief(results, meta_with(60))

    assert brief.track1 == [], "미달 항목을 강점으로 재라벨링하지 않는다"
    notice = next(g for g in brief.gaps if g.startswith(TRACK1_EMPTY_NOTICE))
    assert brief.track2[0].name in notice, "트랙2 최우선 항목을 인용한다"
    assert "우선순위 1" in notice


def test_empty_track1_and_track2_still_states_the_fixed_notice():
    brief = build_strategy_brief(
        [make_result("unmet-long", MatchState.UNMET, mine=None)], meta_with(10)
    )

    assert brief.track1 == []
    assert TRACK1_EMPTY_NOTICE in brief.gaps


# --- 공백 케이스 ② 트랙이 빔 (정상 상태) ------------------------------------


def test_empty_tracks_stay_empty_and_counts_are_empty():
    brief = build_strategy_brief([], EXPECTED.meta)

    assert (brief.track1, brief.track2, brief.track3) == ([], [], [])
    assert brief.summary_counts == {}
    assert brief.meta == EXPECTED.meta


def test_no_giveup_track_when_everything_fits():
    results = [make_result("adj-01", MatchState.ADJACENT, mine=Level.USED)]
    brief = build_strategy_brief(results, meta_with(365))

    assert brief.track3 == [], "포기할 것이 없는 것은 정상 상태다"
    assert len(brief.track2) == 1


# --- 공백 케이스 ③ 근거 없는 항목은 카드 미생성 (§11-2 ②) -------------------


@pytest.mark.parametrize("state", [MatchState.MET, MatchState.ADJACENT])
def test_met_or_adjacent_without_evidence_is_dropped(state):
    brief = build_strategy_brief(
        [make_result("no-ev", state, mine=Level.LED, evidence=False)], meta_with(60)
    )

    assert brief.track1 + brief.track2 + brief.track3 == []
    assert any("카드 미생성" in g for g in brief.gaps)


def test_unmet_without_evidence_still_gets_a_card():
    brief = build_strategy_brief(
        [make_result("unmet", MatchState.UNMET, mine=Level.USED, evidence=False)],
        meta_with(60),
    )

    (card,) = brief.track2
    assert card.evidence == [], "부재 자체가 근거인 경우다"


def test_result_without_verdicts_is_dropped():
    brief = build_strategy_brief(
        [make_result("nojudge", MatchState.UNMET, verdicts=False)], meta_with(60)
    )

    assert brief.track1 + brief.track2 + brief.track3 == []
    assert any("판정된 기준이 없어" in g for g in brief.gaps)


def test_duplicate_evidence_is_deduped_on_the_card():
    result = make_result("dup", MatchState.MET, mine=Level.LED)
    result.verdicts.append(result.verdicts[0].model_copy())

    (card,) = build_strategy_brief([result], meta_with(60)).track1

    assert len(card.evidence) == 1


# --- 트리아지 규칙 -----------------------------------------------------------


def test_adjacent_ranks_above_unmet_and_short_above_mid():
    """정렬은 상태(인접 먼저) → 구간(단기 먼저) 순. comp_id 사전순이 아니다."""
    results = [
        make_result("a-unmet-short", MatchState.UNMET, mine=Level.USED),
        make_result("b-adj-mid", MatchState.ADJACENT, required=Level.LED, mine=Level.LEARNED),
        make_result("z-adj-short", MatchState.ADJACENT, mine=Level.USED),
    ]
    brief = build_strategy_brief(results, meta_with(60))

    assert [effort_band(r) for r in results] == [SHORT_TERM, MID_TERM, SHORT_TERM]
    assert [c.comp_id for c in brief.track2] == [
        "z-adj-short",
        "b-adj-mid",
        "a-unmet-short",
    ]
    assert [c.priority for c in brief.track2] == [1, 2, 3]


def test_exceeding_strength_comes_first_in_track1():
    results = [
        make_result("met-equal", MatchState.MET, mine=Level.OPERATED),
        make_result("met-exceed", MatchState.MET, mine=Level.LED),
    ]
    brief = build_strategy_brief(results, meta_with(60))

    assert [c.comp_id for c in brief.track1] == ["met-exceed", "met-equal"]


def test_adjacent_never_falls_to_giveup_track():
    """인접은 남은 기간이 짧아도 트랙2에 남는다 (카드 규칙)."""
    brief = build_strategy_brief(
        [make_result("adj", MatchState.ADJACENT, mine=None)], meta_with(1)
    )

    assert len(brief.track2) == 1
    assert brief.track3 == []


@pytest.mark.parametrize(
    "days,expected_track3",
    [(BAND_MIN_DAYS[LONG_TERM], False), (BAND_MIN_DAYS[LONG_TERM] - 1, True)],
)
def test_unmet_moves_to_giveup_when_band_exceeds_remaining_days(days, expected_track3):
    brief = build_strategy_brief(
        [make_result("unmet-long", MatchState.UNMET, mine=None)], meta_with(days)
    )

    assert bool(brief.track3) is expected_track3
    assert bool(brief.track2) is not expected_track3


def test_effort_band_shrinks_for_adjacent():
    adjacent = make_result("adj", MatchState.ADJACENT, mine=Level.LEARNED)
    unmet = make_result("unmet", MatchState.UNMET, mine=Level.LEARNED)

    assert effort_band(adjacent) == SHORT_TERM
    assert effort_band(unmet) == MID_TERM
    assert effort_band(make_result("none", MatchState.UNMET, mine=None)) == LONG_TERM


# --- 메타: days_remaining 은 코드 계산 ---------------------------------------


def test_days_remaining_is_computed_from_target_date():
    today = date(2026, 8, 3)
    assert days_remaining_from(date(2026, 9, 15), today) == 43
    assert days_remaining_from(today, today) == 0
    assert days_remaining_from(today - timedelta(days=5), today) == 0, "지난 날짜는 0"


def test_build_brief_meta_fills_days_and_reliability():
    today = date(2026, 8, 3)
    meta = build_brief_meta(
        company="테크노베이션",
        role="백엔드 엔지니어",
        selected_jobs=["jd-backend-001"],
        target_date=date(2026, 9, 15),
        source_coverage=0.75,
        missing_sources=["기술블로그", "인재상"],
        today=today,
    )

    assert meta.days_remaining == EXPECTED.meta.days_remaining
    assert meta.reliability == RELIABILITY_NORMAL


def test_missing_jd_downgrades_reliability():
    meta = build_brief_meta(
        company="테크노베이션",
        role="백엔드 엔지니어",
        selected_jobs=[],
        target_date=date(2026, 9, 15),
        source_coverage=0.2,
        missing_sources=["JD"],
        today=date(2026, 8, 3),
    )

    assert meta.reliability == RELIABILITY_ESTIMATED


def test_missing_sources_are_reported_in_gaps():
    brief = build_strategy_brief(GOLDEN_RESULTS, EXPECTED.meta)

    assert any(
        all(source in gap for source in EXPECTED.meta.missing_sources)
        for gap in brief.gaps
    ), "미수집 소스는 공백 고지에 남아야 한다"

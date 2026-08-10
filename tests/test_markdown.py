"""T07 · Markdown 렌더러 검증.

입력은 골든 픽스처 `fixtures/brief_expected.json` 하나다 (R5). 빈 트랙·빈 슬롯
케이스는 그 픽스처를 깎아내 만든다 — 손으로 지어낸 브리프를 쓰지 않는다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from contracts.models import StrategyBrief
from render.markdown import (
    DROP_NOTICE,
    EMPTY_TRACK,
    NOT_COLLECTED,
    NOT_FILLED,
    SECTIONS,
    filename_for,
    render_markdown,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
BRIEF = StrategyBrief.model_validate_json((FIXTURES / "brief_expected.json").read_bytes())

EMPTY_BRIEF = BRIEF.model_copy(
    update={"track1": [], "track2": [], "track3": [], "gaps": [], "summary_counts": {}}
)


# --- 골격: 7개 섹션이 순서대로 -----------------------------------------------


def test_all_seven_sections_are_present_in_order():
    text = render_markdown(BRIEF)

    positions = [text.index(f"## {title}") for title in SECTIONS]
    assert len(SECTIONS) == 7
    assert positions == sorted(positions), "§11-4 골격 순서를 지켜야 한다"


def test_sections_survive_when_everything_is_empty():
    text = render_markdown(EMPTY_BRIEF)

    for title in SECTIONS:
        assert f"## {title}" in text, "빈 브리프에서도 섹션을 없애지 않는다"


def test_empty_tracks_say_not_applicable():
    text = render_markdown(EMPTY_BRIEF)

    assert text.count(EMPTY_TRACK) == 3, "세 트랙 모두 '해당 없음'으로 표기"


def test_drop_notice_only_appears_with_track3_cards():
    notice = DROP_NOTICE.format(days=BRIEF.meta.days_remaining)

    assert notice in render_markdown(BRIEF), "최종 결정은 사용자라는 고지 (§11-4)"
    assert "최종 결정은 사용자" not in render_markdown(EMPTY_BRIEF)


# --- 메타 헤더 ---------------------------------------------------------------


def test_meta_header_carries_every_field():
    text = render_markdown(BRIEF)
    meta = BRIEF.meta

    assert meta.company in text
    assert meta.role in text
    assert meta.selected_jobs[0] in text
    assert f"{meta.days_remaining}일" in text
    assert "75%" in text, "소스 충족률은 퍼센트로 표기"
    assert all(source in text for source in meta.missing_sources)
    assert meta.reliability in text


def test_missing_sources_absent_reads_as_none():
    brief = BRIEF.model_copy(
        update={"meta": BRIEF.meta.model_copy(update={"missing_sources": []})}
    )
    assert "| 미수집 | 없음 |" in render_markdown(brief)


# --- 요약 판정 ---------------------------------------------------------------


def test_summary_counts_are_rendered_for_all_three_states():
    text = render_markdown(BRIEF)

    assert "충족 1" in text
    assert "인접 1" in text
    assert "미보유 1" in text


def test_absent_state_is_rendered_as_zero():
    assert "충족 0" in render_markdown(EMPTY_BRIEF)


# --- 카드 ---------------------------------------------------------------------


def test_every_card_appears_with_state_and_levels():
    text = render_markdown(BRIEF)

    for card in BRIEF.track1 + BRIEF.track2 + BRIEF.track3:
        assert card.name in text
        assert f"**{card.state.value}**" in text
    assert "요구 실무운영 → 보유 설계주도" in text
    assert "요구 실무운영 → 보유 없음" in text, "보유 레벨 없음을 감추지 않는다"


def test_track2_is_rendered_in_priority_order():
    second = BRIEF.track2[0].model_copy(update={"comp_id": "req-be-99", "priority": 2})
    third = BRIEF.track2[0].model_copy(
        update={"comp_id": "req-be-98", "name": "세 번째 항목", "priority": 3}
    )
    shuffled = BRIEF.model_copy(update={"track2": [third, second, BRIEF.track2[0]]})

    text = render_markdown(shuffled)
    assert text.index("우선순위 1") < text.index("우선순위 2") < text.index("우선순위 3")


def test_evidence_is_folded_into_details():
    text = render_markdown(BRIEF)
    quote = BRIEF.track1[0].evidence[0].quote

    assert "<details>" in text
    assert "<summary>근거 1건</summary>" in text
    assert f'"{quote}"' in text
    assert BRIEF.track1[0].evidence[0].source_name in text, "출처명도 함께 남는다"


def test_card_without_evidence_states_the_absence():
    text = render_markdown(BRIEF)

    assert BRIEF.track3[0].evidence == [], "픽스처 전제: 트랙3 카드는 근거가 없다"
    assert "근거: 프로필에서 관련 서술을 찾지 못함" in text


def test_evidence_url_becomes_a_link():
    card = BRIEF.track1[0]
    linked = card.evidence[0].model_copy(update={"url": "https://example.com/resume"})
    brief = BRIEF.model_copy(
        update={"track1": [card.model_copy(update={"evidence": [linked]})]}
    )

    assert "[포트폴리오](https://example.com/resume)" in render_markdown(brief)


# --- ◇ 슬롯이 비어 있을 때 (DEVLOG D13) --------------------------------------


def test_unfilled_slots_are_marked_not_fabricated():
    text = render_markdown(BRIEF)

    assert BRIEF.summary_line == "", "픽스처 전제: ◇ 슬롯이 비어 있다"
    assert NOT_FILLED in text
    assert text.count(NOT_FILLED) == 1 + len(BRIEF.track1 + BRIEF.track2 + BRIEF.track3)


def test_filled_slots_are_rendered_verbatim():
    brief = BRIEF.model_copy(
        update={
            "summary_line": "필수 요건 다수를 이미 충족하고 있다.",
            "culture_fit": "자율성과 실험을 중시하는 조직이다.",
        }
    )
    text = render_markdown(brief)

    assert "필수 요건 다수를 이미 충족하고 있다." in text
    assert "자율성과 실험을 중시하는 조직이다." in text


def test_culture_fit_absence_is_reported_as_not_collected():
    text = render_markdown(BRIEF)

    assert BRIEF.culture_fit is None
    section = text.split(f"## {SECTIONS[5]}")[1]
    assert NOT_COLLECTED in section.split(f"## {SECTIONS[6]}")[0]


# --- 공백 고지 ---------------------------------------------------------------


def test_gaps_are_listed():
    text = render_markdown(BRIEF)

    for gap in BRIEF.gaps:
        assert f"- {gap}" in text


def test_no_gaps_reads_as_none():
    text = render_markdown(EMPTY_BRIEF)
    assert "없음" in text.split(f"## {SECTIONS[6]}")[1]


# --- 파일명 -------------------------------------------------------------------


def test_filename_follows_the_contract_format():
    assert filename_for(BRIEF) == "테크노베이션_백엔드 엔지니어_전략브리프_20260915.md".replace(
        " ", "_"
    )


@pytest.mark.parametrize(
    "company,role",
    [("테크/노베이션", "백엔드 엔지니어"), ("A:B*C", "데이터?엔지니어"), ("  ", "직무")],
)
def test_filename_strips_path_hostile_characters(company, role):
    brief = BRIEF.model_copy(
        update={"meta": BRIEF.meta.model_copy(update={"company": company, "role": role})}
    )
    name = filename_for(brief)

    assert not set(name) & set('\\/:*?"<>| ')
    assert name.endswith(".md")


def test_filename_uses_target_date():
    brief = BRIEF.model_copy(
        update={"meta": BRIEF.meta.model_copy(update={"target_date": date(2026, 1, 2)})}
    )
    assert filename_for(brief).endswith("_20260102.md")


# --- 순수성 -------------------------------------------------------------------


def test_rendering_is_deterministic_and_side_effect_free():
    before = BRIEF.model_dump()

    first = render_markdown(BRIEF)
    second = render_markdown(BRIEF)

    assert first == second
    assert BRIEF.model_dump() == before, "입력 브리프를 변형하지 않는다"

"""Streamlit 카드 렌더러 — `StrategyBrief`를 읽는 **두 번째** 렌더러.

`render/markdown.py`와 같은 객체를 읽는다. 그래서 내용이 갈라질 수 없다 —
갈라진다면 그건 데이터가 아니라 렌더러 둘 중 하나의 결함이다.

## 이 모듈이 두 층으로 나뉜 이유

아래쪽 `render_*`는 Streamlit을 부르므로 값을 돌려주지 않는다. 그 위에 순수 계층
`brief_items()`를 둔 것은 **불변식을 테스트할 수 있게 하기 위해서다.**

> 출력 항목 하나 = UI 카드 하나 = .md 섹션 하나 = 루브릭 평가 단위 하나 (T27)

`st.markdown()` 호출을 가로채 문자열을 비교하는 대신, 두 렌더러가 **무엇을
표시하기로 했는가**를 자료구조로 뽑아 `.md` 본문과 대조한다. 형식과 순서는 달라도
되지만 항목 집합과 그 항목이 말하는 사실은 같아야 한다.

## 보강 데이터 (`matches` · `required`)는 항목을 늘리지 않는다

카드 규격의 "중요도"와 "기준별 판정"은 `BriefCard`에 없다 — 전자는
`CompetencyRecord.importance`, 후자는 `MatchResult.verdicts`에 있고 둘 다
`GraphState`에서 온다. 이것들은 **이미 있는 카드 안의 상세**로만 들어가며 새 항목을
만들지 않는다. `brief_items()`가 이 인자를 아예 받지 않는 것이 그 보장이다.

**유사도 점수는 표시하지 않는다.** 검색기가 애초에 점수를 반환하지 않으며(설계 §7-2,
DEVLOG D39), "매칭 87%" 같은 숫자를 여기서 만들어 붙이면 근거 없는 확신을 파는 셈이 된다.
표시할 근거는 기준별 판정과 원문 인용뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from contracts.enums import Importance, MatchState
from contracts.models import (
    BriefCard,
    CompetencyRecord,
    MatchResult,
    StrategyBrief,
)
from render.markdown import NOT_FILLED

# 트랙 제목은 `.md`와 같은 문구를 쓴다. 화면과 문서에서 다른 이름으로 불리면
# 사용자가 두 산출물을 같은 것으로 못 읽는다.
TRACK_TITLES: tuple[str, str, str] = (
    "트랙1 · 지금 내세울 것",
    "트랙2 · 기간 내 채울 것",
    "트랙3 · 이번엔 포기할 것",
)

NO_LEVEL = "없음"          # 레벨 미상 — `.md`의 "보유 없음"과 같은 표기
NO_CULTURE = "미수집"      # 컬처핏 미수집 — `.md`와 같은 표기

_STATE_ICON: dict[MatchState, str] = {
    MatchState.MET: "🟢",
    MatchState.ADJACENT: "🟡",
    MatchState.UNMET: "🔴",
}


# --- 순수 계층: 표시 항목 -------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """표시 항목 하나 = 루브릭 평가 단위 하나.

    `facts`는 이 항목이 **화면에 내놓는 사실**이다. 형식(굵게·표·아이콘)은 빼고
    값만 담는다 — 형식까지 넣으면 두 렌더러가 같은 내용을 다르게 꾸몄다는 이유로
    parity가 깨져서, 정작 잡아야 할 "항목이 빠졌다"를 못 잡는다.
    """

    key: str
    label: str
    facts: tuple[str, ...]


def track_cards(brief: StrategyBrief) -> list[tuple[str, list[BriefCard]]]:
    """(트랙 제목, 카드들) 셋 — 렌더러와 항목 추출이 같은 순회를 쓰게 하는 지점."""
    return list(zip(TRACK_TITLES, (brief.track1, brief.track2, brief.track3)))


def levels_of(card: BriefCard) -> tuple[str, str]:
    """(요구, 보유). `my_level`이 `None`인 카드는 정상이다 — UNMET이면 후보쌍이
    있어도 비운다(DEVLOG D43). "없음"은 버그가 아니라 판정 결과다."""
    required = card.required_level.value if card.required_level else NO_LEVEL
    mine = card.my_level.value if card.my_level else NO_LEVEL
    return required, mine


def _meta_facts(brief: StrategyBrief) -> tuple[str, ...]:
    meta = brief.meta
    return (
        meta.company,
        meta.role,
        *meta.selected_jobs,
        meta.target_date.isoformat(),
        f"{meta.days_remaining}일",
        f"{meta.source_coverage:.0%}",
        *meta.missing_sources,
        meta.reliability,
    )


def _summary_facts(brief: StrategyBrief) -> tuple[str, ...]:
    counts = tuple(f"{state.value} {n}" for state, n in brief.summary_counts.items())
    return (*counts, brief.summary_line or NOT_FILLED)


def _card_facts(card: BriefCard) -> tuple[str, ...]:
    required, mine = levels_of(card)
    facts = [card.name, card.state.value, required, mine, card.body or NOT_FILLED]
    if card.priority is not None:
        facts.append(str(card.priority))
    # 근거가 없는 카드는 사실을 더하지 않는다. 두 렌더러 모두 "근거 없음"을
    # 알리지만 문구가 서로 다르고, 그 문구는 대조 대상이 아니다.
    facts.extend(ev.quote for ev in card.evidence)
    return tuple(facts)


def brief_items(brief: StrategyBrief) -> list[Item]:
    """이 브리프가 만들어내는 표시 항목 전부. **`brief`만으로 결정된다.**"""
    items = [
        Item("meta", "메타 헤더", _meta_facts(brief)),
        Item("summary", "요약 판정", _summary_facts(brief)),
    ]
    for title, cards in track_cards(brief):
        for card in cards:
            items.append(Item(f"card:{card.comp_id}", title, _card_facts(card)))
    items.append(Item("culture_fit", "컬처핏", (brief.culture_fit or NO_CULTURE,)))
    items.append(Item("gaps", "공백 고지", tuple(brief.gaps)))
    return items


# --- 순수 계층: 배지 -----------------------------------------------------------


def importance_map(required: list[CompetencyRecord] | None) -> dict[str, Importance]:
    """comp_id → 중요도. `BriefCard`에 중요도가 없어 `GraphState.required`에서 가져온다."""
    return {rec.comp_id: rec.importance for rec in (required or [])}


def min_requirement_badge(
    brief: StrategyBrief, required: list[CompetencyRecord] | None = None
) -> tuple[bool, str]:
    """(최소 요건 충족 여부, 설명).

    "최소 요건"은 **필수(REQUIRED) 역량 중 미보유가 없는 상태**다. 중요도를 모르면
    (= `required`를 안 넘기면) 전체 카드로 판정한다 — 기준이 넓어지므로 통과가 더
    어려워질 뿐, 없는 근거로 통과시키지는 않는다.
    """
    importance = importance_map(required)
    cards = [c for _, group in track_cards(brief) for c in group]
    if importance:
        scoped = [c for c in cards if importance.get(c.comp_id) is Importance.REQUIRED]
        label = "필수"
    else:
        scoped = cards
        label = "요구"

    if not scoped:
        return False, f"{label} 역량을 판정하지 못했습니다"

    unmet = [c for c in scoped if c.state is MatchState.UNMET]
    if unmet:
        return False, f"{label} {len(scoped)}건 중 {len(unmet)}건 미보유"
    return True, f"{label} {len(scoped)}건 모두 충족·인접"


# --- Streamlit 계층 ------------------------------------------------------------


def render_badges(
    brief: StrategyBrief, required: list[CompetencyRecord] | None = None
) -> None:
    """상단 배지 — 남은 기간 / 최소 요건 충족 여부.

    **최소 요건이 `st.metric`이 아니라 배너인 이유** — 합불 판정 하나는 숫자 넷 사이에
    끼면 묻힌다. 색이 있는 배너가 "지원 가능한가"라는 질문에 먼저 답한다. 부수적으로
    지표 네 칸이 T09 때 모습 그대로라 T12의 화면 단언(`tests/test_hitl.py`)도 그대로
    산다 — 사용자 판단으로 이 배치를 골랐다.
    """
    meta = brief.meta
    cards = [c for _, group in track_cards(brief) for c in group]

    cols = st.columns(4)
    cols[0].metric("남은 기간", f"{meta.days_remaining}일")
    cols[1].metric("소스 충족률", f"{meta.source_coverage:.0%}")
    cols[2].metric("신뢰등급", meta.reliability)
    cols[3].metric("카드", f"{len(cards)}장")

    ok, detail = min_requirement_badge(brief, required)
    if ok:
        st.success(f"최소 요건 충족 — {detail}")
    else:
        st.warning(f"최소 요건 미충족 — {detail}")


def _render_verdicts(match: MatchResult | None) -> None:
    """기준별 판정. **기본 접힘** — 펼치면 텍스트 늪이 된다."""
    if match is None or not match.verdicts:
        return
    with st.expander(f"기준별 판정 {len(match.verdicts)}건"):
        for verdict in match.verdicts:
            st.markdown(f"**{verdict.state.value}** — {verdict.rationale}")
            for ev in verdict.evidence:
                st.caption(f'"{ev.quote}" — {ev.source_name} ({ev.collected_at})')


def render_card(
    card: BriefCard,
    index: int,
    *,
    importance: Importance | None = None,
    match: MatchResult | None = None,
) -> None:
    """카드 = 역량명 / 중요도 / 3-state 배지 / 내 레벨 vs 요구 / 기준별 판정."""
    required, mine = levels_of(card)
    head = f"**{index}. {card.name}**"
    if importance is not None:
        head += f" · {importance.value}"
    if card.priority is not None:
        head += f" · 우선순위 {card.priority}"

    with st.container(border=True):
        st.markdown(head)
        st.markdown(
            f"{_STATE_ICON[card.state]} **{card.state.value}** · "
            f"요구 {required} → 보유 {mine}"
        )
        st.write(card.body or NOT_FILLED)

        if card.evidence:
            with st.expander(f"근거 {len(card.evidence)}건"):
                for ev in card.evidence:
                    st.markdown(f'- "{ev.quote}" — {ev.source_name} ({ev.collected_at})')
        else:
            st.caption("근거 없음 — 프로필에서 관련 서술을 찾지 못했습니다.")

        _render_verdicts(match)


def render_tracks(
    brief: StrategyBrief,
    *,
    matches: list[MatchResult] | None = None,
    required: list[CompetencyRecord] | None = None,
) -> None:
    by_comp = {m.comp_id: m for m in (matches or [])}
    importance = importance_map(required)

    for title, cards in track_cards(brief):
        st.markdown(f"#### {title} ({len(cards)})")
        if not cards:
            st.write("해당 없음")
            continue
        for index, card in enumerate(cards, start=1):
            render_card(
                card,
                index,
                importance=importance.get(card.comp_id),
                match=by_comp.get(card.comp_id),
            )


def render_culture_fit(brief: StrategyBrief) -> None:
    st.markdown("#### 컬처핏")
    st.write(brief.culture_fit or NO_CULTURE)


def render_gaps(brief: StrategyBrief) -> None:
    """판단하지 못한 항목을 '왜 못 했나'로 남긴다 — 빈 칸으로 지나가지 않는다(D24)."""
    if not brief.gaps:
        return
    with st.expander(f"공백 고지 ({len(brief.gaps)})"):
        for gap in brief.gaps:
            st.write(f"- {gap}")


def render_brief(
    brief: StrategyBrief,
    *,
    matches: list[MatchResult] | None = None,
    required: list[CompetencyRecord] | None = None,
) -> None:
    """화면 진입점 — 요약·배지·트랙·컬처핏·공백 고지.

    `brief_items()`가 세는 항목을 빠짐없이 그린다. 항목을 하나 더 그리려면
    `brief_items()`에도 추가해야 하며, 안 그러면 parity 테스트가 잡는다.
    """
    meta = brief.meta
    st.subheader(f"{meta.company} · {meta.role}")
    st.caption(
        f"선택 공고 {' · '.join(meta.selected_jobs) or '없음'} · "
        f"목표 {meta.target_date.isoformat()}"
        + (f" · 미수집 {' · '.join(meta.missing_sources)}" if meta.missing_sources else "")
    )

    render_badges(brief, required)

    if brief.summary_line:
        st.markdown(f"**{brief.summary_line}**")
    else:
        st.info(f"총평 슬롯이 아직 비어 있습니다 — .md에는 {NOT_FILLED}로 표기됩니다.")

    counts = " · ".join(f"{state.value} {n}" for state, n in brief.summary_counts.items())
    st.caption(counts or "집계된 역량이 없습니다.")

    st.divider()
    render_tracks(brief, matches=matches, required=required)
    st.divider()
    render_culture_fit(brief)
    render_gaps(brief)

"""3트랙 트리아지 — 브리프 골격(■)을 전부 코드가 결정한다 (설계도 §11-3 ~ §11-6).

LLM은 이 모듈을 거치지 않는다. `body`·`summary_line`·`culture_fit`(◇)은 빈 값으로
남으며, 채우는 것은 이후 단계의 몫이다. 구조와 계산만 멱등이라는 §11-1의 주장이
성립하려면 트랙 배정·우선순위가 여기서 결정론적으로 끝나야 한다.

학습 소요는 **절대 시간이 아니라 구간(단기/중기/장기)** 으로만 다룬다(§11-6).
구간을 남은 기간과 비교할 때만 최소 일수를 쓰며, 그 경계값은 실측 전까지 잠정이다.
"""

from __future__ import annotations

from datetime import date

from contracts.enums import Level, MatchState, Track
from contracts.models import (
    BriefCard,
    BriefMeta,
    Evidence,
    MatchResult,
    StrategyBrief,
)

# --- 튜닝 대상 경계값 (설계도 §16 S5·S6 — 실측 후 확정) ---------------------

SHORT_TERM = "단기"
MID_TERM = "중기"
LONG_TERM = "장기"

#: 각 구간을 현실적으로 소화하는 데 필요한 최소 일수. 남은 기간과의 비교에만 쓴다.
BAND_MIN_DAYS: dict[str, int] = {SHORT_TERM: 14, MID_TERM: 30, LONG_TERM: 90}
BAND_ORDER: dict[str, int] = {SHORT_TERM: 0, MID_TERM: 1, LONG_TERM: 2}

#: 4단계 사다리의 순서. 요구/보유 레벨 격차 계산용.
LEVEL_ORDER: dict[Level, int] = {
    Level.LEARNED: 0,
    Level.USED: 1,
    Level.OPERATED: 2,
    Level.LED: 3,
}
#: 요구 레벨이 명시되지 않았을 때의 가정 (실무운영).
DEFAULT_REQUIRED_INDEX = LEVEL_ORDER[Level.OPERATED]
#: 보유 레벨이 없으면 사다리 아래(미보유)로 둔다.
NO_LEVEL_INDEX = -1

RELIABILITY_NORMAL = "정상"
RELIABILITY_ESTIMATED = "추정 기반"

TRACK1_EMPTY_NOTICE = "이 공고 기준을 상회하는 강점 없음"

# 트랙2 정렬 시 상태 우선순위 — 인접이 먼저다(학습비용이 낮으므로, §7-2).
STATE_ORDER: dict[MatchState, int] = {MatchState.ADJACENT: 0, MatchState.UNMET: 1}


def days_remaining_from(target_date: date, today: date | None = None) -> int:
    """목표일까지 남은 일수. 지난 날짜는 0으로 바닥을 친다."""
    return max((target_date - (today or date.today())).days, 0)


def build_brief_meta(
    *,
    company: str,
    role: str,
    selected_jobs: list[str],
    target_date: date,
    source_coverage: float,
    missing_sources: list[str],
    today: date | None = None,
) -> BriefMeta:
    """■ 메타 헤더를 코드 결정론으로 만든다.

    `days_remaining`은 `target_date`에서 계산하며, JD가 한 건도 없으면 신뢰등급을
    "추정 기반"으로 강등한다(설계도 §12-1 하드 요건).
    """
    return BriefMeta(
        company=company,
        role=role,
        selected_jobs=list(selected_jobs),
        target_date=target_date,
        days_remaining=days_remaining_from(target_date, today),
        source_coverage=source_coverage,
        missing_sources=list(missing_sources),
        reliability=RELIABILITY_NORMAL if selected_jobs else RELIABILITY_ESTIMATED,
    )


def _level_index(level: Level | None, default: int) -> int:
    return LEVEL_ORDER[level] if level is not None else default


def level_gap(result: MatchResult) -> int:
    """요구 레벨 - 보유 레벨. 클수록 메워야 할 거리가 멀다."""
    required = _level_index(result.required_level, DEFAULT_REQUIRED_INDEX)
    mine = _level_index(result.my_level, NO_LEVEL_INDEX)
    return required - mine


def exceeds_requirement(result: MatchResult) -> bool:
    """보유 레벨이 요구를 상회하는가 — 트랙1의 강점 판정 기준."""
    if result.my_level is None:
        return False
    required = _level_index(result.required_level, DEFAULT_REQUIRED_INDEX)
    return LEVEL_ORDER[result.my_level] > required


def effort_band(result: MatchResult) -> str:
    """메우는 데 걸리는 **구간**. 절대 시간이 아니라 레벨 격차에서 파생한다.

    인접(ADJACENT)은 격차를 1 깎는다 — "오케스트레이션만 얹으면 되는" 상태이므로
    같은 레벨 격차라도 실제 학습비용이 낮다(설계도 §7-2).
    """
    gap = level_gap(result)
    if result.state is MatchState.ADJACENT:
        gap -= 1
    if gap <= 1:
        return SHORT_TERM
    if gap == 2:
        return MID_TERM
    return LONG_TERM


def fits_in_period(band: str, days_remaining: int) -> bool:
    """이 구간을 남은 기간 안에 시도할 수 있는가."""
    return BAND_MIN_DAYS[band] <= days_remaining


def card_evidence(result: MatchResult) -> list[Evidence]:
    """소판정에 달린 근거를 카드 단위로 모은다 (중복 인용 제거, 순서 보존)."""
    collected: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for verdict in result.verdicts:
        for ev in verdict.evidence:
            key = (ev.source_name, ev.quote)
            if key not in seen:
                seen.add(key)
                collected.append(ev)
    return collected


def is_card_worthy(result: MatchResult) -> bool:
    """카드를 만들 수 있는 항목인가 (설계도 §11-2 ② 근거 부재 → 카드 미생성).

    - 판정된 기준이 하나도 없으면 만들지 않는다. 등급이 아니라 **판단 불가**다
      (T03 `aggregate_states`가 분모 0에서 UNMET을 돌려주는 경우 — DEVLOG D05).
    - 충족·인접인데 인용이 하나도 없으면 만들지 않는다. 근거 없는 충족 주장이다.
      미보유는 "프로필에 없다"는 부재 자체가 근거이므로 인용이 없어도 카드가 된다.
    """
    if not result.verdicts:
        return False
    if result.state in (MatchState.MET, MatchState.ADJACENT):
        return bool(card_evidence(result))
    return True


def _make_card(result: MatchResult, track: Track, priority: int | None) -> BriefCard:
    return BriefCard(
        comp_id=result.comp_id,
        name=result.name,
        track=track,
        state=result.state,
        required_level=result.required_level,
        my_level=result.my_level,
        priority=priority,
        body="",  # ◇ LLM 슬롯 — 이 모듈은 채우지 않는다
        evidence=card_evidence(result),
    )


def build_strategy_brief(
    results: list[MatchResult], meta: BriefMeta
) -> StrategyBrief:
    """집계된 매칭 결과를 3트랙(내세울것/채울것/포기할것) 전략 브리프로 구성한다.

    입력: MatchResult 목록(집계 완료 상태), BriefMeta(코드가 결정론적으로 채운
        메타데이터 — company/role/days_remaining/source_coverage 등).
    출력: StrategyBrief — summary_counts·meta 등 ■ 필드는 코드 결정론으로 채워지고,
        ◇ 필드(`summary_line`·`BriefCard.body`·`culture_fit`)는 **비워둔 채** 반환된다.
        빈 슬롯의 표기는 `""`다(빈 문자열 = 아직 채우지 않음, DEVLOG D03).
    불변식: 트랙 분류·우선순위(priority) 산출은 규칙 기반이며 LLM을 호출하지 않는다.
        같은 입력은 항상 같은 트랙 배정·순서를 낸다.

    트리아지 (설계도 §11-3):
        트랙1 내세울것 — 충족(MET) 항목. 요구 레벨을 상회하는 항목이 앞에 온다.
        트랙2 채울것   — 인접 전부 + 남은 기간에 들어오는 미보유. priority는 1부터.
        트랙3 포기할것 — 소요 구간이 남은 기간을 넘는 미보유.
    트랙1이 비면 미달 항목을 강점으로 재라벨링하지 않고(§11-6), 고정 문구와 트랙2
    최우선 항목을 `gaps`에 **인용**한다 — 새 판정이 아니라 이미 계산된 순위의 인용이다.
    """
    dropped = [r for r in results if not is_card_worthy(r)]
    usable = [r for r in results if is_card_worthy(r)]

    met = [r for r in usable if r.state is MatchState.MET]
    fillable: list[MatchResult] = []
    give_up: list[MatchResult] = []
    for result in usable:
        if result.state is MatchState.ADJACENT:
            fillable.append(result)
        elif result.state is MatchState.UNMET:
            if fits_in_period(effort_band(result), meta.days_remaining):
                fillable.append(result)
            else:
                give_up.append(result)

    met.sort(key=lambda r: (not exceeds_requirement(r), r.comp_id))
    fillable.sort(
        key=lambda r: (STATE_ORDER[r.state], BAND_ORDER[effort_band(r)], r.comp_id)
    )
    give_up.sort(key=lambda r: (BAND_ORDER[effort_band(r)], r.comp_id))

    track1 = [_make_card(r, Track.SHOWCASE, None) for r in met]
    track2 = [
        _make_card(r, Track.FILL, i) for i, r in enumerate(fillable, start=1)
    ]
    track3 = [_make_card(r, Track.DROP, None) for r in give_up]

    summary_counts: dict[MatchState, int] = {}
    for card in track1 + track2 + track3:
        summary_counts[card.state] = summary_counts.get(card.state, 0) + 1

    gaps = [
        f"{r.name}: 판정된 기준이 없어 카드 미생성"
        if not r.verdicts
        else f"{r.name}: 충족 근거 인용이 없어 카드 미생성"
        for r in dropped
    ]
    if meta.missing_sources:
        gaps.append(f"미수집 소스: {'·'.join(meta.missing_sources)} — 관련 판단 보류")
    if not track1:
        notice = TRACK1_EMPTY_NOTICE
        if track2:
            notice += f" — 우선 보완 항목: {track2[0].name}(우선순위 {track2[0].priority})"
        gaps.append(notice)

    return StrategyBrief(
        meta=meta,
        summary_counts=summary_counts,
        summary_line="",  # ◇
        track1=track1,
        track2=track2,
        track3=track3,
        culture_fit=None,  # ◇ (조건부)
        gaps=gaps,
    )

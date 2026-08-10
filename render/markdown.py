"""StrategyBrief → Markdown 직렬화 (설계도 §11-4 골격 순서, §11-5 이중 렌더러).

순수 직렬화다 — LLM을 부르지 않고, 브리프에 없는 판단을 새로 만들지 않는다.
채워지지 않은 ◇ 슬롯은 지어내지 않고 **비어 있다고 표기**한다.

T15(Streamlit 카드 뷰)와 **같은 객체를 읽는다**. 출력 항목 하나 = 카드 하나 =
.md 섹션 하나 = 루브릭 평가 단위 하나라는 §11-5 불변식이 여기서 지켜진다.
"""

from __future__ import annotations

import re

from contracts.enums import MatchState
from contracts.models import BriefCard, Evidence, StrategyBrief

#: §11-4 골격 순서. 렌더러는 이 순서·개수를 바꾸지 않는다.
SECTIONS = (
    "메타 헤더",
    "요약 판정",
    "트랙1 · 지금 내세울 것",
    "트랙2 · 기간 내 채울 것",
    "트랙3 · 이번엔 포기할 것",
    "컬처핏",
    "공백 고지",
)

EMPTY_TRACK = "해당 없음"
NOT_COLLECTED = "미수집"
NOT_FILLED = "_(아직 작성되지 않음)_"
SUMMARY_ORDER = (MatchState.MET, MatchState.ADJACENT, MatchState.UNMET)

#: 트랙3은 배제 판단을 사용자에게 되돌린다 (§11-4).
DROP_NOTICE = "최종 결정은 사용자 몫이다. 아래는 남은 기간 {days}일 안에 메우기 어렵다고 판단한 항목이다."

_FILENAME_UNSAFE = re.compile(r'[\\/:*?"<>|\s]+')


def filename_for(brief: StrategyBrief) -> str:
    """`{회사}_{직무}_전략브리프_{YYYYMMDD}.md`.

    날짜는 **목표 시점(`meta.target_date`)** 이다 — 브리프에 생성 시각 필드가 없고,
    같은 브리프는 언제 내려받아도 같은 파일명이어야 하기 때문이다.
    경로로 쓸 수 없는 문자와 공백은 `_`로 바꾼다(Windows 파일명 규칙).
    """
    company = _safe(brief.meta.company)
    role = _safe(brief.meta.role)
    return f"{company}_{role}_전략브리프_{brief.meta.target_date:%Y%m%d}.md"


def _safe(value: str) -> str:
    return _FILENAME_UNSAFE.sub("_", value.strip()) or "미상"


def _slot(text: str | None) -> str:
    """◇ 슬롯 — 비어 있으면 지어내지 않고 비었다고 적는다 (DEVLOG D03·D13)."""
    return text.strip() if text and text.strip() else NOT_FILLED


def _levels(card: BriefCard) -> str:
    required = card.required_level.value if card.required_level else "명시 없음"
    mine = card.my_level.value if card.my_level else "없음"
    return f"요구 {required} → 보유 {mine}"


def _evidence_block(evidence: list[Evidence]) -> list[str]:
    """근거는 접어서 평탄화한다 — 본문이 텍스트 늪이 되지 않게 (§11-5)."""
    if not evidence:
        return ["- 근거: 프로필에서 관련 서술을 찾지 못함"]

    lines = ["<details>", f"<summary>근거 {len(evidence)}건</summary>", ""]
    for ev in evidence:
        source = f"[{ev.source_name}]({ev.url})" if ev.url else ev.source_name
        lines.append(f'- "{ev.quote}" — {source} ({ev.collected_at:%Y-%m-%d})')
    lines += ["", "</details>"]
    return lines


def _render_card(card: BriefCard, index: int) -> list[str]:
    heading = f"### {index}. {card.name}"
    if card.priority is not None:
        heading += f" (우선순위 {card.priority})"

    lines = [heading, "", f"- 상태: **{card.state.value}** · {_levels(card)}", ""]
    lines.append(_slot(card.body))
    lines.append("")
    lines += _evidence_block(card.evidence)
    lines.append("")
    return lines


def _render_track(title: str, cards: list[BriefCard], note: str = "") -> list[str]:
    lines = [f"## {title}", ""]
    if note:
        lines += [note, ""]
    if not cards:
        lines += [EMPTY_TRACK, ""]
        return lines
    for i, card in enumerate(cards, start=1):
        lines += _render_card(card, i)
    return lines


def render_markdown(brief: StrategyBrief) -> str:
    """전략 브리프를 `.md` 문자열로 직렬화한다.

    §11-4 골격 7개 섹션을 **순서 그대로, 하나도 빠뜨리지 않고** 출력한다.
    빈 트랙은 섹션을 없애지 않고 "해당 없음"을 적는다(§11-2 ③ 정상 상태).
    LLM을 호출하지 않으므로 같은 브리프는 항상 같은 문자열이 된다.
    """
    meta = brief.meta
    lines: list[str] = [f"# {meta.company} {meta.role} 전략 브리프", ""]

    # 1. 메타 헤더
    lines += [
        f"## {SECTIONS[0]}",
        "",
        "| 항목 | 내용 |",
        "| --- | --- |",
        f"| 대상 회사 | {meta.company} |",
        f"| 대상 직무 | {meta.role} |",
        f"| 선택 공고 | {', '.join(meta.selected_jobs) or NOT_COLLECTED} |",
        f"| 목표 시점 | {meta.target_date:%Y-%m-%d} (남은 기간 {meta.days_remaining}일) |",
        f"| 소스 충족률 | {meta.source_coverage:.0%} |",
        f"| 미수집 | {'·'.join(meta.missing_sources) or '없음'} |",
        f"| 신뢰등급 | {meta.reliability} |",
        "",
    ]

    # 2. 요약 판정
    counts = " · ".join(
        f"{state.value} {brief.summary_counts.get(state, 0)}" for state in SUMMARY_ORDER
    )
    lines += [f"## {SECTIONS[1]}", "", f"- 집계: {counts}", "", _slot(brief.summary_line), ""]

    # 3~5. 3트랙
    lines += _render_track(SECTIONS[2], brief.track1)
    lines += _render_track(
        SECTIONS[3], sorted(brief.track2, key=lambda c: (c.priority is None, c.priority))
    )
    lines += _render_track(
        SECTIONS[4],
        brief.track3,
        DROP_NOTICE.format(days=meta.days_remaining) if brief.track3 else "",
    )

    # 6. 컬처핏
    lines += [f"## {SECTIONS[5]}", ""]
    lines += [brief.culture_fit.strip() if brief.culture_fit else NOT_COLLECTED, ""]

    # 7. 공백 고지
    lines += [f"## {SECTIONS[6]}", ""]
    lines += [f"- {gap}" for gap in brief.gaps] if brief.gaps else ["없음"]
    lines.append("")

    return "\n".join(lines)

"""브리프 ◇ 슬롯 채우기 — LLM이 서술형 문장만 쓰고 골격은 못 건드린다.

T06이 만든 ■(구조·계산)는 이 모듈을 통과해도 **한 글자도 바뀌지 않는다.**
그 보장이 §11-1 멱등성 주장의 실체이므로 두 겹으로 지킨다:
    1. 병합은 `model_copy(update={"body": ...})` 로만 한다 — 구조적으로 ◇ 외에는
       손댈 경로가 없다.
    2. 그럼에도 병합 결과의 골격을 원본과 대조하고, 어긋나면 채우기를 통째로 버린다.

LLM이 실패해도 브리프는 나간다 — 부분 실패를 전체 실패로 만들지 않는다(§12).
"""

from __future__ import annotations

from pydantic import BaseModel

from contracts.models import BriefCard, StrategyBrief
from llm.client import DEFAULT_INSTRUCTIONS, LLMError, complete_structured

#: `model_dump`에서 ◇ 슬롯을 뺀 나머지 = 골격(■). 대조 기준이다.
SKELETON_EXCLUDE = {
    "summary_line": True,
    "culture_fit": True,
    "track1": {"__all__": {"body"}},
    "track2": {"__all__": {"body"}},
    "track3": {"__all__": {"body"}},
}

TRACK_GUIDE = """
- 트랙1(지금 내세울 것) : 이력서·자기소개서에 그대로 옮길 수 있는 문구. 근거로 붙은
  인용의 범위 안에서만 쓴다. 인용에 없는 성과·수치를 덧붙이지 않는다.
- 트랙2(기간 내 채울 것) : 무엇을 어떻게 보완할지. **기간을 숫자로 약속하지 않는다**
  ("2주면 됩니다" 금지). 순서와 범위로만 말한다.
- 트랙3(이번엔 포기할 것) : 이번 지원에서 뒤로 미루기를 권하는 사유. 단정하지 않는다.
  최종 결정은 사용자 몫이라는 전제를 깨지 않는다(결정을 대신 내리지 않는다).
""".strip()


class FilledCard(BaseModel):
    comp_id: str
    body: str


class FilledSlots(BaseModel):
    summary_line: str
    cards: list[FilledCard]
    culture_fit: str | None


def skeleton_of(brief: StrategyBrief) -> dict:
    """◇ 슬롯을 제외한 브리프 전체 — 이것이 같으면 골격이 보존된 것이다."""
    return brief.model_dump(exclude=SKELETON_EXCLUDE)


def _card_lines(card: BriefCard, track: str) -> str:
    required = card.required_level.value if card.required_level else "명시 없음"
    mine = card.my_level.value if card.my_level else "없음"
    priority = f" | 우선순위 {card.priority}" if card.priority is not None else ""
    quotes = (
        "\n".join(f"    · \"{ev.quote}\" ({ev.source_name})" for ev in card.evidence)
        or "    · 근거 인용 없음"
    )
    return (
        f"- comp_id={card.comp_id} | 트랙={track} | 상태={card.state.value} | "
        f"요구 {required} → 보유 {mine}{priority}\n"
        f"  역량: {card.name}\n"
        f"  근거:\n{quotes}"
    )


def build_fill_prompt(brief: StrategyBrief) -> str:
    """브리프 전체를 한 프롬프트로 묶는다 — 카드별 개별 호출 금지 (§8-4)."""
    meta = brief.meta
    counts = " · ".join(f"{s.value} {n}" for s, n in brief.summary_counts.items()) or "없음"
    blocks = [
        _card_lines(card, track)
        for track, cards in (
            ("내세울것", brief.track1),
            ("채울것", brief.track2),
            ("포기할것", brief.track3),
        )
        for card in cards
    ]
    cards_block = "\n".join(blocks)
    gaps = "\n".join(f"- {gap}" for gap in brief.gaps) or "- 없음"

    return f"""{meta.company} {meta.role} 지원자의 전략 브리프에 들어갈 **서술형 문장만** 작성해라.
판정·순위·집계는 이미 확정돼 있다. 너는 그것을 **설명**할 뿐 바꾸지 않는다.

상황
- 목표 시점까지 남은 기간: {meta.days_remaining}일
- 집계: {counts}
- 신뢰등급: {meta.reliability} (미수집: {'·'.join(meta.missing_sources) or '없음'})

작성 규칙
1. summary_line — 집계 결과에 근거한 총평 한 줄. 새로운 판정을 만들지 않는다.
2. cards — 각 카드의 body를 트랙 성격에 맞게 쓴다. comp_id는 아래 목록의 값을 그대로 쓰고,
   목록에 없는 comp_id를 만들지 않는다.
{TRACK_GUIDE}
3. culture_fit — 인재상 자료가 근거로 붙어 있을 때만 쓴다. 근거가 없으면 null로 둔다.
   **시험·면접 공략법이나 대응 요령은 쓰지 않는다.** 조직 성향과 지원자의 궁합만 서술한다.
4. 근거(인용)에 없는 사실을 지어내지 않는다. 쓸 말이 없으면 body를 빈 문자열로 둔다.
   빈 칸은 정직한 상태이고, 지어낸 문장은 그렇지 않다.

{DEFAULT_INSTRUCTIONS}

<cards>
{cards_block}
</cards>

<gaps 판단하지 못한 항목>
{gaps}
</gaps>"""


def merge_slots(brief: StrategyBrief, filled: FilledSlots) -> StrategyBrief:
    """채운 문장을 브리프에 병합한다. ◇ 외의 필드는 손대지 않는다.

    - 브리프에 없는 comp_id는 버린다.
    - 모델이 빠뜨렸거나 공백만 준 슬롯은 **원래 값(빈 문자열)을 유지**한다.
      렌더러가 "아직 작성되지 않음"으로 표기한다(DEVLOG D18).
    - 병합 후 골격이 원본과 다르면 채우기를 **통째로 버리고** 원본을 돌려준다.
    """
    bodies = {c.comp_id: c.body.strip() for c in filled.cards if c.body and c.body.strip()}

    def fill(cards: list[BriefCard]) -> list[BriefCard]:
        return [
            card.model_copy(update={"body": bodies[card.comp_id]})
            if card.comp_id in bodies
            else card
            for card in cards
        ]

    culture_fit = filled.culture_fit.strip() if filled.culture_fit else ""
    merged = brief.model_copy(
        update={
            "summary_line": filled.summary_line.strip() or brief.summary_line,
            "culture_fit": culture_fit or brief.culture_fit,
            "track1": fill(brief.track1),
            "track2": fill(brief.track2),
            "track3": fill(brief.track3),
        }
    )

    if skeleton_of(merged) != skeleton_of(brief):
        return brief
    return merged


def fill_brief_slots(brief: StrategyBrief) -> StrategyBrief:
    """브리프의 ◇ 슬롯(`summary_line`·`BriefCard.body`·`culture_fit`)을 배치 1콜로 채운다.

    입력: ■가 확정된 StrategyBrief (T06 `build_strategy_brief` 산출).
    출력: ◇만 채워진 **새** StrategyBrief. 입력 객체는 변형하지 않는다.
    불변식: 골격(■) 전 필드가 입력과 동일하다. LLM 호출은 카드 수와 무관하게 1회.
    실패: `LLMError`(키 부재·파싱 실패 등)가 나면 예외를 전파하지 않고 **입력 브리프를
        그대로 반환**한다. 슬롯이 비어도 브리프는 사용자에게 나가야 한다(§12).
    """
    if not (brief.track1 or brief.track2 or brief.track3):
        return brief  # 채울 카드가 없다 — 부를 이유가 없다

    try:
        filled = complete_structured(build_fill_prompt(brief), FilledSlots)
    except LLMError:
        return brief

    return merge_slots(brief, filled)

"""T15 완료 조건 — 같은 brief를 두 렌더러로 냈을 때 **표시 항목 집합이 같다.**

순서·형식은 달라도 된다. 같아야 하는 것은 ① 항목의 집합과 ② 각 항목이 말하는
사실이다. 그래서 `.md` 문자열을 통째로 비교하지 않고, `render/cards.py`가 표시하기로
한 사실이 `.md` 본문 안에 실제로 있는지를 본다.

**한 방향만 봐도 되는 이유** — 두 렌더러의 입력은 같은 `StrategyBrief` 객체 하나다.
그래서 "`.md`에만 있고 화면엔 없는 항목"은 곧 `brief_items()`가 브리프의 일부를
빠뜨렸다는 뜻이고, 그건 `test_items_cover_the_whole_brief`가 브리프에서 직접 다시
세어 잡는다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from contracts.enums import Category, Importance, Level, MatchState, VerdictState
from contracts.models import (
    BriefCard,
    CompetencyRecord,
    CriterionVerdict,
    Evidence,
    MatchResult,
    StrategyBrief,
)
from render.cards import (
    NO_CULTURE,
    NO_LEVEL,
    brief_items,
    levels_of,
    min_requirement_badge,
)
from render.markdown import NOT_FILLED, render_markdown

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "brief_expected.json"


@pytest.fixture
def brief() -> StrategyBrief:
    return StrategyBrief.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def filled(brief: StrategyBrief) -> StrategyBrief:
    """◇ 슬롯과 컬처핏이 채워진 변형.

    골든 픽스처는 ◇ 슬롯이 빈 상태(D03)라 `NOT_FILLED` 경로만 밟는다. 채워진 값이
    양쪽에 실리는지도 봐야 parity가 "둘 다 빈칸이라 우연히 같다"로 통과하지 않는다.
    """
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["summary_line"] = "핵심 격차는 API 설계 경험이다"
    data["culture_fit"] = "자율과 책임을 강조하는 조직"
    for track in ("track1", "track2", "track3"):
        for i, card in enumerate(data[track]):
            card["body"] = f"{track} 카드 {i} 본문"
    return StrategyBrief.model_validate(data)


def all_cards(brief: StrategyBrief) -> list[BriefCard]:
    return [*brief.track1, *brief.track2, *brief.track3]


# --- 완료 조건 -----------------------------------------------------------------


@pytest.mark.parametrize("which", ["brief", "filled"])
def test_every_ui_fact_appears_in_markdown(which, request):
    """화면이 내놓는 사실은 전부 `.md`에도 있다 — 화면에만 있는 내용은 없다."""
    subject: StrategyBrief = request.getfixturevalue(which)
    md = render_markdown(subject)

    missing = [
        (item.key, fact)
        for item in brief_items(subject)
        for fact in item.facts
        if fact not in md
    ]
    assert not missing, f".md에 없는 화면 표시 사실: {missing}"


@pytest.mark.parametrize("which", ["brief", "filled"])
def test_item_set_matches_markdown_sections(which, request):
    """항목 하나 = `.md` 섹션 하나. 카드 수와 고정 섹션 수가 양쪽에서 같다."""
    subject: StrategyBrief = request.getfixturevalue(which)
    md = render_markdown(subject)
    items = brief_items(subject)

    card_items = [i for i in items if i.key.startswith("card:")]
    # `.md`의 카드 섹션은 `### `로 시작한다 (트랙 제목은 `## `).
    md_card_headings = [line for line in md.splitlines() if line.startswith("### ")]
    assert len(card_items) == len(md_card_headings) == len(all_cards(subject))

    # 고정 섹션 넷: 메타 헤더 · 요약 판정 · 컬처핏 · 공백 고지
    assert {"meta", "summary", "culture_fit", "gaps"} <= {i.key for i in items}
    assert len(items) == len(card_items) + 4


def test_items_cover_the_whole_brief(brief: StrategyBrief):
    """`.md`에만 있고 화면엔 없는 항목이 생기지 않도록, 브리프에서 직접 다시 센다."""
    items = brief_items(brief)
    keys = {i.key for i in items}

    assert {f"card:{c.comp_id}" for c in all_cards(brief)} <= keys

    facts = {fact for item in items for fact in item.facts}
    for card in all_cards(brief):
        assert card.name in facts
        assert card.state.value in facts
        for ev in card.evidence:
            assert ev.quote in facts
    for gap in brief.gaps:
        assert gap in facts
    assert brief.meta.reliability in facts
    assert f"{brief.meta.days_remaining}일" in facts


# --- 항목 집합이 흔들리지 않는다는 증명 -----------------------------------------


def test_enrichment_does_not_change_items(brief: StrategyBrief):
    """중요도·기준별 판정은 `brief_items()`에 닿지 못한다.

    `brief_items()`가 브리프 하나만 받는 것이 그 구조적 보장이라, 이 테스트는
    보강 데이터를 만들어 놓고 **항목 계산에 끼어들 통로가 없음**을 고정한다.
    """
    import inspect

    from render import cards

    params = inspect.signature(cards.brief_items).parameters
    assert list(params) == ["brief"], "brief_items()가 브리프 밖 입력을 받기 시작했다"

    before = brief_items(brief)
    _ = [
        MatchResult(
            comp_id=card.comp_id,
            name=card.name,
            category=Category.D2_BACKEND,
            required_level=card.required_level,
            my_level=card.my_level,
            state=card.state,
            verdicts=[
                CriterionVerdict(
                    criterion_id=f"c-{card.comp_id}",
                    state=VerdictState.MET,
                    rationale="테스트 판정",
                    evidence=[],
                )
            ],
        )
        for card in all_cards(brief)
    ]
    assert brief_items(brief) == before


@pytest.mark.parametrize(
    "drop",
    ["track1", "track2", "track3"],
    ids=["트랙1 카드 누락", "트랙2 카드 누락", "트랙3 카드 누락"],
)
def test_dropping_a_card_breaks_parity(brief: StrategyBrief, drop: str):
    """뮤테이션 — 화면에서 카드를 하나 빠뜨리면 이 테스트들이 실제로 잡는가.

    parity 테스트가 통과만 하고 아무것도 막지 못하는 상태를 방지한다.
    """
    mutated = brief.model_copy(update={drop: []})
    assert len(brief_items(mutated)) < len(brief_items(brief))
    assert len(all_cards(mutated)) != len(
        [line for line in render_markdown(brief).splitlines() if line.startswith("### ")]
    )


def test_a_fact_the_markdown_lacks_is_caught(brief: StrategyBrief):
    """뮤테이션 — 화면이 `.md`에 없는 값을 지어내면 substring 대조가 잡는가."""
    md = render_markdown(brief)
    assert "매칭 87%" not in md  # 유사도 점수는 어느 쪽에도 없어야 한다 (D39, §7-2)


# --- 카드 규격 -----------------------------------------------------------------


def test_levels_render_none_as_no_level(brief: StrategyBrief):
    """`my_level=None`은 정상 표시다 — UNMET이면 후보쌍이 있어도 비운다(D43)."""
    unmet = [c for c in all_cards(brief) if c.state is MatchState.UNMET]
    assert unmet, "픽스처에 UNMET 카드가 없어 이 경로를 못 본다"
    for card in unmet:
        _, mine = levels_of(card)
        assert mine == NO_LEVEL
    assert f"보유 {NO_LEVEL}" in render_markdown(brief)


def test_met_card_shows_both_levels(brief: StrategyBrief):
    met = next(c for c in all_cards(brief) if c.state is MatchState.MET)
    required, mine = levels_of(met)
    assert (required, mine) == (Level.OPERATED.value, Level.LED.value)


def test_empty_slots_are_disclosed_not_hidden(brief: StrategyBrief):
    """빈 ◇ 슬롯은 양쪽 다 '아직 작성되지 않음'으로 드러난다 (D18·D20)."""
    facts = {fact for item in brief_items(brief) for fact in item.facts}
    assert NOT_FILLED in facts
    assert NOT_FILLED in render_markdown(brief)


def test_missing_culture_fit_matches_markdown(brief: StrategyBrief):
    assert brief.culture_fit is None
    item = next(i for i in brief_items(brief) if i.key == "culture_fit")
    assert item.facts == (NO_CULTURE,)
    assert NO_CULTURE in render_markdown(brief)


# --- 상단 배지 -----------------------------------------------------------------


def _required(*pairs: tuple[str, Importance]) -> list[CompetencyRecord]:
    return [
        CompetencyRecord(
            comp_id=comp_id,
            category=Category.D2_BACKEND,
            name=comp_id,
            importance=importance,
        )
        for comp_id, importance in pairs
    ]


def test_min_requirement_fails_when_a_required_card_is_unmet(brief: StrategyBrief):
    unmet = next(c for c in all_cards(brief) if c.state is MatchState.UNMET)
    ok, detail = min_requirement_badge(brief, _required((unmet.comp_id, Importance.REQUIRED)))
    assert ok is False
    assert "미보유" in detail


def test_min_requirement_scopes_to_required_only(brief: StrategyBrief):
    """미보유 카드가 '우대'면 최소 요건은 충족이다 — 범위가 좁혀지는지 본다."""
    unmet = next(c for c in all_cards(brief) if c.state is MatchState.UNMET)
    met = next(c for c in all_cards(brief) if c.state is MatchState.MET)
    ok, detail = min_requirement_badge(
        brief,
        _required((unmet.comp_id, Importance.PREFERRED), (met.comp_id, Importance.REQUIRED)),
    )
    assert ok is True
    assert "필수 1건" in detail


def test_min_requirement_without_importance_uses_all_cards(brief: StrategyBrief):
    """중요도를 모르면 전체 카드로 판정한다 — 없는 근거로 통과시키지 않는다."""
    ok, detail = min_requirement_badge(brief, None)
    assert ok is False
    assert detail.startswith("요구")


def test_min_requirement_with_no_cards():
    empty = StrategyBrief(
        meta=_meta(),
        summary_counts={},
        summary_line="",
        track1=[],
        track2=[],
        track3=[],
    )
    ok, detail = min_requirement_badge(empty, None)
    assert ok is False
    assert "판정하지 못했습니다" in detail


def _meta():
    from contracts.models import BriefMeta

    return BriefMeta(
        company="테크노베이션",
        role="백엔드 엔지니어",
        selected_jobs=[],
        target_date=date(2026, 9, 15),
        days_remaining=43,
        source_coverage=0.0,
        missing_sources=[],
        reliability="추정 기반",
    )


def test_evidence_quotes_are_shown_verbatim(brief: StrategyBrief):
    """인용은 원문 그대로 — 요약·가공 금지(AGENTS.md 고유 규약)."""
    card = next(c for c in all_cards(brief) if c.evidence)
    item = next(i for i in brief_items(brief) if i.key == f"card:{card.comp_id}")
    for ev in card.evidence:
        assert ev.quote in item.facts
        assert ev.quote in render_markdown(brief)


def test_evidence_objects_are_not_invented(brief: StrategyBrief):
    """근거가 없는 카드는 근거 사실도 없다 — 채워 넣지 않는다."""
    bare = next(c for c in all_cards(brief) if not c.evidence)
    item = next(i for i in brief_items(brief) if i.key == f"card:{bare.comp_id}")
    quotes = [f for f in item.facts if f.startswith('"')]
    assert quotes == []


def test_priority_is_carried_for_track2(brief: StrategyBrief):
    card = next(c for c in brief.track2 if c.priority is not None)
    item = next(i for i in brief_items(brief) if i.key == f"card:{card.comp_id}")
    assert str(card.priority) in item.facts
    assert f"(우선순위 {card.priority})" in render_markdown(brief)


def test_evidence_is_typed(brief: StrategyBrief):
    """픽스처가 실제 `Evidence`로 역직렬화됐는지 — 문자열 흉내가 아니다(R5)."""
    card = next(c for c in all_cards(brief) if c.evidence)
    assert all(isinstance(ev, Evidence) for ev in card.evidence)


# --- Streamlit 계층이 실제로 그리는가 -------------------------------------------


class RecordingSt:
    """`st.*` 호출에 실린 문자열을 모으는 대역.

    `AppTest`를 쓰지 않는 이유는 여기서 볼 것이 앱 전체가 아니라 **렌더러가 무엇을
    내놓는가** 하나이기 때문이다. 컨테이너·확장자는 `with` 문에 쓰이므로 자기 자신을
    돌려주고, `columns`만 개수만큼 돌려준다.
    """

    def __init__(self) -> None:
        self.text: list[str] = []

    def _record(self, *args, **kwargs):
        self.text.extend(str(a) for a in args)
        self.text.extend(str(v) for v in kwargs.values())
        return self

    def __getattr__(self, name):
        return self._record

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(count)]

    @property
    def drawn(self) -> str:
        return "\n".join(self.text)


@pytest.mark.parametrize("which", ["brief", "filled"])
def test_streamlit_layer_draws_every_item(which, request, monkeypatch):
    """`brief_items()`가 세는 사실을 화면 계층이 실제로 그린다.

    순수 계층만 검사하면 "항목은 세지만 화면엔 안 그린다"가 통과한다. 두 계층이
    갈라지는 걸 여기서 막는다.
    """
    subject: StrategyBrief = request.getfixturevalue(which)

    from render import cards

    stub = RecordingSt()
    monkeypatch.setattr(cards, "st", stub)
    cards.render_brief(subject)

    missing = [
        (item.key, fact)
        for item in brief_items(subject)
        for fact in item.facts
        if fact not in stub.drawn
    ]
    assert not missing, f"항목으로는 세지만 화면에 안 그려진 사실: {missing}"


def test_min_requirement_is_drawn_as_a_banner(brief: StrategyBrief, monkeypatch):
    """합불 판정은 숫자 넷 사이가 아니라 배너로 나온다."""
    from render import cards

    stub = RecordingSt()
    monkeypatch.setattr(cards, "st", stub)
    cards.render_badges(brief, None)

    assert "최소 요건 미충족" in stub.drawn


def test_metric_row_stays_as_t09_left_it(brief: StrategyBrief, monkeypatch):
    """지표 네 칸은 T09 때 그대로 — 여기를 바꾸면 T12의 화면 단언이 깨진다.

    `tests/test_hitl.py`가 `at.metric`의 라벨 집합을 고정하고 있고 그 파일은 T15의
    소유가 아니다(R2). 배지를 배너로 뺀 것이 그 제약을 지키려는 선택이었으므로,
    선택의 근거를 여기에 못으로 박아 둔다.
    """
    from render import cards

    labels: list[str] = []

    class LabelSpy(RecordingSt):
        def metric(self, label, value, **kwargs):
            labels.append(label)
            return self._record(label, value, **kwargs)

    monkeypatch.setattr(cards, "st", LabelSpy())
    cards.render_badges(brief, None)

    assert set(labels) == {"남은 기간", "소스 충족률", "신뢰등급", "카드"}


def test_verdicts_and_importance_reach_the_card(brief: StrategyBrief, monkeypatch):
    """보강 데이터는 항목을 늘리진 않지만 **카드 안에는 실제로 들어간다.**"""
    from render import cards

    card = brief.track1[0]
    matches = [
        MatchResult(
            comp_id=card.comp_id,
            name=card.name,
            category=Category.D3_CLOUD_INFRA,
            required_level=card.required_level,
            my_level=card.my_level,
            state=card.state,
            verdicts=[
                CriterionVerdict(
                    criterion_id="c-1",
                    state=VerdictState.MET,
                    rationale="클러스터 운영 이력이 확인된다",
                    evidence=[
                        Evidence(
                            source_name="포트폴리오",
                            quote="Kubernetes 클러스터 운영 및 배포 자동화",
                            collected_at=date(2026, 7, 15),
                        )
                    ],
                )
            ],
        )
    ]
    required = _required((card.comp_id, Importance.REQUIRED))

    stub = RecordingSt()
    monkeypatch.setattr(cards, "st", stub)
    cards.render_brief(brief, matches=matches, required=required)

    assert "클러스터 운영 이력이 확인된다" in stub.drawn
    assert Importance.REQUIRED.value in stub.drawn
    assert "기준별 판정 1건" in stub.drawn


# --- 배선 (이게 없으면 `render/cards.py`는 죽은 코드다) -------------------------


class _Recorder:
    def __init__(self) -> None:
        self.seen: dict = {}

    def __call__(self, brief, *, matches=None, required=None) -> None:
        self.seen = {"brief": brief, "matches": matches, "required": required}


def test_app_calls_the_card_renderer(brief: StrategyBrief, monkeypatch):
    """`app/main.py`가 결과 화면을 `render/cards.py`에 넘기는가.

    T14가 만들어 놓고 아무도 부르지 않아 죽어 있던 전례(D39)를 되풀이하지 않기 위한
    검증이다. 배선은 테스트가 없으면 조용히 빠진다(DEVPLAN §2-1).
    """
    import app.main as main

    recorder = _Recorder()
    monkeypatch.setattr(main, "render_brief", recorder)
    monkeypatch.setattr(main, "st", RecordingSt())

    matches: list[MatchResult] = []
    required = _required(("req-be-06", Importance.REQUIRED))
    main.render_result(
        {
            "brief": brief,
            "match_results": matches,
            "required": required,
            "interview_round": 2,
        }
    )

    assert recorder.seen["brief"] is brief
    assert recorder.seen["matches"] is matches
    assert recorder.seen["required"] is required


def test_app_still_offers_the_markdown_download(brief: StrategyBrief, monkeypatch):
    """다운로드 버튼은 T15가 건드리지 않는다 — `render_markdown` 경로 유지."""
    import app.main as main

    stub = RecordingSt()
    monkeypatch.setattr(main, "render_brief", _Recorder())
    monkeypatch.setattr(main, "st", stub)
    main.render_result({"brief": brief})

    assert "전략 브리프 .md 다운로드" in stub.drawn
    assert main.filename_for(brief) in stub.drawn


def test_inline_renderers_are_gone():
    """T09의 인라인 렌더링이 **교체**됐는지 — 남아 있으면 화면이 둘로 갈라진다."""
    import app.main as main

    for name in ("render_summary", "render_tracks", "render_gaps"):
        assert not hasattr(main, name), f"app/main.py에 {name}()가 아직 남아 있다"


def test_app_reports_a_missing_brief(monkeypatch):
    """브리프가 없으면 렌더러를 부르지 않고 사유를 띄운다."""
    import app.main as main

    recorder = _Recorder()
    stub = RecordingSt()
    monkeypatch.setattr(main, "render_brief", recorder)
    monkeypatch.setattr(main, "st", stub)
    main.render_result({})

    assert recorder.seen == {}
    assert "브리프가 생성되지 않았습니다" in stub.drawn

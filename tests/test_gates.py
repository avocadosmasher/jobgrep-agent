"""T25 · 품질 게이트 + 실패 처리.

카드의 완료 조건은 하나다 — **JD 0건 시나리오에서 리포트가 강등 표기와 함께
생성되거나 차단됨.** 이 제품은 둘 중 **강등**을 골랐고(§12의 관통 규칙 "부분 실패를
전체 실패로 만들지 않는다"), 그래서 이 파일이 재는 것은 "차단됐는가"가 아니라
**강등이 결과물에 실제로 적혀 나가는가**다.

층은 다섯이다:
    ① UC-2 게이트   — 빈 문서를 걷어내고 판정을 다시 매기는가 (순수/노드)
    ② 표기 일치     — 신뢰등급과 미수집 목록이 **같은 근거**에서 나오는가 (관통)
    ③ UC-1 게이트   — 폭·깊이 임계값이 프로필을 가르는가 + **실측 표**
    ④ 화면 문구     — 게이트가 말하는 것을 화면이 그대로 띄우는가
    ⑤ 배선          — 두 그래프에 꽂혔는가, 자리가 맞는가 (§2-1)

**판정 규칙은 T18 것을 그대로 쓴다**(`build_gate_status`). 그래서 여기서 "JD 1건이면
통과"류의 단위 테스트를 다시 쓰지 않는다 — `tests/test_collect.py`가 이미 갖고 있고,
두 벌이 되면 한쪽만 고쳐진다. 이 파일이 새로 거는 것은 **게이트를 읽는 쪽**이다.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from contracts.enums import Category, Confidence, Level, SourceType
from contracts.models import (
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    ProfileJSON,
    SourceDocument,
    StrategyBrief,
)
from contracts.state import GraphState
from graphs.analysis_graph import NODE_NAMES, build_analysis_graph
from graphs.profile_graph import PROFILE_NODE_NAMES
from app.main import MODE_KEY, MODE_LABELS, MODE_PROFILE
from app.progress import load_labels
from nodes import analysis_nodes, collect as collect_mod, level_survey as level_survey_mod
from nodes.collect import BUDGET_EXHAUSTED_LABEL, RELIABILITY_ESTIMATED, RELIABILITY_NORMAL
from nodes.gates import (
    MIN_PROFILE_BREADTH,
    MIN_PROFILE_COVERAGE,
    average_coverage,
    coverage_breadth,
    entry_warning,
    gate_notices,
    profile_gate,
    profile_gate_status,
    profile_is_complete,
    profile_notice,
    quality_gate,
    usable_documents,
)
from nodes.level_survey import load_survey, score

import graphs.profile_graph as profile_mod

FIXTURES = Path(__file__).parent.parent / "fixtures"
APP = str(Path(__file__).parent.parent / "app" / "main.py")

JD = SourceDocument.model_validate_json((FIXTURES / "jd_sample_backend.json").read_bytes())
PROFILE = ProfileJSON.model_validate_json((FIXTURES / "profile_sample.json").read_bytes())

ALL_REQUIRED = [
    CompetencyRecord.model_validate(item)
    for item in json.loads((FIXTURES / "competencies_required.json").read_text("utf-8"))
]
BACKEND_REQUIRED = [c for c in ALL_REQUIRED if c.comp_id.startswith("req-be-")]

CRITERIA_ALL: dict[str, list[Criterion]] = {
    comp_id: [Criterion.model_validate(c) for c in group]
    for comp_id, group in json.loads(
        (FIXTURES / "criteria_sample.json").read_text("utf-8")
    ).items()
}
VERDICTS_ALL = [
    CriterionVerdict.model_validate(v)
    for v in json.loads((FIXTURES / "verdicts_all_met.json").read_text("utf-8"))
]

TARGET_DATE = date.today() + timedelta(days=90)

SURVEY = load_survey()


def empty_doc(doc_id: str, source_type: SourceType = SourceType.JD) -> SourceDocument:
    """수집 실패의 표현 — 예외가 아니라 **빈 본문 문서**다(D52)."""
    return SourceDocument(
        doc_id=doc_id,
        source_type=source_type,
        company=JD.company,
        title="본문 미수집 — 직접 붙여넣어 주세요",
        collected_at=date.today(),
        raw_text="",
        confidence=Confidence.LOW,
    )


def initial_state(**overrides) -> GraphState:
    state: GraphState = {
        "mode": "analysis",
        "company": JD.company,
        "role": "백엔드 엔지니어",
        "target_date": TARGET_DATE,
        "profile": PROFILE,
        "raw_jd_input": JD.raw_text,
    }
    state.update(overrides)
    return state


# --- 스텁 ---------------------------------------------------------------------


@pytest.fixture
def stubbed_tools(monkeypatch):
    """LLM·네트워크를 타는 것을 전부 갈아끼운다.

    수집 도구(T16·T17)까지 거는 것이 T08 스모크와 다른 점이다 — 이 파일은
    **수집이 실패했을 때**를 보는 파일이라 그 실패를 직접 주입해야 한다.
    """

    def fake_extract(docs, role):
        return list(BACKEND_REQUIRED) if docs else []

    def fake_decompose(comps):
        comp_ids = {c.comp_id for c in comps}
        return {k: list(v) for k, v in CRITERIA_ALL.items() if k in comp_ids}

    def fake_verify(criteria, profile, answers=None):
        wanted = {c.criterion_id for c in criteria}
        return [v for v in VERDICTS_ALL if v.criterion_id in wanted], []

    def fake_fill(brief):
        return brief.model_copy(update={"summary_line": "슬롯이 채워졌다"})

    def fake_retrieve(required, owned, top_k=3):
        return [(r.comp_id, owned[0].comp_id) for r in required] if owned else []

    monkeypatch.setattr(analysis_nodes, "extract_competencies", fake_extract)
    monkeypatch.setattr(analysis_nodes, "decompose_criteria", fake_decompose)
    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(analysis_nodes, "verify_criteria", fake_verify)
    monkeypatch.setattr(analysis_nodes, "fill_brief_slots", fake_fill)

    # 소프트 요건은 이 파일의 관심사가 아니다 — 늘 빈 결과로 두면 "미수집"이
    # 고정돼 하드 게이트의 효과만 남는다. 네트워크도 타지 않는다.
    monkeypatch.setattr(collect_mod, "fetch_tech_blog", lambda *a, **kw: [])
    monkeypatch.setattr(collect_mod, "get_company_values", lambda *a, **kw: [])
    return None


def run_analysis(**overrides) -> GraphState:
    return build_analysis_graph().invoke(initial_state(**overrides))


# --- ① UC-2 게이트 -------------------------------------------------------------


def test_empty_documents_are_dropped_before_the_brief():
    """**빈 문서를 1건으로 세면 게이트도 표기도 헛통과한다**(T16 수집 규약 ②)."""
    docs = [JD, empty_doc("jd-failed-001")]

    update = quality_gate({"source_docs": docs})

    assert update["source_docs"] == [JD]
    assert update["selected_job_ids"] == [JD.doc_id]
    assert update["gate_status"]["jd_count"] == 1


def test_a_document_that_was_never_fetched_fails_the_hard_gate():
    """본문이 없는 JD 1건은 **0건과 같다.** 여기가 강등의 분기점이다."""
    update = quality_gate({"source_docs": [empty_doc("jd-failed-001")]})

    gate = update["gate_status"]
    assert gate["hard_gate_passed"] is False
    assert gate["reliability"] == RELIABILITY_ESTIMATED
    assert SourceType.JD.value in gate["missing"]


def test_a_collected_jd_passes_the_hard_gate():
    gate = quality_gate({"source_docs": [JD]})["gate_status"]

    assert gate["hard_gate_passed"] is True
    assert gate["reliability"] == RELIABILITY_NORMAL


def test_budget_exhaustion_is_carried_over_from_collection():
    """예산 소진은 **수집 시점의 사실**이다 — 문서만 봐서는 알 수 없다.

    못 돌아서 없는 것과 돌았는데 없는 것은 사용자에게 전혀 다른 말이다.
    """
    before = {"jd_count": 1, "hard_gate_passed": True, "reliability": RELIABILITY_NORMAL,
              "missing": [BUDGET_EXHAUSTED_LABEL]}

    gate = quality_gate({"source_docs": [JD], "gate_status": before})["gate_status"]

    assert BUDGET_EXHAUSTED_LABEL in gate["missing"]


def test_the_gate_returns_only_contract_keys():
    """노드 규약 — 부분 갱신만, 계약에 있는 칸만(contracts/state.py)."""
    update = quality_gate({"source_docs": [JD]})

    assert set(update) <= set(GraphState.__annotations__)
    assert set(update) == {"source_docs", "selected_job_ids", "gate_status"}


def test_usable_documents_keeps_order_and_drops_nothing_else():
    docs = [JD, empty_doc("x"), JD.model_copy(update={"doc_id": "jd-2"})]

    assert [d.doc_id for d in usable_documents(docs)] == [JD.doc_id, "jd-2"]


# --- ② 표기 일치 (관통) --------------------------------------------------------


def test_a_failed_fetch_shows_up_as_missing_in_the_brief(stubbed_tools, monkeypatch):
    """**이 카드가 닫는 구멍.** 신뢰등급과 미수집 목록이 어긋나던 자리다.

    `source_coverage()`(T04)는 문서의 *유형*만 보고 세서 본문이 빈 JD 문서도
    "수집됨"으로 친다. 그러면 신뢰등급은 "추정 기반"인데 미수집 목록에는 JD가
    없는, **스스로 모순된 리포트**가 나간다. 게이트가 빈 문서를 걷어내 둘을
    같은 근거 위에 세운다.
    """
    monkeypatch.setattr(collect_mod, "fetch_jd_body", lambda *a, **kw: empty_doc("jd-failed"))

    final = run_analysis(raw_jd_input="https://example.com/jobs/1")
    meta = final["brief"].meta

    assert meta.reliability == RELIABILITY_ESTIMATED
    assert SourceType.JD.value in meta.missing_sources, "강등은 했는데 왜 강등인지는 안 적혔다"
    assert final["gate_status"]["hard_gate_passed"] is False


def test_zero_jd_still_produces_a_downgraded_report(stubbed_tools):
    """**카드의 완료 조건.** JD 0건 → 차단이 아니라 **강등 표기와 함께 생성**된다."""
    final = run_analysis(raw_jd_input="")

    assert isinstance(final["brief"], StrategyBrief), "부분 실패를 전체 실패로 만들지 않는다"
    assert final["brief"].meta.reliability == RELIABILITY_ESTIMATED
    assert final["gate_status"]["hard_gate_passed"] is False
    assert gate_notices(final["gate_status"]), "강등을 화면에 알릴 문구가 없다"


def test_a_pasted_jd_is_not_downgraded(stubbed_tools):
    """붙여넣기 백본은 **정상**이어야 한다 — 게이트가 멀쩡한 경로를 깎으면 안 된다."""
    final = run_analysis()

    assert final["brief"].meta.reliability == RELIABILITY_NORMAL
    assert final["gate_status"]["hard_gate_passed"] is True
    assert gate_notices(final["gate_status"]) == []
    # **개수 회귀** — 판정이 실제로 돌았는지 카드 수로 건다(D65 M21).
    brief = final["brief"]
    assert len(brief.track1 + brief.track2 + brief.track3) == len(
        set(CRITERIA_ALL) & {c.comp_id for c in BACKEND_REQUIRED}
    )


# --- ③ UC-1 게이트 + 실측 ------------------------------------------------------


def coverage_for(axis_ids: list[str]) -> dict[Category, float]:
    """실제 프리셋·채점 함수로 커버리지를 만든다 — 손으로 지어내지 않는다(R5)."""
    answers = {
        axis.axis_id: axis.markers[sorted(axis.markers)[0]]
        for axis in SURVEY.axes
        if axis.axis_id in set(axis_ids)
    }
    _, coverage = score(SURVEY, answers, {})
    return coverage


def profile_with(axis_ids: list[str]) -> ProfileJSON:
    return ProfileJSON(
        competencies=[],
        level_coordinates={},
        coverage=coverage_for(axis_ids),
        built_at=date.today(),
    )


def axes_of(category: Category) -> list[str]:
    return [a.axis_id for a in SURVEY.axes if a.category is category]


def first_axis_per_category(count: int) -> list[str]:
    seen: dict[Category, str] = {}
    for axis in SURVEY.axes:
        seen.setdefault(axis.category, axis.axis_id)
    return list(seen.values())[:count]


CATEGORIES = [a.category for a in SURVEY.axes]
FIRST_CATEGORY = SURVEY.axes[0].category

SCENARIOS: list[tuple[str, list[str], bool]] = [
    ("답 없음", [], False),
    ("이력서 사전 채움만(2축)", [a.axis_id for a in SURVEY.axes[:2]], False),
    ("한 대분류만 전부", axes_of(FIRST_CATEGORY), False),
    ("대분류 3개 각 1축", first_axis_per_category(3), False),
    ("자기 영역 2개 + 1축", axes_of(CATEGORIES[0]) + axes_of(CATEGORIES[6])
     + first_axis_per_category(3)[2:], True),
    ("축 절반", [a.axis_id for a in SURVEY.axes[: len(SURVEY.axes) // 2]], True),
    ("전부", [a.axis_id for a in SURVEY.axes], True),
]


@pytest.mark.parametrize(
    "name,axis_ids,expected", SCENARIOS, ids=[s[0] for s in SCENARIOS]
)
def test_profile_gate_separates_thin_profiles_from_usable_ones(name, axis_ids, expected):
    """**임계값 실측** — 시나리오마다 통과/미달이 갈리는지 못 박는다(D78).

    임계가 흔들리면 여기가 먼저 죽는다. 특히 "한 대분류만 전부"(깊이는 있고 폭이
    없다)와 "대분류 3개 각 1축"(폭은 있고 깊이가 없다)이 **둘 다 미달**인 것이
    두 값을 함께 보는 이유다.
    """
    assert profile_is_complete(profile_with(axis_ids)) is expected


def test_the_measured_coverage_table(capsys):
    """실측 표를 남긴다 — DEVLOG D78의 숫자가 여기서 나온다."""
    print("\n[UC-1 커버리지 실측 — 임계 폭 ≥ %d · 평균 ≥ %.2f]" % (
        MIN_PROFILE_BREADTH, MIN_PROFILE_COVERAGE))
    for name, axis_ids, _ in SCENARIOS:
        profile = profile_with(axis_ids)
        print(
            f"  {name:22s} 답 {len(axis_ids):2d}축 · 폭 {coverage_breadth(profile)}/"
            f"{len(profile.coverage)} · 평균 {average_coverage(profile):.2f} · "
            f"{'통과' if profile_is_complete(profile) else '미달'}"
        )

    assert len(SURVEY.axes) == 26, "프리셋이 바뀌면 위 임계값을 다시 재야 한다"


def test_breadth_alone_is_not_enough():
    """폭만 넘긴 프로필(대분류 5개 각 1축)은 통과하면 안 된다 — 26축 중 5축이다."""
    profile = profile_with(first_axis_per_category(5))

    assert coverage_breadth(profile) >= MIN_PROFILE_BREADTH
    assert average_coverage(profile) < MIN_PROFILE_COVERAGE
    assert not profile_is_complete(profile)


def test_depth_alone_is_not_enough():
    """깊이만 넘긴 프로필(한 대분류 전부)도 통과하면 안 된다 — JD가 다른 영역이면 무용이다."""
    profile = profile_with(axes_of(FIRST_CATEGORY))

    assert average_coverage(profile) >= MIN_PROFILE_COVERAGE
    assert coverage_breadth(profile) < MIN_PROFILE_BREADTH
    assert not profile_is_complete(profile)


def test_a_missing_profile_is_never_complete():
    """없는 프로필을 "완성"이라 하면, 이 판별자를 새로 부르는 화면이 조용히 통과한다."""
    assert not profile_is_complete(None)


def test_profile_gate_status_speaks_the_brief_vocabulary():
    """어휘를 새로 만들지 않는다 — 미완성 프로필의 예고가 곧 UC-2의 등급이다."""
    thin = profile_gate_status(profile_with([]))
    full = profile_gate_status(profile_with([a.axis_id for a in SURVEY.axes]))

    assert thin["reliability"] == RELIABILITY_ESTIMATED
    assert full["reliability"] == RELIABILITY_NORMAL
    assert thin["jd_count"] == 0, "UC-1에는 공고가 없다"
    assert set(thin["missing"]) == {c.value for c in set(CATEGORIES)}
    assert full["missing"] == []


def test_profile_gate_writes_nothing_when_there_is_no_profile():
    """만들다 만 스레드에 "미완성"을 찍으면 화면이 실패를 두 번 말한다."""
    assert profile_gate({}) == {}
    assert profile_gate({"profile": None}) == {}


def test_profile_gate_node_returns_only_the_gate():
    update = profile_gate({"profile": profile_with([])})

    assert set(update) == {"gate_status"}


# --- ④ 화면 문구 ---------------------------------------------------------------


def test_notices_are_silent_when_everything_passed():
    """통과한 분석에 경고가 뜨면 정작 필요한 경고가 안 읽힌다."""
    assert gate_notices(quality_gate({"source_docs": [JD]})["gate_status"]) == []
    assert gate_notices(None) == []


def test_the_hard_gate_notice_says_what_is_missing_and_what_to_do():
    notices = gate_notices(quality_gate({"source_docs": []})["gate_status"])

    assert len(notices) == 1
    assert "추정 기반" in notices[0]
    assert "붙여넣" in notices[0], "무엇을 하면 되는지가 없다"


def test_budget_and_hard_gate_are_two_separate_notices():
    """둘은 원인이 다르다 — 하나로 뭉치면 사용자가 무엇을 고쳐야 할지 모른다."""
    before = {"jd_count": 0, "hard_gate_passed": False,
              "reliability": RELIABILITY_ESTIMATED, "missing": [BUDGET_EXHAUSTED_LABEL]}

    notices = gate_notices(quality_gate({"source_docs": [], "gate_status": before})["gate_status"])

    assert len(notices) == 2


def test_soft_sources_are_not_repeated_in_the_notices():
    """기술블로그·인재상 미수집은 브리프 헤더와 공백 고지에 이미 나간다."""
    gate = quality_gate({"source_docs": [JD]})["gate_status"]

    assert SourceType.TECH_BLOG.value in gate["missing"], "게이트는 알고 있어야 한다"
    assert gate_notices(gate) == [], "그러나 상단 경고로 또 말하지는 않는다"


def test_profile_notices_name_the_unanswered_areas():
    thin = profile_with(axes_of(FIRST_CATEGORY))

    notice = profile_notice(thin)
    assert notice is not None
    assert "미완성" in notice
    assert CATEGORIES[6].value in notice, "어디를 채우면 되는지가 없다"

    assert profile_notice(profile_with([a.axis_id for a in SURVEY.axes])) is None
    assert profile_notice(None) is None


def test_entry_warning_only_fires_for_a_thin_profile():
    """프로필이 아예 없을 때는 화면이 이미 다른 경고를 갖고 있다 — 겹치면 둘 다 안 읽힌다."""
    assert entry_warning(None) is None
    assert entry_warning(profile_with([a.axis_id for a in SURVEY.axes])) is None
    assert "미완성" in (entry_warning(profile_with([])) or "")


# --- ⑤ 배선 (§2-1) -------------------------------------------------------------


def test_quality_gate_sits_between_aggregate_and_the_brief():
    """자리가 곧 계약이다 — 브리프 뒤에 두면 메타가 이미 만들어진 뒤라 늦는다."""
    assert "quality_gate" in NODE_NAMES, "게이트가 그래프에 없다"
    assert NODE_NAMES.index("aggregate") < NODE_NAMES.index("quality_gate")
    assert NODE_NAMES.index("quality_gate") < NODE_NAMES.index("build_brief")


def test_profile_gate_is_the_last_step_of_uc1():
    assert PROFILE_NODE_NAMES.index("build_profile") < PROFILE_NODE_NAMES.index("profile_gate")
    assert PROFILE_NODE_NAMES[-1] == "profile_gate"


def test_both_gates_have_user_facing_labels():
    """내부 노드명이 화면에 새지 않아야 한다 (T13 불변식)."""
    analysis = load_labels()
    profile = load_labels(Path(__file__).parent.parent / "presets" / "profile_labels.yaml")

    assert analysis.node("quality_gate").label != analysis.fallback
    assert profile.node("profile_gate").label != profile.fallback


# --- ⑤-b 화면 관통 (AppTest) ---------------------------------------------------


def run_app(monkeypatch):
    import graphs.session as session

    monkeypatch.setattr(session, "build_checkpointer", lambda *a, **kw: InMemorySaver())
    return AppTest.from_file(APP, default_timeout=60).run()


def test_the_downgrade_warning_reaches_the_result_screen(monkeypatch, stubbed_tools):
    """**게이트를 읽는 쪽이 실제로 있는가.** 여기가 T18이 못 닫은 구멍이다.

    `gate_status`는 이 카드 전까지 **쓰이기만 하고 아무도 안 읽는 칸**이었다.
    화면에 강등 경고가 뜨는 것이 그 구멍이 닫혔다는 증거다.
    """
    monkeypatch.setattr(collect_mod, "fetch_jd_body", lambda *a, **kw: empty_doc("jd-failed"))

    at = run_app(monkeypatch)
    at.text_input[0].set_value(JD.company)
    at.text_input[1].set_value("백엔드 엔지니어")
    at.text_area[0].set_value("https://example.com/jobs/1")
    at = at.button[0].click().run()

    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "추정 기반" in warnings, "강등 경고가 결과 화면에 안 떴다"
    assert at.download_button, "경고는 떴는데 리포트가 안 나왔다 — 차단이 아니라 강등이다"


def test_a_thin_profile_is_called_out_on_the_result_screen(monkeypatch, tmp_path, stubbed_tools):
    """**UC-2 진입 경고**(§12-1). 막지는 않되, 결과가 왜 얇은지 말한다.

    **문구가 결과 화면에 있는 이유** — 실행 화면에서 띄우면 바로 뒤의 `st.rerun()`이
    화면을 갈아치워 사용자는 한 번도 못 본다. 이 테스트가 그 자리를 고정한다.
    화면에서만 나는 결함이라 게이트 함수를 따로 재는 것으로는 **호출부가 빠진 것**을
    못 잡는다(뮤테이션 M34가 그 자리였다).
    """
    from graphs.profile_graph import save_profile

    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")
    save_profile(profile_with(axes_of(FIRST_CATEGORY)), directory=tmp_path / "profiles")

    at = run_app(monkeypatch)
    at.text_input[0].set_value(JD.company)
    at.text_input[1].set_value("백엔드 엔지니어")
    at.text_area[0].set_value(JD.raw_text)
    next(c for c in at.checkbox if "직전에 만든" in c.label).set_value(True)
    next(c for c in at.checkbox if "샘플" in c.label).set_value(False)
    at = at.button[0].click().run()

    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "미완성" in warnings, "얇은 프로필로 분석을 시작했는데 아무 말도 안 했다"
    assert at.download_button, "경고는 경고일 뿐 — 분석은 그대로 돌아야 한다"


def test_a_thin_profile_is_marked_incomplete_on_screen(monkeypatch, tmp_path):
    """UC-1 하드 요건 미달 표시 (§12-1). 다운로드는 막지 않는다."""
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")
    monkeypatch.setattr(analysis_nodes, "extract_competencies", lambda docs, role: [])
    monkeypatch.setattr(
        level_survey_mod, "retrieve_candidates", lambda *a, **kw: []
    )

    at = run_app(monkeypatch)
    next(w for w in at.radio if w.key == MODE_KEY).set_value(MODE_LABELS[MODE_PROFILE])
    at = at.run()
    at = at.button[0].click().run()  # 이력서 없이 설문만

    # 한 대분류만 답한다 — 깊이는 있고 폭이 없는, 실측 표의 "미달" 시나리오다.
    answered = set(axes_of(FIRST_CATEGORY))
    for widget in [w for w in at.radio if w.key != MODE_KEY]:
        if any(widget.key.endswith(axis_id) for axis_id in answered):
            widget.set_value(widget.options[0])
    at = at.button[0].click().run()

    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "미완성" in warnings, "얇은 프로필인데 아무 말도 안 했다"
    assert at.download_button, "미완성이라고 다운로드까지 막으면 안 된다"


def test_a_full_profile_gets_no_warning(monkeypatch, tmp_path):
    """반대 방향도 건다 — 다 채운 사용자에게 "미완성"이 뜨면 게이트를 아무도 안 믿는다."""
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")
    monkeypatch.setattr(analysis_nodes, "extract_competencies", lambda docs, role: [])
    monkeypatch.setattr(level_survey_mod, "retrieve_candidates", lambda *a, **kw: [])

    at = run_app(monkeypatch)
    next(w for w in at.radio if w.key == MODE_KEY).set_value(MODE_LABELS[MODE_PROFILE])
    at = at.run()
    at = at.button[0].click().run()

    for widget in [w for w in at.radio if w.key != MODE_KEY]:
        widget.set_value(widget.options[0])
    at = at.button[0].click().run()

    assert not at.exception
    assert "미완성" not in " ".join(w.value for w in at.warning)

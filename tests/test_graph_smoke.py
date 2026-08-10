"""T08 · 분석 그래프 스모크.

오프라인: LLM을 타는 도구 4종을 **픽스처를 돌려주는 스텁**으로 갈아끼우고 그래프를
end-to-end 1회 돌린다. 스텁이 반환하는 값은 전부 `fixtures/`의 골든 데이터이며
이 테스트가 지어낸 값이 아니다 (R5).

온라인(`-m llm`): 붙여넣은 JD 1건으로 실제 API를 태워 P0 완료 판정을 증명한다.

여기서 검사하는 것은 **배선**이다 — 판정·집계·트랙 배정이 맞는지는 각 도구의
테스트가 이미 본다. 이 파일은 상태가 노드 사이를 제대로 흘러가는지만 본다.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from contracts.models import (
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    ProfileJSON,
    SourceDocument,
    StrategyBrief,
)
from contracts.state import GraphState
from graphs.analysis_graph import NODE_NAMES, build_analysis_graph, node_sequence
from nodes import analysis_nodes

FIXTURES = Path(__file__).parent.parent / "fixtures"

JD = SourceDocument.model_validate_json((FIXTURES / "jd_sample_backend.json").read_bytes())
PROFILE = ProfileJSON.model_validate_json((FIXTURES / "profile_sample.json").read_bytes())

ALL_REQUIRED = [
    CompetencyRecord.model_validate(item)
    for item in json.loads((FIXTURES / "competencies_required.json").read_text("utf-8"))
]
# 백엔드 JD 1건만 붙여넣는 시나리오이므로 그 JD에서 나온 역량만 쓴다.
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

# 픽스처가 기준을 갖고 있는 백엔드 역량 — 집계·브리프에 실제로 실리는 것들.
SCORED_COMP_IDS = sorted(set(CRITERIA_ALL) & {c.comp_id for c in BACKEND_REQUIRED})

TARGET_DATE = date.today() + timedelta(days=90)


def initial_state() -> GraphState:
    return {
        "mode": "analysis",
        "company": JD.company,
        "role": "백엔드 엔지니어",
        "target_date": TARGET_DATE,
        "profile": PROFILE,
        "raw_jd_input": JD.raw_text,
    }


# --- 오프라인 스텁 ------------------------------------------------------------


@pytest.fixture
def stubbed_tools(monkeypatch):
    """LLM을 타는 4개 도구를 픽스처 반환 스텁으로 대체하고 호출 기록을 남긴다."""
    calls: list[str] = []

    def fake_extract(docs, role):
        calls.append("extract_competencies")
        # 실제 도구는 문서가 없으면 LLM을 부르지 않고 빈 목록을 낸다(T04 계약).
        return list(BACKEND_REQUIRED) if docs else []

    def fake_decompose(comps):
        calls.append("decompose_criteria")
        comp_ids = {c.comp_id for c in comps}
        return {k: list(v) for k, v in CRITERIA_ALL.items() if k in comp_ids}

    def fake_verify(criteria, profile, answers=None):
        calls.append("verify_criteria")
        wanted = {c.criterion_id for c in criteria}
        return [v for v in VERDICTS_ALL if v.criterion_id in wanted], []

    def fake_fill(brief):
        calls.append("fill_brief_slots")
        return brief.model_copy(update={"summary_line": "슬롯이 채워졌다"})

    def fake_retrieve(required, owned, top_k=3):
        # 임베딩도 네트워크를 탄다(T14b). 갈아끼우지 않으면 오프라인 실행이 실제
        # API를 왕복해 §4의 "pytest -q는 LLM 없이 통과한다"가 깨진다.
        calls.append("retrieve_candidates")
        return [(r.comp_id, owned[0].comp_id) for r in required] if owned else []

    monkeypatch.setattr(analysis_nodes, "extract_competencies", fake_extract)
    monkeypatch.setattr(analysis_nodes, "decompose_criteria", fake_decompose)
    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(analysis_nodes, "verify_criteria", fake_verify)
    monkeypatch.setattr(analysis_nodes, "fill_brief_slots", fake_fill)
    return calls


# --- 완료 조건 ----------------------------------------------------------------


def test_graph_invoke_produces_strategy_brief(stubbed_tools):
    """T08 완료 조건 — invoke 한 번에 state['brief']가 StrategyBrief로 채워진다."""
    final = build_analysis_graph().invoke(initial_state())

    brief = final["brief"]
    assert isinstance(brief, StrategyBrief)
    assert brief.meta.company == JD.company
    assert brief.meta.days_remaining == 90

    cards = brief.track1 + brief.track2 + brief.track3
    assert {c.comp_id for c in cards} == set(SCORED_COMP_IDS)


def test_every_stage_runs_in_order(stubbed_tools):
    build_analysis_graph().invoke(initial_state())
    assert stubbed_tools == [
        "extract_competencies",
        "retrieve_candidates",   # T14b
        "decompose_criteria",
        "verify_criteria",
        "fill_brief_slots",
    ]


def test_state_carries_intermediate_results(stubbed_tools):
    """중간 산출이 상태에 남아야 T13(진행 표시)·T12(재개)가 볼 것이 있다."""
    final = build_analysis_graph().invoke(initial_state())

    # T18 — JD 문서의 정본은 `fetch_jd_body`를 지난 것이다. `ingest_pasted_jd`가
    # 만든 P0 임시 문서(`PASTED_DOC_ID`)는 `collect`가 밀어내므로 여기 없다.
    # id는 본문 해시라 고정값으로 못 박지 않는다 — 문서가 **한 건**이고 그것이
    # `selected_job_ids`와 같은 id라는 것만 본다.
    assert len(final["source_docs"]) == 1
    assert final["selected_job_ids"] == [final["source_docs"][0].doc_id]
    # 앞뒤 공백만 다듬고 본문은 한 글자도 안 바뀐다 — P0의 `ingest_pasted_jd`도
    # 같은 `.strip()`을 했으므로 이 경로의 동작은 T18 전후로 동일하다.
    assert final["source_docs"][0].raw_text == JD.raw_text.strip()
    assert len(final["required"]) == len(BACKEND_REQUIRED)
    assert set(final["criteria"]) == set(SCORED_COMP_IDS)
    assert len(final["verdicts"]) == sum(len(CRITERIA_ALL[c]) for c in SCORED_COMP_IDS)
    assert {r.comp_id for r in final["match_results"]} == set(SCORED_COMP_IDS)


def test_fill_slots_runs_after_build_brief(stubbed_tools):
    """DEVLOG D19 — 그래프에 fill_slots 노드가 실제로 들어 있어야 한다."""
    final = build_analysis_graph().invoke(initial_state())
    assert final["brief"].summary_line == "슬롯이 채워졌다"


# --- 불변식 -------------------------------------------------------------------


def test_graph_wires_the_documented_sequence():
    assert NODE_NAMES == [
        "ingest_pasted_jd",
        "collect",    # T18 — 수집 도구 3종을 예산 안에서 돌린다. 소비처(extract) 앞
        "extract",
        "retrieve",   # T14b — 입력이 갖춰지는 extract 직후, 소비처(aggregate) 앞
        "decompose",
        "verify",
        "aggregate",
        "build_brief",
        "fill_slots",
    ]


def test_nodes_return_partial_updates_not_whole_state(stubbed_tools):
    """각 노드는 자기가 바꾼 칸만 담은 dict를 반환한다 (contracts/state.py 규약).

    노드 함수를 `analysis_nodes`에서 `getattr`로 찾지 않고 **`node_sequence()`가
    들고 있는 것을 그대로** 부른다 — 노드가 전부 한 모듈에 살지 않기 때문이다
    (T18의 `collect`는 `nodes/collect.py`에 있다).
    """
    state: GraphState = initial_state()
    allowed = set(GraphState.__annotations__)

    for name, fn in node_sequence():
        update = fn(state)
        assert isinstance(update, dict), f"{name}이 dict를 반환하지 않았다"
        assert set(update) <= allowed, f"{name}이 계약에 없는 칸을 만들었다"
        assert "raw_jd_input" not in update, f"{name}이 입력 칸을 되돌려줬다"
        state = {**state, **update}

    assert isinstance(state["brief"], StrategyBrief)


def test_aggregate_uses_rules_only(monkeypatch, stubbed_tools):
    """집계 노드는 LLM을 부르지 않는다 (§8-2 ★ 규칙 전용)."""

    def explode(*args, **kw):
        raise AssertionError("aggregate는 LLM을 타면 안 된다")

    monkeypatch.setattr("llm.client.complete_structured", explode)

    state = {**initial_state(), "required": BACKEND_REQUIRED}
    state = {**state, **analysis_nodes.decompose(state)}
    state = {**state, **analysis_nodes.verify(state)}

    results = analysis_nodes.aggregate(state)["match_results"]
    assert {r.comp_id for r in results} == set(SCORED_COMP_IDS)
    # P0에는 후보쌍 검색(T14)이 없으므로 보유 레벨을 특정할 수 없다.
    assert all(r.my_level is None for r in results)


def test_pending_questions_do_not_stop_the_flow(monkeypatch):
    """verify가 질문을 남겨도 P0는 그대로 끝까지 간다 (HITL은 T11)."""
    question_criteria = CRITERIA_ALL[SCORED_COMP_IDS[0]]

    monkeypatch.setattr(
        analysis_nodes, "extract_competencies", lambda docs, role: list(BACKEND_REQUIRED)
    )
    monkeypatch.setattr(
        analysis_nodes,
        "decompose_criteria",
        lambda comps: {SCORED_COMP_IDS[0]: list(question_criteria)},
    )
    monkeypatch.setattr(
        analysis_nodes,
        "verify_criteria",
        lambda criteria, profile, answers=None: (
            [],
            [
                {"question_id": f"q-{c.criterion_id}", "criterion_id": c.criterion_id, "text": "?"}
                for c in criteria
            ],
        ),
    )
    monkeypatch.setattr(analysis_nodes, "fill_brief_slots", lambda brief: brief)
    # 임베딩도 네트워크를 탄다(T14b). 이 줄이 없으면 이 테스트만 실제 API를 왕복하고,
    # §4의 "`pytest -q`는 LLM 없이 통과한다"가 조용히 깨진다 — 실제로 그렇게 새고
    # 있었고 429가 날 때까지 초록불이었다(D66).
    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", lambda *a, **kw: [])

    final = build_analysis_graph().invoke(initial_state())

    assert final["pending_questions"], "질문이 상태에 남아야 T11이 이어받는다"
    assert isinstance(final["brief"], StrategyBrief)


def test_empty_jd_input_still_produces_a_brief(stubbed_tools):
    """본문이 비어도 그래프가 예외로 죽지 않는다 (§11-2 결측 처리)."""
    final = build_analysis_graph().invoke({**initial_state(), "raw_jd_input": ""})

    assert final["source_docs"] == []
    assert isinstance(final["brief"], StrategyBrief)
    # 공고가 하나도 없으면 신뢰등급이 강등된다 (§12-1).
    assert final["brief"].meta.reliability == "추정 기반"


def test_source_coverage_counts_missing_types():
    coverage, missing = analysis_nodes.source_coverage([JD])
    assert coverage == pytest.approx(1 / 3)
    assert missing == ["기술블로그", "인재상"]


# --- 온라인 (실제 API) --------------------------------------------------------


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY 없음")
def test_end_to_end_with_real_llm():
    """P0 완료 판정 — 붙여넣은 JD 1건이 실제로 3트랙 브리프까지 관통한다."""
    final = build_analysis_graph().invoke(initial_state())

    brief = final["brief"]
    assert isinstance(brief, StrategyBrief)
    assert final["required"], "JD에서 요구 역량이 추출돼야 한다"
    assert final["criteria"], "역량이 기준으로 분해돼야 한다"
    assert final["match_results"], "집계 결과가 있어야 한다"
    assert brief.meta.company == JD.company
    assert brief.meta.reliability == "정상"

    # 근거가 없어 카드가 안 서는 역량은 집계에서 빠지고 gaps로 간다(§11-2 ②, D16).
    # 따라서 요약 카운트는 전체 match_results가 아니라 **생성된 카드 수**와 맞는다.
    cards = brief.track1 + brief.track2 + brief.track3
    assert cards, "카드가 한 장은 나와야 관통했다고 할 수 있다"
    assert sum(brief.summary_counts.values()) == len(cards)
    assert len(cards) <= len(final["match_results"])

"""T14b · 후보쌍 배선 + `my_level` 확정 검증.

이 파일이 보는 것은 **배선과 규칙**이다. 후보쌍 검색 자체의 품질(어느 쌍이 top-1로
오는가)은 `tests/test_retrieve.py`가 이미 본다.

오프라인 테스트는 `retrieve_candidates`를 **주입된 가짜로 갈아끼운다.** 임베딩은
네트워크를 타므로 그대로 두면 `uv run pytest -q`가 LLM 없이 통과해야 한다는 정책이
깨진다. 가짜가 돌려주는 쌍은 픽스처 역량 id로 만든 것이며 실제 유사도와 무관하다 —
여기서 검증하는 것이 "무엇이 매칭되는가"가 아니라 "매칭 결과를 어떻게 쓰는가"이기
때문에 그래도 된다.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from contracts.enums import Level, MatchState
from contracts.models import (
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    ProfileJSON,
    SourceDocument,
)
from contracts.state import GraphState
from graphs.analysis_graph import NODE_NAMES, node_sequence
from nodes import analysis_nodes
from nodes.analysis_nodes import my_level_for

FIXTURES = Path(__file__).parent.parent / "fixtures"

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
VERDICTS_MET = [
    CriterionVerdict.model_validate(v)
    for v in json.loads((FIXTURES / "verdicts_all_met.json").read_text("utf-8"))
]
VERDICTS_UNMET = [
    CriterionVerdict.model_validate(v)
    for v in json.loads((FIXTURES / "verdicts_mostly_unmet.json").read_text("utf-8"))
]

SCORED_COMP_IDS = sorted(set(CRITERIA_ALL) & {c.comp_id for c in BACKEND_REQUIRED})
OWNED_BY_ID = {c.comp_id: c for c in PROFILE.competencies}
TARGET_DATE = date.today() + timedelta(days=90)


def base_state(**overrides) -> GraphState:
    state: GraphState = {
        "mode": "analysis",
        "company": JD.company,
        "role": "백엔드 엔지니어",
        "target_date": TARGET_DATE,
        "profile": PROFILE,
        "raw_jd_input": JD.raw_text,
        "required": list(BACKEND_REQUIRED),
        "criteria": {k: list(v) for k, v in CRITERIA_ALL.items() if k in SCORED_COMP_IDS},
    }
    state.update(overrides)
    return state


def pairs_all_to(owned_id: str) -> list[tuple[str, str]]:
    """모든 요구 역량을 같은 보유 역량 하나에 이어 붙인 후보쌍."""
    return [(comp_id, owned_id) for comp_id in SCORED_COMP_IDS]


def two_owned_with_different_levels() -> tuple[CompetencyRecord, CompetencyRecord]:
    """레벨이 서로 다른 보유 역량 2건. 순서 의존을 구별하려면 값이 달라야 한다.

    인덱스로 집지 않는 이유 — 픽스처의 앞 두 건은 레벨이 같아서(둘 다 `실무운영`)
    무엇을 골랐는지 구별이 안 된다. 픽스처가 바뀌어도 이 함수는 계속 맞는다.
    """
    leveled = [c for c in PROFILE.competencies if c.level is not None]
    first = leveled[0]
    other = next((c for c in leveled if c.level is not first.level), None)
    assert other is not None, "픽스처 전제: 레벨이 서로 다른 보유 역량이 2건 이상 있어야 한다"
    return first, other


# --- ① 배선 -------------------------------------------------------------------


def test_retrieve_is_wired_between_extract_and_decompose():
    """뮤테이션 ② — `NODE_SEQUENCE`에서 retrieve를 빼면 이 테스트가 죽는다."""
    assert "retrieve" in NODE_NAMES, "retrieve 노드가 그래프에 없다"
    assert NODE_NAMES.index("extract") + 1 == NODE_NAMES.index("retrieve")
    assert NODE_NAMES.index("retrieve") + 1 == NODE_NAMES.index("decompose")


def test_retrieve_is_wired_in_the_interactive_path_too():
    """대화형 배선은 `node_sequence()`가 자동 계산한다 — 별도 작업이 없음을 고정한다."""
    names = [name for name, _ in node_sequence(interactive=True)]
    assert names.index("extract") + 1 == names.index("retrieve")
    # 인터뷰가 끼어도 retrieve는 verify 앞에 그대로 있어야 한다.
    assert names.index("retrieve") < names.index("verify")


def test_retrieve_node_has_a_user_facing_label():
    """내부 노드명이 화면에 새지 않아야 한다 (T13 불변식)."""
    from app.progress import load_labels

    labels = load_labels()
    assert "retrieve" in labels.nodes
    assert labels.node("retrieve").label != labels.fallback


# --- ② retrieve 노드 ----------------------------------------------------------


def test_retrieve_returns_pairs_from_required_and_profile(monkeypatch):
    seen: dict = {}

    def fake_retrieve(required, owned, top_k=3):
        seen["required"] = [c.comp_id for c in required]
        seen["owned"] = [c.comp_id for c in owned]
        seen["top_k"] = top_k
        return [(required[0].comp_id, owned[0].comp_id)]

    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", fake_retrieve)

    update = analysis_nodes.retrieve(base_state())

    assert list(update) == ["candidate_pairs"], "부분 갱신만 반환해야 한다"
    assert seen["required"] == [c.comp_id for c in BACKEND_REQUIRED]
    assert seen["owned"] == [c.comp_id for c in PROFILE.competencies]
    assert seen["top_k"] == analysis_nodes.CANDIDATE_TOP_K


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile": None},
        {"required": []},
    ],
    ids=["프로필 없음", "요구 역량 없음"],
)
def test_retrieve_skips_the_api_when_there_is_nothing_to_match(monkeypatch, overrides):
    """짝지을 상대가 없으면 **호출하지 않는다.** 임베딩 왕복은 공짜가 아니다."""

    def explode(*args, **kw):
        raise AssertionError("짝지을 게 없는데 임베딩을 불렀다")

    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", explode)

    assert analysis_nodes.retrieve(base_state(**overrides)) == {"candidate_pairs": []}


# --- ③ my_level 규칙 ----------------------------------------------------------


def test_unmet_never_gets_a_level_even_with_a_candidate():
    """**뮤테이션 ① — 이 규칙을 빼면 여기가 죽는다.**

    임베딩 top-1은 "가장 가까운 것"이지 "대응하는 것"이 아니다. 근거로 확인되지
    않은(UNMET) 역량에 레벨을 찍으면 없는 경력을 지어내는 것이다 (§7-3).
    """
    owned_id = PROFILE.competencies[0].comp_id
    assert OWNED_BY_ID[owned_id].level is not None, "픽스처 전제: 이 보유 역량엔 레벨이 있다"

    level = my_level_for(
        SCORED_COMP_IDS[0], MatchState.UNMET, {SCORED_COMP_IDS[0]: owned_id}, OWNED_BY_ID
    )
    assert level is None


@pytest.mark.parametrize("match_state", [MatchState.MET, MatchState.ADJACENT])
def test_confirmed_match_takes_the_level_of_its_top_candidate(match_state):
    owned = PROFILE.competencies[0]
    level = my_level_for(
        SCORED_COMP_IDS[0], match_state, {SCORED_COMP_IDS[0]: owned.comp_id}, OWNED_BY_ID
    )
    assert level is owned.level


def test_no_candidate_means_no_level():
    assert my_level_for(SCORED_COMP_IDS[0], MatchState.MET, {}, OWNED_BY_ID) is None


def test_candidate_without_a_level_is_left_empty():
    """보유 역량에 레벨이 안 적혀 있으면 추정하지 않는다."""
    bare = CompetencyRecord(
        comp_id="pf-bare",
        category=PROFILE.competencies[0].category,
        name="레벨이 적히지 않은 역량",
        importance=PROFILE.competencies[0].importance,
    )
    level = my_level_for(
        SCORED_COMP_IDS[0], MatchState.MET, {SCORED_COMP_IDS[0]: "pf-bare"}, {"pf-bare": bare}
    )
    assert level is None


def test_aggregate_uses_the_first_pair_not_the_last():
    """**뮤테이션 ③ — 첫 후보 대신 마지막을 고르면 여기가 죽는다.**

    `retrieve_candidates`는 같은 요구 역량 안에서 유사도 내림차순으로 돌려주므로
    (`tools/retrieve.py` docstring) **첫 쌍이 top-1이다.** 이 테스트가 그 의존을
    직접 고정한다 — 계약(`contracts/tools.py`)에는 순서 보장이 적혀 있지 않아서다.
    """
    comp_id = SCORED_COMP_IDS[0]
    first, second = two_owned_with_different_levels()

    state = base_state(
        verdicts=list(VERDICTS_MET),
        candidate_pairs=[(comp_id, first.comp_id), (comp_id, second.comp_id)],
    )
    results = {r.comp_id: r for r in analysis_nodes.aggregate(state)["match_results"]}

    assert results[comp_id].my_level is first.level


# --- ④ 완료 조건 --------------------------------------------------------------


def test_aggregate_fills_at_least_one_level_with_the_fixture_profile():
    """완료 조건 ② — 픽스처로 돌리면 `my_level`이 최소 1건은 채워진다."""
    owned = PROFILE.competencies[0]
    state = base_state(
        verdicts=list(VERDICTS_MET), candidate_pairs=pairs_all_to(owned.comp_id)
    )
    results = analysis_nodes.aggregate(state)["match_results"]

    assert results, "집계 결과가 비었다"
    assert any(r.my_level is not None for r in results)


def test_aggregate_leaves_unmet_results_empty_end_to_end():
    """완료 조건 ③ — 후보쌍이 있어도 UNMET 역량은 레벨이 비어 있다."""
    owned = PROFILE.competencies[0]
    state = base_state(
        verdicts=list(VERDICTS_UNMET), candidate_pairs=pairs_all_to(owned.comp_id)
    )
    results = analysis_nodes.aggregate(state)["match_results"]

    unmet = [r for r in results if r.state is MatchState.UNMET]
    assert unmet, "픽스처 전제: UNMET이 최소 1건 나와야 이 테스트가 의미 있다"
    assert all(r.my_level is None for r in unmet)


def test_aggregate_without_a_profile_behaves_exactly_as_before():
    """완료 조건 ④ — 프로필이 없어도 예외 없이 이전과 같은 결과가 나온다."""
    state = base_state(profile=None, verdicts=list(VERDICTS_MET), candidate_pairs=[])
    results = analysis_nodes.aggregate(state)["match_results"]

    assert {r.comp_id for r in results} == set(SCORED_COMP_IDS)
    assert all(r.my_level is None for r in results)


def test_aggregate_still_calls_no_llm(monkeypatch):
    """후보쌍이 붙어도 집계는 순수 규칙 함수로 남는다 (§8-2).

    `retrieve` 노드가 이미 계산해 상태에 넣어 뒀으므로 `aggregate`는 읽기만 한다.
    """

    def explode(*args, **kw):
        raise AssertionError("aggregate는 LLM을 타면 안 된다")

    monkeypatch.setattr("llm.client.complete_structured", explode)
    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", explode)

    owned = PROFILE.competencies[0]
    state = base_state(
        verdicts=list(VERDICTS_MET), candidate_pairs=pairs_all_to(owned.comp_id)
    )
    assert analysis_nodes.aggregate(state)["match_results"]


# --- ⑤ 온라인: 실제 임베딩으로 관통 -------------------------------------------


@pytest.mark.llm
def test_llm_real_embedding_fills_levels_through_the_nodes():
    """실제 임베딩으로 retrieve → aggregate가 이어지는지 1회 확인한다."""
    update = analysis_nodes.retrieve(base_state())
    pairs = update["candidate_pairs"]

    assert pairs, "후보쌍이 하나도 안 나왔다"
    assert all(req in {c.comp_id for c in BACKEND_REQUIRED} for req, _ in pairs)
    assert all(owned in OWNED_BY_ID for _, owned in pairs)

    state = base_state(verdicts=list(VERDICTS_MET), candidate_pairs=pairs)
    results = analysis_nodes.aggregate(state)["match_results"]

    assert any(r.my_level is not None for r in results), "실제 후보쌍으로도 레벨이 안 붙었다"
    assert all(
        r.my_level is None for r in results if r.state is MatchState.UNMET
    ), "UNMET에 레벨이 붙었다"

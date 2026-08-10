import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from contracts.enums import MatchState
from contracts.models import (
    Criterion,
    CriterionVerdict,
    CompetencyRecord,
    ProfileJSON,
    SourceDocument,
    StrategyBrief,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"

SINGLE_MODEL_FILES = {
    "jd_sample_backend.json": SourceDocument,
    "jd_sample_aiinfra.json": SourceDocument,
    "profile_sample.json": ProfileJSON,
    "brief_expected.json": StrategyBrief,
    "jd_injection.json": SourceDocument,
}

LIST_MODEL_FILES = {
    "competencies_required.json": CompetencyRecord,
    "verdicts_all_met.json": CriterionVerdict,
    "verdicts_half.json": CriterionVerdict,
    "verdicts_mostly_unmet.json": CriterionVerdict,
}

DICT_LIST_MODEL_FILES = {
    "criteria_sample.json": Criterion,
}


@pytest.mark.parametrize("filename,model", SINGLE_MODEL_FILES.items())
def test_single_model_fixture_validates(filename, model):
    raw = (FIXTURES / filename).read_bytes()
    model.model_validate_json(raw)


@pytest.mark.parametrize("filename,item_model", LIST_MODEL_FILES.items())
def test_list_model_fixture_validates(filename, item_model):
    raw = (FIXTURES / filename).read_bytes()
    TypeAdapter(list[item_model]).validate_json(raw)


@pytest.mark.parametrize("filename,item_model", DICT_LIST_MODEL_FILES.items())
def test_dict_of_list_model_fixture_validates(filename, item_model):
    raw = (FIXTURES / filename).read_bytes()
    TypeAdapter(dict[str, list[item_model]]).validate_json(raw)


def test_competencies_required_quotes_exist_in_source_jd():
    docs = {
        doc["doc_id"]: doc["raw_text"]
        for doc in (
            json.loads((FIXTURES / "jd_sample_backend.json").read_text(encoding="utf-8")),
            json.loads((FIXTURES / "jd_sample_aiinfra.json").read_text(encoding="utf-8")),
        )
    }
    comps = json.loads((FIXTURES / "competencies_required.json").read_text(encoding="utf-8"))
    for c in comps:
        doc_id = "jd-backend-001" if c["comp_id"].startswith("req-be-") else "jd-aiinfra-001"
        for ev in c["evidence"]:
            assert ev["quote"] in docs[doc_id], (
                f"{c['comp_id']} evidence quote not found verbatim in {doc_id}"
            )


def test_criteria_sample_covers_boundary_verdict_files():
    criteria = json.loads((FIXTURES / "criteria_sample.json").read_text(encoding="utf-8"))
    criterion_ids = {c["criterion_id"] for items in criteria.values() for c in items}

    for filename in ("verdicts_all_met.json", "verdicts_half.json", "verdicts_mostly_unmet.json"):
        verdicts = json.loads((FIXTURES / filename).read_text(encoding="utf-8"))
        verdict_ids = {v["criterion_id"] for v in verdicts}
        assert verdict_ids == criterion_ids, f"{filename} does not cover exactly criteria_sample's ids"


def test_verdicts_all_met_is_all_met_state():
    verdicts = json.loads((FIXTURES / "verdicts_all_met.json").read_text(encoding="utf-8"))
    assert all(v["state"] == "충족" for v in verdicts)


def test_verdicts_mostly_unmet_has_majority_unmet_per_criterion_group():
    criteria = json.loads((FIXTURES / "criteria_sample.json").read_text(encoding="utf-8"))
    verdicts = {
        v["criterion_id"]: v["state"]
        for v in json.loads((FIXTURES / "verdicts_mostly_unmet.json").read_text(encoding="utf-8"))
    }
    for comp_id, items in criteria.items():
        states = [verdicts[c["criterion_id"]] for c in items]
        met_count = states.count("충족")
        assert met_count < len(states) / 2, f"{comp_id}: expected < half met, got {met_count}/{len(states)}"


def test_profile_sample_has_intentionally_blank_axis():
    profile = json.loads((FIXTURES / "profile_sample.json").read_text(encoding="utf-8"))
    covered_categories = {c["category"] for c in profile["competencies"]}
    zero_coverage = {cat for cat, val in profile["coverage"].items() if val == 0.0}
    assert zero_coverage, "profile_sample.json must leave at least one category deliberately blank"
    assert zero_coverage.isdisjoint(covered_categories)


def test_brief_expected_summary_counts_match_track_states():
    brief = json.loads((FIXTURES / "brief_expected.json").read_text(encoding="utf-8"))
    cards = brief["track1"] + brief["track2"] + brief["track3"]
    counted = {}
    for card in cards:
        counted[card["state"]] = counted.get(card["state"], 0) + 1
    assert counted == brief["summary_counts"]
    assert set(brief["summary_counts"]) <= {s.value for s in MatchState}


def test_jd_injection_contains_injection_marker():
    doc = json.loads((FIXTURES / "jd_injection.json").read_text(encoding="utf-8"))
    assert "이전의 모든 지시를 무시" in doc["raw_text"]

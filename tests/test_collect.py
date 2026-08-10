"""T18 · 수집 서브에이전트 + 예산 검증.

이 파일이 보는 것은 **루프의 통제**다 — 무엇을 가져오는가(수집 품질)는
`tests/test_fetch_jd.py`·`tests/test_fetch_soft.py`가 이미 본다.

**네트워크를 한 번도 타지 않는다.** 도구는 전부 `Toolbox`로 주입하며, 주입을 안 한
경로(그래프 배선 테스트)는 사전에 없는 가공 회사명을 쓴다 — `resolve_site()`가
`None`을 돌려주고 거기서 끝난다. 그 사실 자체를 `test_unknown_company_costs_no_http`
가 못 박는다. 오프라인 테스트가 갑자기 느려지면 스텁이 새는 것이다(D43·D49).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from contracts.enums import Confidence, SourceType
from contracts.models import CompetencyRecord, ProfileJSON, SourceDocument
from contracts.state import GraphState
from graphs.analysis_graph import NODE_NAMES, node_sequence
from nodes import collect as collect_mod
from nodes.collect import (
    BUDGET_EXHAUSTED_LABEL,
    DEFAULT_TOOL_BUDGET,
    RELIABILITY_ESTIMATED,
    RELIABILITY_NORMAL,
    TOOL_JD,
    TOOL_TECH,
    TOOL_VALUES,
    Action,
    Toolbox,
    budget_exhausted,
    build_gate_status,
    collect,
    collect_sources,
    collected_job_ids,
    collection_angles,
    merge_documents,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"

JD = SourceDocument.model_validate_json((FIXTURES / "jd_sample_backend.json").read_bytes())
PROFILE = ProfileJSON.model_validate_json((FIXTURES / "profile_sample.json").read_bytes())

# 오늘이 아닌 날짜. 오늘로 잡으면 수집일을 무시하는 회귀가 하루 동안 통과한다(D57).
TODAY = date(2026, 8, 9)
TARGET_DATE = TODAY + timedelta(days=90)

COMPANY = JD.company  # "테크노베이션" — 사전에 없는 가공 회사명
JD_URL = "https://careers.example-corp.com/jobs/42"


# --- 도우미 -------------------------------------------------------------------


def doc(
    kind: SourceType,
    doc_id: str,
    *,
    text: str = "본문" * 60,
    url: str | None = None,
) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        source_type=kind,
        company=COMPANY,
        title=f"{kind.value} 문서",
        url=url,
        collected_at=TODAY,
        raw_text=text,
        confidence=Confidence.MID,
    )


def pasted_jd(text: str = JD.raw_text) -> SourceDocument:
    """`ingest_pasted_jd`(P0)가 만드는 모양의 문서."""
    return SourceDocument(
        doc_id="jd-pasted-001",
        source_type=SourceType.JD,
        company=COMPANY,
        title="백엔드 엔지니어 채용공고 (붙여넣기)",
        collected_at=TODAY,
        raw_text=text,
        confidence=Confidence.HIGH,
    )


class Recorder:
    """도구 호출을 기록하는 가짜 도구 상자.

    반환값은 시나리오마다 지정하고, **기록은 호출 순서 그대로** 남긴다 —
    "몇 번 불렀나"와 "어떤 순서로 불렀나"를 둘 다 봐야 예산과 재시도 각도를
    동시에 검증할 수 있다.
    """

    def __init__(
        self,
        *,
        jd: SourceDocument | None = None,
        tech: dict[str, list[SourceDocument]] | None = None,
        values: dict[str, list[SourceDocument]] | None = None,
        raises: set[str] | None = None,
    ) -> None:
        self.jd = jd
        self.tech = tech or {}
        self.values = values or {}
        self.raises = raises or set()
        self.calls: list[Action] = []

    def _record(self, tool: str, target: str) -> None:
        self.calls.append(Action(tool, target))
        if tool in self.raises:
            raise RuntimeError(f"{tool} 폭발")

    def fetch_jd_body(self, target: str, company: str) -> SourceDocument:
        self._record(TOOL_JD, target)
        return self.jd if self.jd is not None else pasted_jd(target)

    def fetch_tech_blog(self, target: str, company: str) -> list[SourceDocument]:
        self._record(TOOL_TECH, target)
        return list(self.tech.get(target, []))

    def get_company_values(self, target: str, company: str) -> list[SourceDocument]:
        self._record(TOOL_VALUES, target)
        return list(self.values.get(target, []))

    def box(self) -> Toolbox:
        return Toolbox(self.fetch_jd_body, self.fetch_tech_blog, self.get_company_values)

    def names(self) -> list[str]:
        return [c.tool for c in self.calls]


def base_state(**overrides) -> GraphState:
    state: GraphState = {
        "mode": "analysis",
        "company": COMPANY,
        "role": "백엔드 엔지니어",
        "target_date": TARGET_DATE,
        "profile": PROFILE,
        "raw_jd_input": JD.raw_text,
        "source_docs": [pasted_jd()],
        "selected_job_ids": ["jd-pasted-001"],
    }
    state.update(overrides)
    return state


# --- ① 배선 (§2-1 — 이 카드가 만든 것을 누가 부르는가) --------------------------


def test_collect_is_wired_between_ingest_and_extract():
    """**뮤테이션 M01 — `NODE_SEQUENCE`에서 collect를 빼면 여기가 죽는다.**

    T14·T16·T17이 만들고 아무도 안 부른 채 남았던 구멍과 같은 모양이라(D39·D43),
    배선 자체를 검증에 넣는다(§2-1).
    """
    assert "collect" in NODE_NAMES, "collect 노드가 그래프에 없다"
    assert NODE_NAMES.index("ingest_pasted_jd") + 1 == NODE_NAMES.index("collect")
    assert NODE_NAMES.index("collect") + 1 == NODE_NAMES.index("extract")


def test_collect_is_wired_in_the_interactive_path_too():
    """대화형에는 H1(T19)이 `ingest`와 `collect` 사이에 끼므로 **바로 뒤**는 아니다.

    T18이 실제로 요구하는 것은 순서 관계 둘뿐이다 — 붙여넣기 정리 **뒤**,
    소비처(extract) **앞**. 인접을 못 박으면 그 사이에 노드를 넣을 때마다 깨진다.
    """
    names = [name for name, _ in node_sequence(interactive=True)]
    assert names.index("ingest_pasted_jd") < names.index("collect")
    assert names.index("collect") + 1 == names.index("extract")


def test_collect_node_has_a_user_facing_label():
    """내부 노드명이 화면에 새지 않아야 한다 (T13 불변식)."""
    from app.progress import load_labels

    labels = load_labels()
    assert "collect" in labels.nodes
    assert labels.node("collect").label != labels.fallback


@pytest.mark.parametrize("tool", [TOOL_JD, TOOL_TECH, TOOL_VALUES])
def test_every_tool_has_a_sub_line_label(tool):
    """**뮤테이션 M02 — 도구 이름 상수를 바꾸면 여기가 죽는다.**

    `_on_tool`은 모르는 이름을 `fallback`으로 떨어뜨리므로 라벨이 어긋나도
    화면은 안 깨진다 — 그래서 조용히 썩는다.
    """
    from app.progress import load_labels

    labels = load_labels()
    assert labels.tool(tool) != labels.fallback


def test_collect_returns_a_partial_update_only():
    update = collect(base_state(tool_budget=0))
    assert set(update) <= set(GraphState.__annotations__)
    assert "raw_jd_input" not in update, "입력 칸을 되돌려주면 안 된다"


# --- ② JD — P0 문서를 교체한다 --------------------------------------------------


def test_pasted_text_goes_through_the_jd_tool_once():
    """붙여넣기도 `fetch_jd_body`를 지난다 — ①층은 네트워크를 안 탄다(T16 불변식)."""
    rec = Recorder()
    outcome = collect_sources(
        [pasted_jd()], company=COMPANY, raw_jd_input=JD.raw_text, tools=rec.box()
    )

    assert rec.calls[0] == Action(TOOL_JD, JD.raw_text)
    assert rec.names().count(TOOL_JD) == 1, "JD 도구를 두 번 불렀다"
    assert [d.source_type for d in outcome.docs].count(SourceType.JD) == 1


def test_url_input_replaces_the_p0_placeholder_document():
    """**뮤테이션 M03 — JD를 교체하지 않고 추가하면 여기가 죽는다.**

    P0의 `ingest_pasted_jd`는 입력이 URL이어도 그 URL 문자열을 본문으로 삼는다.
    그 문서가 살아남으면 `extract`가 URL 한 줄에서 역량을 뽑으려 든다.
    """
    fetched = doc(SourceType.JD, "jd-fetched-001", text="진짜 공고 본문" * 40, url=JD_URL)
    rec = Recorder(jd=fetched)

    outcome = collect_sources(
        [pasted_jd(JD_URL)], company=COMPANY, raw_jd_input=JD_URL, tools=rec.box()
    )

    jds = [d for d in outcome.docs if d.source_type is SourceType.JD]
    assert [d.doc_id for d in jds] == ["jd-fetched-001"]
    assert all("jd-pasted" not in d.doc_id for d in outcome.docs)


def test_failed_jd_fetch_leaves_an_empty_document_not_the_url():
    """수집 실패는 예외가 아니라 빈 본문 문서다(D52). 게이트가 그걸 0건으로 센다."""
    empty = doc(SourceType.JD, "jd-empty", text="", url=JD_URL)
    rec = Recorder(jd=empty)

    outcome = collect_sources(
        [pasted_jd(JD_URL)], company=COMPANY, raw_jd_input=JD_URL, tools=rec.box()
    )
    gate = build_gate_status(outcome.docs, exhausted=outcome.exhausted)

    assert gate["jd_count"] == 0
    assert gate["hard_gate_passed"] is False
    assert gate["reliability"] == RELIABILITY_ESTIMATED
    assert SourceType.JD.value in gate["missing"]


def test_empty_input_calls_no_jd_tool():
    rec = Recorder()
    outcome = collect_sources([], company="", raw_jd_input="", tools=rec.box())

    assert rec.calls == [], "부를 대상이 없는데 도구를 불렀다"
    assert outcome.docs == []
    assert outcome.calls == 0


# --- ③ 소프트 요건 + 재시도 각도 ------------------------------------------------


def test_soft_sources_are_collected_after_the_jd():
    tech = [doc(SourceType.TECH_BLOG, "tech-1")]
    values = [doc(SourceType.VALUES, "values-1")]
    rec = Recorder(tech={COMPANY: tech}, values={COMPANY: values})

    outcome = collect_sources(
        [pasted_jd()], company=COMPANY, raw_jd_input=JD.raw_text, tools=rec.box()
    )

    assert rec.names() == [TOOL_JD, TOOL_TECH, TOOL_VALUES], "JD가 먼저여야 한다"
    kinds = {d.source_type for d in outcome.docs}
    assert kinds == {SourceType.JD, SourceType.TECH_BLOG, SourceType.VALUES}
    assert outcome.exhausted is False


def test_the_second_angle_comes_from_the_collected_jd_url():
    """**뮤테이션 M04 — 두 번째 각도를 빼면 여기가 죽는다.**

    회사명으로 못 찾으면 *이미 수집한 JD의 주소*로 한 번 더 두드린다. 지어낸
    주소가 아니라 수집 결과에서 유도한 것이다(D56 — 주소를 추측하지 말 것).
    """
    fetched = doc(SourceType.JD, "jd-fetched-001", url=JD_URL)
    found = [doc(SourceType.TECH_BLOG, "tech-1")]
    rec = Recorder(jd=fetched, tech={"careers.example-corp.com": found})

    outcome = collect_sources(
        [pasted_jd(JD_URL)], company=COMPANY, raw_jd_input=JD_URL, tools=rec.box()
    )

    tech_targets = [c.target for c in rec.calls if c.tool == TOOL_TECH]
    assert tech_targets == [COMPANY, "careers.example-corp.com"]
    assert any(d.doc_id == "tech-1" for d in outcome.docs)


def test_angles_are_deduplicated_and_ordered():
    docs = [
        doc(SourceType.JD, "jd-1", url="https://www.example.com/jobs/1"),
        doc(SourceType.JD, "jd-2", url="https://example.com/jobs/2"),
    ]
    assert collection_angles(COMPANY, docs) == [COMPANY, "example.com"]
    # 회사명이 비면 그 자리는 아예 없다 — 빈 문자열로 도구를 부르면 안 된다.
    assert collection_angles("   ", docs) == ["example.com"]


def test_a_url_less_jd_offers_no_second_angle():
    rec = Recorder()
    collect_sources(
        [pasted_jd()], company=COMPANY, raw_jd_input=JD.raw_text, tools=rec.box()
    )
    assert [c.target for c in rec.calls if c.tool == TOOL_TECH] == [COMPANY]


def test_an_already_collected_kind_is_not_fetched_again():
    """상태에 이미 기술블로그가 있으면 그 도구는 아예 안 부른다."""
    rec = Recorder()
    collect_sources(
        [pasted_jd(), doc(SourceType.TECH_BLOG, "tech-existing")],
        company=COMPANY,
        raw_jd_input=JD.raw_text,
        tools=rec.box(),
    )
    assert TOOL_TECH not in rec.names()
    assert TOOL_VALUES in rec.names()


def test_an_empty_soft_document_does_not_count_as_collected():
    """빈 본문은 없는 것으로 친다 — `is_uncollected()`가 판별자다(D52)."""
    rec = Recorder()
    collect_sources(
        [pasted_jd(), doc(SourceType.TECH_BLOG, "tech-empty", text="")],
        company=COMPANY,
        raw_jd_input=JD.raw_text,
        tools=rec.box(),
    )
    assert TOOL_TECH in rec.names()


# --- ④ 종료 — 완료 조건 ---------------------------------------------------------


def test_loop_terminates_when_every_tool_returns_nothing():
    """**완료 조건 · 무한 루프 없음.** 실패해도 같은 (도구, 인자)를 두 번 안 부른다.

    도구는 실패를 예외가 아니라 빈 결과로 표현하므로(D52·D55), 실패를 근거로
    재시도하면 영원히 돈다. `tried` 집합이 그 통로를 막는다.
    """
    fetched = doc(SourceType.JD, "jd-fetched-001", url=JD_URL)
    rec = Recorder(jd=fetched)  # tech·values는 전부 빈 목록

    outcome = collect_sources(
        [pasted_jd(JD_URL)],
        company=COMPANY,
        raw_jd_input=JD_URL,
        budget=DEFAULT_TOOL_BUDGET,
        tools=rec.box(),
    )

    assert len(rec.calls) == len(set(rec.calls)), "같은 호출을 두 번 했다"
    # JD 1 + (tech·values) × (회사명·JD 도메인) = 5. 예산 12가 남는다.
    assert len(rec.calls) == 5
    assert outcome.budget_left == DEFAULT_TOOL_BUDGET - 5
    assert outcome.exhausted is False, "할 일이 떨어진 것은 예산 소진이 아니다"


def test_budget_exhaustion_returns_partial_results_without_hanging():
    """**완료 조건 · 예산 소진 시나리오.** 부분 수집으로 진행하고 실패로 만들지 않는다."""
    fetched = doc(SourceType.JD, "jd-fetched-001", url=JD_URL)
    tech = [doc(SourceType.TECH_BLOG, "tech-1")]
    rec = Recorder(jd=fetched, tech={COMPANY: tech}, values={COMPANY: [doc(SourceType.VALUES, "v")]})

    outcome = collect_sources(
        [pasted_jd(JD_URL)], company=COMPANY, raw_jd_input=JD_URL, budget=2, tools=rec.box()
    )

    assert len(rec.calls) == 2, "예산을 넘겨 불렀다"
    assert outcome.budget_left == 0
    assert outcome.exhausted is True
    # 부분 결과는 살아 있다 — JD와 기술블로그는 들어왔고 인재상만 못 왔다.
    kinds = {d.source_type for d in outcome.docs}
    assert kinds == {SourceType.JD, SourceType.TECH_BLOG}

    gate = build_gate_status(outcome.docs, exhausted=outcome.exhausted)
    assert gate["hard_gate_passed"] is True, "예산 소진은 실패가 아니다"
    assert gate["reliability"] == RELIABILITY_NORMAL
    assert SourceType.VALUES.value in gate["missing"]
    assert budget_exhausted(gate) is True


def test_zero_budget_runs_no_tool_at_all():
    """**뮤테이션 M05 — `while left > 0`을 `>=`로 바꾸면 여기가 죽는다.**"""
    rec = Recorder()
    outcome = collect_sources(
        [pasted_jd()], company=COMPANY, raw_jd_input=JD.raw_text, budget=0, tools=rec.box()
    )

    assert rec.calls == []
    assert outcome.calls == 0
    assert outcome.exhausted is True, "할 일이 남았는데 한 번도 못 돌았다"
    assert outcome.docs == [pasted_jd()], "입력 문서는 그대로 남아야 한다"


def test_spending_the_last_slot_exactly_is_not_exhaustion():
    """**뮤테이션 M06 — 소진 판정을 `left == 0`으로 바꾸면 여기가 죽는다.**

    마지막 한 칸을 정확히 쓰고 할 일이 없어진 것은 소진이 아니다. 그걸 소진으로
    적으면 게이트가 거짓을 말하고 T25가 멀쩡한 수집을 강등한다.
    """
    tech = [doc(SourceType.TECH_BLOG, "tech-1")]
    values = [doc(SourceType.VALUES, "values-1")]
    rec = Recorder(tech={COMPANY: tech}, values={COMPANY: values})

    outcome = collect_sources(
        [pasted_jd()], company=COMPANY, raw_jd_input=JD.raw_text, budget=3, tools=rec.box()
    )

    assert len(rec.calls) == 3
    assert outcome.budget_left == 0
    assert outcome.exhausted is False


def test_a_tool_that_raises_does_not_stop_the_loop():
    """도구는 예외를 안 던지기로 돼 있지만(D52·D55) 경계에서 한 번 더 받는다.

    **어디서 실패하는지를 지정한다** — 앞 계층이 먹어버리면 뒤 계층이 검증되지
    않는다(D54). 여기서 터지는 것은 기술블로그 도구뿐이다.
    """
    values = [doc(SourceType.VALUES, "values-1")]
    rec = Recorder(values={COMPANY: values}, raises={TOOL_TECH})

    outcome = collect_sources(
        [pasted_jd()], company=COMPANY, raw_jd_input=JD.raw_text, tools=rec.box()
    )

    assert TOOL_TECH in rec.names(), "터지는 도구를 실제로 부르긴 했나"
    assert any(d.source_type is SourceType.VALUES for d in outcome.docs), "뒤 도구가 안 돌았다"
    gate = build_gate_status(outcome.docs, exhausted=outcome.exhausted)
    assert SourceType.TECH_BLOG.value in gate["missing"]


# --- ⑤ 예산·반복 회계 -----------------------------------------------------------


def test_node_defaults_the_budget_to_the_card_value():
    """상태에 예산이 없으면 카드 초기값 12."""
    state = base_state()
    state.pop("tool_budget", None)
    update = collect(state)
    assert update["tool_budget"] == DEFAULT_TOOL_BUDGET - update["iteration"]


def test_node_spends_from_the_state_budget_and_accumulates_iterations():
    """**뮤테이션 M07 — `iteration`을 누적 대신 대입으로 바꾸면 여기가 죽는다.**"""
    update = collect(base_state(tool_budget=2, iteration=7))

    assert update["tool_budget"] == 0
    assert update["iteration"] == 7 + 2


def test_a_negative_budget_is_treated_as_zero():
    update = collect(base_state(tool_budget=-5))
    assert update["tool_budget"] == 0
    assert update["iteration"] == 0


# --- ⑥ 게이트 -------------------------------------------------------------------


def test_gate_counts_bodies_not_documents():
    """**뮤테이션 M08 — `jd_count`를 문서 수로 바꾸면 여기가 죽는다.**

    빈 문서를 1건으로 세면 하드 게이트가 헛통과한다(D52).
    """
    gate = build_gate_status([doc(SourceType.JD, "jd-empty", text="")])
    assert gate["jd_count"] == 0
    assert gate["hard_gate_passed"] is False


def test_gate_uses_the_same_missing_vocabulary_as_the_brief():
    """라벨 어휘는 `SourceType` 값이 정본이다 — 새로 짓지 않는다(D55)."""
    gate = build_gate_status([pasted_jd()])
    assert gate["missing"] == [SourceType.TECH_BLOG.value, SourceType.VALUES.value]


def test_soft_sources_never_downgrade_reliability():
    """**뮤테이션 M09 — 강등 조건에 소프트 요건을 넣으면 여기가 죽는다.**

    `build_brief_meta`(tools/brief.py)와 같은 규칙이어야 한다. 게이트와 브리프
    머리말이 서로 다른 등급을 말하면 화면이 자기모순에 빠진다(§12-1 하드 요건).
    """
    from tools.brief import build_brief_meta

    docs = [pasted_jd()]
    gate = build_gate_status(docs)
    meta = build_brief_meta(
        company=COMPANY,
        role="백엔드 엔지니어",
        selected_jobs=collected_job_ids(docs),
        target_date=TARGET_DATE,
        source_coverage=1 / 3,
        missing_sources=[],
        today=TODAY,
    )

    assert gate["reliability"] == RELIABILITY_NORMAL == meta.reliability
    assert gate["missing"], "소프트 요건 부재는 표기는 된다"


def test_budget_exhausted_reads_the_label_not_the_string():
    assert budget_exhausted(None) is False
    assert budget_exhausted(build_gate_status([pasted_jd()], exhausted=False)) is False
    exhausted = build_gate_status([pasted_jd()], exhausted=True)
    assert BUDGET_EXHAUSTED_LABEL in exhausted["missing"]
    assert budget_exhausted(exhausted) is True


def test_selected_job_ids_drop_uncollected_documents():
    """`BriefMeta.selected_jobs`로 흘러가 신뢰등급을 가른다 — 빈 문서를 세면 안 된다."""
    docs = [doc(SourceType.JD, "jd-empty", text=""), doc(SourceType.JD, "jd-real")]
    assert collected_job_ids(docs) == ["jd-real"]


def test_node_reports_no_job_id_when_the_url_could_not_be_fetched(monkeypatch):
    """**뮤테이션 M16 — 노드가 빈 문서까지 `selected_job_ids`에 넣으면 여기가 죽는다.**

    순수 함수(`collected_job_ids`)만 검증하면 노드가 그걸 **안 쓰는** 회귀를 못
    잡는다. 실제로 M16이 그 틈으로 살아남아 이 테스트가 생겼다.

    가져오기 실패는 `http_get`을 터뜨려 만든다 — 주입점을 통과해 T16 본체를
    실제로 지나가므로(robots 조회부터 막힌다) "실패하면 빈 문서"라는 D52 규약이
    노드까지 이어지는지가 함께 확인된다.
    """

    def offline(*args, **kwargs):
        raise RuntimeError("네트워크 없음")

    monkeypatch.setattr("tools.fetch_jd.http_get", offline)
    monkeypatch.setattr("tools.fetch_soft.http_get", offline)

    update = collect(base_state(raw_jd_input=JD_URL, source_docs=[pasted_jd(JD_URL)]))

    assert update["selected_job_ids"] == []
    assert update["gate_status"]["jd_count"] == 0
    assert update["gate_status"]["hard_gate_passed"] is False
    assert update["gate_status"]["reliability"] == RELIABILITY_ESTIMATED
    # P0의 URL 문자열 문서는 사라졌다 — 하류가 URL 한 줄에서 역량을 뽑으면 안 된다.
    assert all("jd-pasted" not in d.doc_id for d in update["source_docs"])


# --- ⑦ 문서 병합 -----------------------------------------------------------------


def test_merge_deduplicates_by_doc_id():
    """같은 글이 후보 URL 두 개로 잡히면 하류가 같은 근거를 두 번 센다."""
    existing = [pasted_jd(), doc(SourceType.TECH_BLOG, "tech-1")]
    merged = merge_documents(
        existing, Action(TOOL_TECH, COMPANY), [doc(SourceType.TECH_BLOG, "tech-1")]
    )
    assert [d.doc_id for d in merged] == ["jd-pasted-001", "tech-1"]


def test_merge_keeps_soft_documents_when_the_jd_is_replaced():
    """JD 교체가 이미 모은 소프트 문서까지 쓸어 가면 안 된다."""
    existing = [pasted_jd(JD_URL), doc(SourceType.VALUES, "values-1")]
    merged = merge_documents(
        existing, Action(TOOL_JD, JD_URL), [doc(SourceType.JD, "jd-fetched-001")]
    )
    assert [d.doc_id for d in merged] == ["jd-fetched-001", "values-1"]


# --- ⑧ 진행 표시 훅 ---------------------------------------------------------------


def test_each_tool_call_emits_one_sub_line_event():
    """T13의 서브 라인으로 흐를 payload를 그대로 검증한다."""
    tech = [doc(SourceType.TECH_BLOG, "tech-1"), doc(SourceType.TECH_BLOG, "tech-2")]
    rec = Recorder(tech={COMPANY: tech})
    seen: list[dict] = []

    collect_sources(
        [pasted_jd()],
        company=COMPANY,
        raw_jd_input=JD.raw_text,
        tools=rec.box(),
        emit=seen.append,
    )

    assert len(seen) == len(rec.calls), "호출마다 이벤트가 하나씩 나와야 한다"
    by_tool = {e["tool"]: e for e in seen}
    assert by_tool[TOOL_TECH]["ok"] is True
    assert by_tool[TOOL_TECH]["detail"] == "2건"
    assert by_tool[TOOL_VALUES]["ok"] is False
    assert by_tool[TOOL_VALUES]["detail"] == "없음"


def test_progress_tracker_folds_the_events_into_the_collect_line():
    """이벤트가 실제로 `collect` 줄 밑에 붙는지 트래커로 확인한다."""
    from app.progress import LineState, tracker_for

    tracker = tracker_for([name for name, _ in node_sequence()])
    tracker.handle(("updates", {"ingest_pasted_jd": {"source_docs": [1]}}))
    tracker.handle(("custom", {"tool": TOOL_TECH, "ok": True, "detail": "2건"}))

    line = next(line for line in tracker.lines if line.node == "collect")
    assert line.state is LineState.RUNNING
    assert line.substeps == ["↳ 기술 블로그 훑기 — 2건"]


def test_the_node_survives_without_a_stream_writer():
    """그래프 밖에서 노드를 직접 부르면 `get_stream_writer()`가 던진다 — 죽지 않아야 한다."""
    assert collect_mod._stream_emitter() is None
    assert collect(base_state(tool_budget=1))["iteration"] == 1


# --- ⑨ 오프라인 안전 --------------------------------------------------------------


def test_unknown_company_costs_no_http(monkeypatch):
    """**주입 없이 도는 경로가 네트워크를 안 타는지 못 박는다.**

    가공 회사명은 `resolve_site()`에서 `None`이 되어 소프트 수집이 **주소를 하나도
    두드리지 않고** 끝나고, 붙여넣은 본문은 T16 ①층이라 네트워크를 안 탄다. 기존
    그래프 테스트들(스모크·HITL·진행 표시)이 조용히 실 HTTP를 왕복하지 않는 근거가
    이것뿐이라 여기서 고정한다(D43·D49).

    도구가 **불리긴 한다** — 예산은 3칸 나간다. 값싼 것과 안 부르는 것은 다르고,
    여기서 검증하는 것은 앞쪽이다.
    """

    def explode(*args, **kwargs):
        raise AssertionError("오프라인 테스트가 실 HTTP를 탔다")

    monkeypatch.setattr("tools.fetch_jd.http_get", explode)
    monkeypatch.setattr("tools.fetch_soft.http_get", explode)

    update = collect(base_state())

    assert update["gate_status"]["hard_gate_passed"] is True
    assert update["iteration"] == 3, "JD 1회 + 소프트 요건 2회"
    assert update["tool_budget"] == DEFAULT_TOOL_BUDGET - 3


def test_a_real_company_name_would_reach_the_network(monkeypatch):
    """위 테스트가 우연이 아님을 보인다 — 사전에 있는 회사는 실제로 나가려 든다.

    이게 없으면 `explode`가 한 번도 안 불리는 것을 "안전"으로 오독하게 된다.
    """
    calls: list[str] = []

    def spy(url, *args, **kwargs):
        calls.append(url)
        raise RuntimeError("네트워크 없음")

    monkeypatch.setattr("tools.fetch_soft.http_get", spy)

    collect_sources([pasted_jd()], company="카카오", raw_jd_input="", budget=1)

    assert calls, "사전에 있는 회사인데 아무 주소도 안 두드렸다"


# --- ⑩ 그래프 관통 -----------------------------------------------------------------


def test_collect_runs_inside_the_graph_and_fills_gate_status(monkeypatch):
    """배선된 노드가 실제 그래프 실행에서 상태를 채우는지 본다.

    LLM 도구는 전부 갈아끼운다 — 이 테스트가 보는 것은 수집 노드가 그래프 안에서
    돌면서 `source_docs`·`gate_status`를 남기는가이지 분석 품질이 아니다.
    """
    from graphs.analysis_graph import build_analysis_graph
    from nodes import analysis_nodes

    required = [
        CompetencyRecord.model_validate(item)
        for item in json.loads(
            (FIXTURES / "competencies_required.json").read_text("utf-8")
        )
        if item["comp_id"].startswith("req-be-")
    ]

    monkeypatch.setattr(analysis_nodes, "extract_competencies", lambda docs, role: list(required))
    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", lambda *a, **kw: [])
    monkeypatch.setattr(analysis_nodes, "decompose_criteria", lambda comps: {})
    monkeypatch.setattr(analysis_nodes, "verify_criteria", lambda *a, **kw: ([], []))
    monkeypatch.setattr(analysis_nodes, "fill_brief_slots", lambda brief: brief)

    final = build_analysis_graph().invoke(base_state())

    assert final["gate_status"]["hard_gate_passed"] is True
    assert final["gate_status"]["jd_count"] == 1
    assert final["selected_job_ids"], "수집된 JD id가 브리프 메타로 흘러야 한다"
    assert final["brief"].meta.selected_jobs == final["selected_job_ids"]
    assert final["tool_budget"] == DEFAULT_TOOL_BUDGET - final["iteration"]

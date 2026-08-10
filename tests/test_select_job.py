"""T19 · H1 공고 선택 검증 — 발견 → 중단 → 선택 → 재개.

카드의 검증란은 "수동"이지만 자동화했다. T12가 "재개가 상류를 다시 안 돌린다"를
호출 횟수로 증명한 전례가 있고(D31), 수동으로 남길 이유가 없다 — 오히려 H1은
**중단점이 둘로 늘어난 첫 카드**라 회귀가 조용히 날 자리다.

**네트워크·LLM을 한 번도 타지 않는다.** 발견 도구와 본문 수집은 `nodes.select_job`의
원본 이름을 갈아끼워 막고(D49 — 스텁은 원본 모듈에), LLM 도구는 T12의 `install_stubs`를
그대로 쓴다. 오프라인 테스트가 갑자기 느려지면 스텁이 새는 것이다(D43·D49).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from app.hitl import MULTI_SEPARATOR, normalize_prompt
from contracts.enums import Confidence, SourceType
from contracts.models import SourceDocument
from graphs.analysis_graph import NODE_NAMES, build_analysis_graph, node_sequence
from graphs.session import ThreadPhase, resume_or_start
from nodes import select_job as select_mod
from nodes.select_job import (
    SELECTION_KEY,
    build_selection_payload,
    discover,
    job_labels,
    label_map,
    resolve_selection,
    select_job,
)
from tests.test_hitl import JD, PROFILE, TARGET_DATE, install_stubs

APP = str(Path(__file__).parent.parent / "app" / "main.py")

TODAY = date(2026, 8, 9)
COMPANY = "테크노베이션"
WIRED_NODES = [name for name, _ in node_sequence(interactive=True)]


# --- 도우미 -------------------------------------------------------------------


def job(title: str, url: str, doc_id: str | None = None) -> SourceDocument:
    """발견된 공고 — **본문이 없다**(T19 `discover_jobs`의 반환 모양)."""
    return SourceDocument(
        doc_id=doc_id or f"job-{abs(hash(url)) % 10**12:012d}",
        source_type=SourceType.JD,
        company=COMPANY,
        title=title,
        url=url,
        collected_at=TODAY,
        raw_text="",
        confidence=Confidence.MID,
    )


JOBS = [
    job("백엔드 엔지니어 (신입)", "https://x.example/jobs/101", "job-be"),
    job("AI 플랫폼 엔지니어", "https://x.example/jobs/102", "job-ai"),
    job("프론트엔드 개발자", "https://x.example/jobs/103", "job-fe"),
]


def body_for(url: str) -> SourceDocument:
    """`fetch_jd_body`가 돌려줄 법한 본문 문서."""
    return SourceDocument(
        doc_id=f"jd-{url.rsplit('/', 1)[-1]}",
        source_type=SourceType.JD,
        company=COMPANY,
        title="가져온 공고",
        url=url,
        collected_at=TODAY,
        raw_text=JD.raw_text,
        confidence=Confidence.MID,
    )


def install_job_stubs(monkeypatch, *, jobs=None, bodies=True) -> dict[str, list]:
    """발견·본문 수집을 갈아끼운다. **원본 모듈에 건다**(D49).

    `bodies=False`면 본문 수집이 전부 실패하는 상황(빈 문서)을 흉내낸다.
    """
    seen: dict[str, list] = {"discover": [], "fetch": []}

    def fake_discover(company, role="", **kw):
        seen["discover"].append((company, role))
        return list(JOBS if jobs is None else jobs)

    def fake_fetch(url_or_text, *, company="", title=None, **kw):
        seen["fetch"].append(url_or_text)
        if not bodies:
            return body_for(url_or_text).model_copy(update={"raw_text": ""})
        return body_for(url_or_text)

    monkeypatch.setattr(select_mod, "discover_jobs_tool", fake_discover)
    monkeypatch.setattr(select_mod, "fetch_jd_body", fake_fetch)
    return seen


def discovery_state(**overrides) -> dict:
    """JD 본문 없이 시작하는 상태 — 발견 경로."""
    state = {
        "mode": "analysis",
        "company": COMPANY,
        "role": "백엔드 엔지니어",
        "target_date": TARGET_DATE,
        "profile": PROFILE,
        "raw_jd_input": "",
    }
    state.update(overrides)
    return state


# --- ① 배선 --------------------------------------------------------------------


def test_h1_nodes_are_wired_before_collect_in_the_interactive_path():
    """**뮤테이션 M01 — `INTERACTIVE_INSERTS`에서 H1을 빼면 여기가 죽는다.**"""
    assert WIRED_NODES.index("ingest_pasted_jd") + 1 == WIRED_NODES.index("discover_jobs")
    assert WIRED_NODES.index("discover_jobs") + 1 == WIRED_NODES.index("select_job")
    # 고른 공고의 본문이 수집·추출로 흘러가야 한다.
    assert WIRED_NODES.index("select_job") < WIRED_NODES.index("collect")
    assert WIRED_NODES.index("collect") < WIRED_NODES.index("extract")


def test_h1_is_absent_from_the_non_interactive_graph():
    """물어볼 사람이 없으면 발견해 봐야 **시스템이 골라야 하고, 그건 금지다**(§12-2).

    checkpointer 없는 그래프에서 `interrupt()`가 터지는 것을 막는 장치이기도 하다.
    """
    assert "discover_jobs" not in NODE_NAMES
    assert "select_job" not in NODE_NAMES


@pytest.mark.parametrize("node", ["discover_jobs", "select_job"])
def test_h1_nodes_have_user_facing_labels(node):
    """내부 노드명이 화면에 새지 않아야 한다 (T13 불변식)."""
    from app.progress import load_labels

    labels = load_labels()
    assert node in labels.nodes
    assert labels.node(node).label != labels.fallback


# --- ② 생략 조건 — 사용자가 이미 공고를 준 경우 ---------------------------------


def test_user_input_skips_discovery_entirely(monkeypatch):
    """**뮤테이션 M02 — 생략 조건을 지우면 여기가 죽는다.**

    붙여넣기 경로가 네트워크를 한 번도 더 안 타는 근거이며, 기존 오프라인
    테스트 전부가 여기에 기대고 있다.
    """
    seen = install_job_stubs(monkeypatch)
    state = discovery_state(raw_jd_input=JD.raw_text)

    assert discover(state) == {}
    assert select_job(state) == {}
    assert seen["discover"] == [], "사용자가 공고를 줬는데 발견을 돌렸다"


def test_blank_company_discovers_nothing_without_calling_the_tool(monkeypatch):
    seen = install_job_stubs(monkeypatch)
    assert discover(discovery_state(company="  ")) == {"discovered_jobs": []}
    assert seen["discover"] == []


def test_discover_passes_company_and_role_through(monkeypatch):
    seen = install_job_stubs(monkeypatch)
    update = discover(discovery_state())

    assert seen["discover"] == [(COMPANY, "백엔드 엔지니어")]
    assert update["discovered_jobs"] == JOBS


def test_select_job_asks_nothing_when_discovery_found_nothing(monkeypatch):
    """발견 0건은 실패가 아니다 — 묻지 않고 지나간다(§6-2 조용히 스킵)."""
    install_job_stubs(monkeypatch)
    assert select_job(discovery_state(discovered_jobs=[])) == {}


# --- ③ 라벨 — 답변 값이므로 유일해야 한다 ----------------------------------------


def test_labels_are_grouped_and_ordered():
    labels = job_labels(JOBS)
    assert labels[0].startswith("[AI·ML]"), "그룹 순서가 표를 따라야 한다"
    assert any(label.startswith("[백엔드]") for label in labels)


def test_duplicate_titles_get_distinct_labels():
    """**뮤테이션 M03 — 일련번호를 빼면 여기가 죽는다.**

    라벨이 곧 답변 값이라, 두 공고가 같은 라벨이면 어느 쪽을 고른 건지 알 수 없다.
    제목만 같고 부서가 다른 공고는 실제로 흔하다.
    """
    twins = [
        job("백엔드 엔지니어", "https://x.example/jobs/1", "job-1"),
        job("백엔드 엔지니어", "https://x.example/jobs/2", "job-2"),
    ]
    mapping = label_map(twins)

    assert len(mapping) == 2, "같은 라벨로 뭉개졌다"
    assert [j.doc_id for j in mapping.values()] == ["job-1", "job-2"]


def test_labels_never_contain_the_multi_separator():
    """구분자가 라벨 안에 있으면 선택 하나가 둘로 쪼개진다."""
    messy = [job("백엔드\n엔지니어\t(신입)", "https://x.example/jobs/1")]
    assert all(MULTI_SEPARATOR not in label for label in job_labels(messy))


# --- ④ 응답 해석 -----------------------------------------------------------------


def test_resolve_accepts_the_joined_string_from_the_form():
    mapping = label_map(JOBS)
    labels = list(mapping)
    replies = {SELECTION_KEY: MULTI_SEPARATOR.join([labels[0], labels[2]])}

    chosen = resolve_selection(replies, mapping)
    assert [j.doc_id for j in chosen] == [mapping[labels[0]].doc_id, mapping[labels[2]].doc_id]


@pytest.mark.parametrize(
    "make",
    [
        lambda labels: {SELECTION_KEY: labels[0]},
        lambda labels: [labels[0]],
        lambda labels: labels[0],
        lambda labels: {SELECTION_KEY: [labels[0]]},
    ],
    ids=["dict-문자열", "목록", "단일값", "dict-목록"],
)
def test_resolve_accepts_every_shape_a_caller_might_send(make):
    mapping = label_map(JOBS)
    labels = list(mapping)
    chosen = resolve_selection(make(labels), mapping)
    assert [j.doc_id for j in chosen] == [mapping[labels[0]].doc_id]


def test_unknown_labels_are_dropped_silently():
    """없는 공고를 지어내지 않는다."""
    mapping = label_map(JOBS)
    assert resolve_selection({SELECTION_KEY: "[백엔드] 있지도 않은 공고"}, mapping) == []
    assert resolve_selection(None, mapping) == []


def test_result_order_follows_the_options_not_the_reply():
    """**뮤테이션 M04 — 응답 순서를 그대로 쓰면 여기가 죽는다.**

    같은 선택이면 같은 결과가 나와야 `selected_job_ids`가 화면 순서와 일치한다.
    """
    mapping = label_map(JOBS)
    labels = list(mapping)
    reversed_reply = {SELECTION_KEY: MULTI_SEPARATOR.join(reversed(labels))}

    chosen = resolve_selection(reversed_reply, mapping)
    assert [j.doc_id for j in chosen] == [mapping[label].doc_id for label in labels]


# --- ⑤ 중단 페이로드 ---------------------------------------------------------------


def test_payload_is_one_multiselect_question():
    """공고마다 묻지 않는다 — 질문 하나에 선택지 여럿(§9-3 배치 원칙)."""
    payload = build_selection_payload(JOBS)

    assert payload["kind"] == "job_selection"
    assert len(payload["questions"]) == 1

    question = payload["questions"][0]
    assert question["multi"] is True, "**뮤테이션 M05 — multi를 끄면 라디오가 된다**"
    assert len(question["options"]) == len(JOBS)


def test_payload_survives_the_ui_normalizer():
    """`app/hitl.py`가 이 페이로드를 multiselect로 읽는가 (계약 대조)."""
    prompt = normalize_prompt(build_selection_payload(JOBS))

    assert prompt.title == "어느 공고로 분석할까요"
    assert len(prompt.questions) == 1
    assert prompt.questions[0].multi is True
    assert prompt.questions[0].key == SELECTION_KEY


# --- ⑥ 노드 동작 (중단 없이 재개 값만 주입) ------------------------------------------


def resume_with(monkeypatch, labels, *, bodies=True):
    """`interrupt()`를 재개 값 주입으로 대체해 노드 본문만 돌린다.

    그래프를 태우는 관통은 아래 ⑦이 따로 본다. 여기서는 선택 → 본문 수집 규칙을
    좁게 보기 위해 중단 메커니즘을 걷어낸다.
    """
    seen = install_job_stubs(monkeypatch, bodies=bodies)
    monkeypatch.setattr(
        select_mod, "interrupt", lambda payload: {SELECTION_KEY: MULTI_SEPARATOR.join(labels)}
    )
    return seen, select_job(discovery_state(discovered_jobs=list(JOBS)))


def test_only_the_chosen_jobs_are_fetched(monkeypatch):
    """**뮤테이션 M06 — 전부 받아 오면 여기가 죽는다.** 고른 것만 가져온다."""
    labels = job_labels(JOBS)
    seen, update = resume_with(monkeypatch, [labels[0]])

    assert len(seen["fetch"]) == 1, "고르지 않은 공고까지 받아 왔다"
    assert len(update["selected_job_ids"]) == 1


def test_multiple_jobs_can_be_analysed_together(monkeypatch):
    """**카드 완료 조건** — 다건 선택이 그대로 상태에 반영된다."""
    labels = job_labels(JOBS)
    seen, update = resume_with(monkeypatch, labels[:2])

    assert len(seen["fetch"]) == 2
    assert len(update["selected_job_ids"]) == 2
    fetched = [d for d in update["source_docs"] if d.raw_text]
    assert len(fetched) == 2


def test_choosing_nothing_is_a_valid_answer(monkeypatch):
    """아무것도 안 고른 것도 사용자의 선택이다 — **대신 고르지 않는다**(§12-2)."""
    seen, update = resume_with(monkeypatch, [])

    assert update == {"selected_job_ids": []}
    assert seen["fetch"] == [], "고른 게 없는데 받아 왔다"


def test_a_job_whose_body_cannot_be_fetched_is_dropped(monkeypatch):
    """수집 실패는 빈 문서다(D52) — 빈 문서를 고른 공고로 세면 게이트가 헛통과한다."""
    labels = job_labels(JOBS)
    _, update = resume_with(monkeypatch, labels[:2], bodies=False)

    assert update["selected_job_ids"] == []
    assert all(not d.raw_text for d in update["source_docs"])


def test_existing_documents_are_kept(monkeypatch):
    """앞 노드가 넣어 둔 문서를 밀어내지 않는다."""
    install_job_stubs(monkeypatch)
    monkeypatch.setattr(
        select_mod,
        "interrupt",
        lambda payload: {SELECTION_KEY: job_labels(JOBS)[0]},
    )
    existing = SourceDocument(
        doc_id="values-1",
        source_type=SourceType.VALUES,
        company=COMPANY,
        title="인재상",
        collected_at=TODAY,
        raw_text="인재상 본문",
        confidence=Confidence.MID,
    )
    update = select_job(discovery_state(discovered_jobs=list(JOBS), source_docs=[existing]))

    assert existing in update["source_docs"]


# --- ⑦ 그래프 관통: 중단 → multiselect → 재개 ----------------------------------------


@pytest.fixture
def h1_graph():
    return build_analysis_graph(InMemorySaver(), interactive=True)


def test_graph_pauses_at_h1_and_resumes_with_the_selection(monkeypatch, h1_graph):
    """**카드 완료 조건 · 자동화.** 다건 상황에서 고르고 재개하면 상태에 반영된다.

    **중단이 둘이라는 것까지 여기서 고정한다.** H1을 재개하면 분석이 끝나는 게
    아니라 델타 인터뷰(H3)에서 한 번 더 선다 — 그게 정상이고, 그 순서가 뒤집히거나
    한쪽이 사라지면 여기가 죽는다.
    """
    install_stubs(monkeypatch)
    seen = install_job_stubs(monkeypatch)

    # ① H1 — 공고 선택
    status = resume_or_start(h1_graph, "t-h1", initial_input=discovery_state())
    assert status.phase is ThreadPhase.INTERRUPTED, "공고가 여럿인데 안 멈췄다"

    prompt = normalize_prompt(status.questions[0].payload)
    assert prompt.kind == "job_selection"
    assert seen["fetch"] == [], "고르기 전에 본문을 받아 왔다"

    labels = prompt.questions[0].options
    status = resume_or_start(
        h1_graph, "t-h1", resume={SELECTION_KEY: MULTI_SEPARATOR.join(labels[:2])}
    )

    assert len(status.values["selected_job_ids"]) == 2, "고른 공고가 상태에 반영돼야 한다"
    assert len(seen["fetch"]) == 2, "고른 것만 받아 온다"

    # ② H3 — 델타 인터뷰. 선택 결과를 들고 흐름이 이어진다.
    assert status.phase is ThreadPhase.INTERRUPTED
    interview = normalize_prompt(status.questions[0].payload)
    assert interview.kind == "delta_interview"

    status = resume_or_start(
        h1_graph, "t-h1", resume={q.key: "직접 담당했습니다." for q in interview.questions}
    )

    assert status.phase is ThreadPhase.COMPLETE
    assert status.values["brief"] is not None, "선택 뒤 분석이 끝까지 갔다"
    assert status.values["brief"].meta.selected_jobs == status.values["selected_job_ids"]


def test_h1_does_not_fire_when_the_user_pasted_a_jd(monkeypatch, h1_graph):
    """붙여넣기 경로는 H1을 통째로 건너뛴다 — 첫 중단은 델타 인터뷰다."""
    install_stubs(monkeypatch)
    seen = install_job_stubs(monkeypatch)

    status = resume_or_start(
        h1_graph, "t-paste", initial_input=discovery_state(raw_jd_input=JD.raw_text)
    )

    assert status.phase is ThreadPhase.INTERRUPTED
    assert normalize_prompt(status.questions[0].payload).kind == "delta_interview"
    assert seen["discover"] == []


def test_resuming_h1_does_not_rerun_upstream(monkeypatch, h1_graph):
    """T12의 불변식이 중단점 둘에서도 유지된다 — 발견을 다시 돌리지 않는다."""
    install_stubs(monkeypatch)
    seen = install_job_stubs(monkeypatch)

    status = resume_or_start(h1_graph, "t-once", initial_input=discovery_state())
    labels = normalize_prompt(status.questions[0].payload).questions[0].options

    resume_or_start(h1_graph, "t-once", resume={SELECTION_KEY: labels[0]})

    assert len(seen["discover"]) == 1, "재개가 발견을 다시 돌렸다"


# --- ⑧ 화면 관통 (AppTest) ------------------------------------------------------------


def run_app(monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_app_shows_the_job_multiselect_and_resumes(monkeypatch):
    """**카드 완료 조건을 화면까지** — JD를 비우고 실행 → multiselect → 재개.

    `AppTest`는 브라우저가 아니다(D41·D48). 실제 확인은 DEVLOG D65에 있다.
    """
    install_stubs(monkeypatch)
    seen = install_job_stubs(monkeypatch)

    at = run_app(monkeypatch)
    at.text_input[0].set_value(COMPANY)
    at.text_input[1].set_value("백엔드 엔지니어")
    # JD 본문은 비운 채로 제출한다 — 이것이 발견 경로의 방아쇠다.
    at = at.button[0].click().run()
    assert not at.exception

    # ① H1 — multiselect가 떴고 결과는 아직 없다.
    assert at.multiselect, "공고 선택 위젯이 안 떴다"
    assert at.subheader[0].value == "어느 공고로 분석할까요"
    assert not at.download_button
    options = at.multiselect[0].options
    assert len(options) == len(JOBS)

    # **뮤테이션 M26** — 제출 문구는 중단점에 맞아야 한다.
    assert at.button[0].label == "이 공고로 분석 시작"

    at.multiselect[0].set_value(options[:2])
    at = at.button[0].click().run()
    assert not at.exception

    # **뮤테이션 M21** — 고른 **두 건이 다** 넘어가야 한다. 위젯이 첫 항목만
    # 보내도 흐름은 그대로 끝까지 가므로, 완주 여부로는 이 회귀를 못 잡는다.
    assert len(seen["fetch"]) == 2, "multiselect가 고른 것 중 일부만 넘어갔다"

    # ② H3 — 선택 뒤 델타 인터뷰로 이어진다. 답하면 결과가 뜬다.
    assert at.text_area, "선택 뒤 흐름이 이어지지 않았다"
    for widget in at.text_area:
        widget.set_value("직접 담당했습니다.")
    at = at.button[0].click().run()

    assert at.download_button, "선택 후 결과 화면까지 못 갔다"
    assert at.download_button[0].label == "전략 브리프 .md 다운로드"


def test_app_still_requires_company_and_role(monkeypatch):
    """JD는 선택이 됐지만 회사·직무는 여전히 필수다."""
    install_stubs(monkeypatch)
    install_job_stubs(monkeypatch)

    at = run_app(monkeypatch)
    at.button[0].click().run()

    assert at.error, "회사·직무 없이 실행됐다"
    assert not at.multiselect


def test_app_notice_matches_the_checkpoint(monkeypatch):
    """**뮤테이션 M07 — 안내 문구를 델타 인터뷰 것으로 되돌리면 여기가 죽는다.**

    T12의 문구("판정 근거가 부족한 항목이…")가 공고 선택 화면에 뜨면 사용자는
    무슨 말인지 알 수 없다.
    """
    install_stubs(monkeypatch)
    install_job_stubs(monkeypatch)

    at = run_app(monkeypatch)
    at.text_input[0].set_value(COMPANY)
    at.text_input[1].set_value("백엔드 엔지니어")
    at.button[0].click().run()

    notices = " ".join(info.value for info in at.info)
    assert "공고" in notices
    assert "판정 근거가 부족" not in notices

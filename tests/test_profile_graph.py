"""T24 · profile_graph + ProfileJSON 저장/로드.

카드의 완료 조건은 하나다 — **프로필 다운로드 → UC-2에서 업로드 → 델타 인터뷰
질문 수가 유의미하게 감소.** 검증란은 "수동 — 업로드 전후 질문 수 비교"라고
적혀 있으나 수동으로 남길 이유가 없다. T12가 같은 종류의 조건("처음부터 다시
돌지 않는다")을 **호출 횟수로** 자동 증명한 전례가 있고, 수동 기록은 다음에
누가 회귀를 넣어도 아무 소리를 내지 않는다.

그래서 이 파일의 중심은 `test_uc1_profile_cuts_uc2_interview_questions`이며,
증명 방식은 **같은 JD를 두 번 분석해 미결 질문 수를 비교**하는 것이다.
프로필은 실제로 UC-1 그래프를 돌려 만들고, 직렬화 → 역직렬화(= 다운로드 →
업로드)를 실제로 거친 뒤 UC-2에 넣는다.

층은 다섯이다:
    ① 노드     — `parse_resume`가 층 신뢰도를 문서로 옮기는가 (순수)
    ② 영속     — 저장/로드가 같은 바이트를 왕복하는가 (순수)
    ③ 그래프   — H2에서 멈추고, 재개가 상류를 다시 돌리지 않는가
    ④ 완료조건 — UC-1 산출이 UC-2의 질문 수를 줄이는가
    ⑤ 화면     — `AppTest`로 모드 전환 → 업로드 → 설문 → 다운로드까지 관통

**LLM·임베딩은 한 번도 타지 않는다.** `extract_competencies`(T04)와
`retrieve_candidates`(T14)가 각각 네트워크를 왕복하므로 둘 다 갈아끼운다 —
특히 후자는 `nodes/level_survey.py`가 사전 채움에 쓰는 것이라, 안 막으면
설문이 뜰 때마다 임베딩 API가 돈다(D66·D71이 잡은 누수와 같은 자리다).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from app.hitl import SKIP_LABEL, normalize_prompt
from app.progress import ProgressTracker, load_labels
from contracts.enums import Confidence, Importance, Level, SourceType
from contracts.models import (
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    ProfileJSON,
    Question,
    SourceDocument,
)
from graphs.analysis_graph import build_analysis_graph
from graphs.profile_graph import (
    PROFILE_NODE_NAMES,
    RESUME_DOC_ID,
    RESUME_PATH_KEY,
    build_profile_graph,
    build_profile_node,
    download_filename,
    initial_profile_state,
    load_saved_profile,
    parse_resume_node,
    profile_bytes,
    profile_node_sequence,
    save_profile,
)
from app.main import (
    GRAPH_KEY,
    MODE_ANALYSIS,
    MODE_KEY,
    MODE_LABELS,
    MODE_PROFILE,
    PROFILE_NS,
    TRACKER_KEY,
)
from graphs.session import (
    THREAD_KEY,
    ThreadPhase,
    build_checkpointer,
    inspect_thread,
    resume_or_start,
)
from nodes import analysis_nodes, interview, level_survey as level_survey_mod
from nodes.level_survey import KIND, load_survey
from tools.verify import question_id_for

import graphs.profile_graph as profile_mod

FIXTURES = Path(__file__).parent.parent / "fixtures"
APP = str(Path(__file__).parent.parent / "app" / "main.py")
PROFILE_LABELS_PATH = Path(__file__).parent.parent / "presets" / "profile_labels.yaml"

RESUME_PDF = FIXTURES / "resume_text.pdf"

JD = SourceDocument.model_validate_json((FIXTURES / "jd_sample_backend.json").read_bytes())

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
VERDICTS_BY_ID = {
    v["criterion_id"]: CriterionVerdict.model_validate(v)
    for v in json.loads((FIXTURES / "verdicts_all_met.json").read_text("utf-8"))
}

SCORED_COMP_IDS = sorted(set(CRITERIA_ALL) & {c.comp_id for c in BACKEND_REQUIRED})

# 이력서가 실제로 증명해 주는 역량 — 골든 픽스처의 요구 역량 중 앞의 둘이다.
# **이름을 지어내지 않는 것이 요점이다**(R5): UC-2에서 "프로필이 이 역량을
# 덮는가"를 이름으로 잇기 때문에, 양쪽이 같은 픽스처에서 나와야 한다.
COVERED_COMP_IDS = SCORED_COMP_IDS[:2]
COVERED_NAMES = {c.name for c in BACKEND_REQUIRED if c.comp_id in COVERED_COMP_IDS}

TARGET_DATE = date.today() + timedelta(days=90)

RESUME_TEXT = "\n".join(
    ["경력기술서", *sorted(COVERED_NAMES), "직전 회사에서 위 업무를 직접 담당했습니다."]
)


# --- 스텁 ---------------------------------------------------------------------


def install_profile_stubs(monkeypatch, *, resume_text: str = RESUME_TEXT) -> Counter:
    """UC-1이 네트워크를 타는 지점 셋을 전부 갈아끼운다.

    ① `parse_resume` — 파일을 실제로 읽지 않는다(픽스처 PDF는 ①에서 따로 본다)
    ② `extract_competencies` — LLM
    ③ `retrieve_candidates` — 임베딩. **`nodes.level_survey`에 건다** — 사전 채움이
       거기서 부르므로 다른 모듈에 걸면 그대로 샌다(D71).

    반환값은 호출 횟수 카운터다. "재개가 상류를 다시 돌리지 않는다"를 T12와 같은
    방식으로 증명하는 데 쓰며, **OCR이 두 번 돌지 않는 것**까지 여기서 걸린다.
    """
    calls: Counter = Counter()

    def fake_parse(file_path):
        calls["parse_resume"] += 1
        return resume_text, Confidence.HIGH

    def fake_extract(docs, role):
        calls["extract_competencies"] += 1
        if not docs:
            return []
        doc = docs[0]
        return [
            CompetencyRecord(
                comp_id=f"req-{doc.doc_id}-{i:02d}",
                category=comp.category,
                name=comp.name,
                importance=Importance.REQUIRED,
                level=Level.USED,
            )
            for i, comp in enumerate(
                (c for c in BACKEND_REQUIRED if c.name in resume_text), start=1
            )
        ]

    def fake_bind(axes, owned, top_k=3, embed=None):
        """축 ↔ 보유 역량을 **자리 순서로** 잇는다.

        유사도를 흉내내지 않는다 — 그건 T14가 실제 임베딩으로 이미 재는 것이고,
        여기서 볼 것은 "이어진 결과가 사전 채움·좌표 키로 이어지는가"뿐이다.
        축 이름(프리셋)과 역량 이름(이력서)은 애초에 같을 수가 없어서, 이름으로
        잇는 스텁은 **한 건도 못 잇고 조용히 빈 결과를 준다.**
        """
        calls["retrieve_candidates"] += 1
        return [(axis.comp_id, own.comp_id) for axis, own in zip(axes, owned)]

    monkeypatch.setattr(profile_mod, "parse_resume", fake_parse)
    monkeypatch.setattr(analysis_nodes, "extract_competencies", fake_extract)
    monkeypatch.setattr(level_survey_mod, "retrieve_candidates", fake_bind)
    return calls


def install_analysis_stubs(monkeypatch) -> Counter:
    """UC-2 쪽 스텁. **`verify`만 프로필을 실제로 본다.**

    T12의 `install_stubs`를 그대로 쓸 수 없다 — 그쪽 `fake_verify`는 프로필을
    무시하고 정해진 기준만 질문으로 승격하므로, 프로필이 좋아져도 질문 수가
    한 톨도 안 줄어 이 카드의 완료 조건을 잴 수가 없다.

    여기 것은 `verify_criteria`의 계약을 규칙으로 옮긴 것이다 — **근거가 있으면
    판정하고 없으면 Question으로 승격한다.** 근거는 "프로필에 같은 이름의 역량이
    있는가"로 보며, 이름은 양쪽 다 골든 픽스처에서 온다(R5).
    """
    calls: Counter = Counter()

    def fake_extract(docs, role):
        calls["extract_competencies"] += 1
        return list(BACKEND_REQUIRED) if docs else []

    def fake_decompose(comps):
        calls["decompose_criteria"] += 1
        comp_ids = {c.comp_id for c in comps}
        return {k: list(v) for k, v in CRITERIA_ALL.items() if k in comp_ids}

    def fake_verify(criteria, profile, answers=None):
        calls["verify_criteria"] += 1
        answers = answers or {}
        owned_names = {c.name for c in profile.competencies}
        by_comp = {c.comp_id: c for c in BACKEND_REQUIRED}

        decided: list[CriterionVerdict] = []
        questions: list[Question] = []
        for criterion in criteria:
            comp = by_comp.get(criterion.comp_id)
            has_evidence = comp is not None and comp.name in owned_names
            answered = question_id_for(criterion.criterion_id) in answers
            if has_evidence or answered:
                decided.append(VERDICTS_BY_ID[criterion.criterion_id])
            else:
                questions.append(
                    Question(
                        question_id=question_id_for(criterion.criterion_id),
                        criterion_id=criterion.criterion_id,
                        text=f"{criterion.text} — 해당 경험이 있나요?",
                    )
                )
        return decided, questions

    def fake_fill(brief):
        calls["fill_brief_slots"] += 1
        return brief.model_copy(update={"summary_line": "슬롯이 채워졌다"})

    def fake_retrieve(required, owned, top_k=3):
        calls["retrieve_candidates"] += 1
        by_name = {c.name: c for c in owned}
        return [(r.comp_id, by_name[r.name].comp_id) for r in required if r.name in by_name]

    monkeypatch.setattr(analysis_nodes, "extract_competencies", fake_extract)
    monkeypatch.setattr(analysis_nodes, "decompose_criteria", fake_decompose)
    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(analysis_nodes, "verify_criteria", fake_verify)
    monkeypatch.setattr(analysis_nodes, "fill_brief_slots", fake_fill)
    # **델타 인터뷰가 `verify_criteria`를 따로 import 해 간다.** 여기를 빼먹으면
    # 재판정 한 번이 실 API로 새고, 크레딧이 있는 동안은 초록불인 채로 과금된다
    # (T12 `install_stubs`가 같은 두 곳을 거는 이유다).
    monkeypatch.setattr(interview, "verify_criteria", fake_verify)
    return calls


@pytest.fixture
def profile_graph():
    return build_profile_graph(InMemorySaver())


@pytest.fixture(autouse=True)
def isolated_profile_dir(tmp_path, monkeypatch):
    """저장 위치를 테스트마다 갈아끼운다.

    안 하면 `build_profile` 노드가 저장소 안 `.jobprep/profiles/`에 파일을 쌓고,
    **그 파일이 다음 테스트의 `load_saved_profile()`에 잡힌다** — 테스트가
    파일 시스템 상태에 따라 달라지는 순간 그물이 아니라 소음이 된다.
    """
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")
    return tmp_path / "profiles"


def survey_answers(status, *, answered: set[str] | None = None) -> dict[str, str]:
    """H2 폼에 답한다. 사전 채움된 값이 있으면 그대로 확정한다."""
    prompt = normalize_prompt(status.questions[0].payload)
    answers: dict[str, str] = {}
    for question in prompt.questions:
        if question.options is None:
            continue  # 근거 한 줄 칸 — 비워 둔다
        if answered is not None and question.key not in answered:
            continue
        answers[question.key] = question.default or question.options[0]
    return answers


# --- ① 노드 --------------------------------------------------------------------


def test_parse_layer_confidence_rides_on_the_document(monkeypatch):
    """**층 신뢰도가 문서로 옮겨진다.** 상태에 담을 칸이 그것뿐이다."""
    monkeypatch.setattr(profile_mod, "parse_resume", lambda p: ("본문", Confidence.MID))

    update = parse_resume_node(initial_profile_state("/tmp/resume.pdf"))

    doc = update["source_docs"][0]
    assert doc.confidence is Confidence.MID
    assert doc.raw_text == "본문"
    assert doc.doc_id == RESUME_DOC_ID
    assert doc.source_type is SourceType.OTHER


def test_resume_document_id_never_collides_with_a_jd():
    """id가 겹치면 UC-2에서 요구 역량 id와 보유 역량 id가 충돌한다."""
    assert RESUME_DOC_ID != analysis_nodes.PASTED_DOC_ID


def test_no_resume_means_no_document(monkeypatch):
    """이력서 없이 설문만 도는 경로 — 빈 문서를 지어내지 않는다."""
    monkeypatch.setattr(profile_mod, "parse_resume", lambda p: pytest.fail("파싱을 시도했다"))

    assert parse_resume_node(initial_profile_state("")) == {"source_docs": []}


def test_failed_parse_still_yields_a_document(monkeypatch):
    """**T21 계약 ②** — 실패해도 건진 것을 넘긴다. H4(T22b)가 고칠 거리가 있어야 한다."""
    monkeypatch.setattr(profile_mod, "parse_resume", lambda p: ("", Confidence.LOW))

    doc = parse_resume_node(initial_profile_state("/tmp/scan.pdf"))["source_docs"][0]

    assert doc.raw_text == ""
    assert doc.confidence is Confidence.LOW


def test_real_fixture_resume_parses_through_the_node():
    """스텁 없이 **실제 PDF 픽스처**로 한 번 지나간다 (R5).

    스텁만으로 통과시키면 `parse_resume`의 반환 모양이 바뀌어도 이 노드는 계속
    초록불이다 — T21이 돌려주는 것이 정말 `(str, Confidence)` 튜플인지 여기서 건다.
    """
    doc = parse_resume_node(initial_profile_state(str(RESUME_PDF)))["source_docs"][0]

    assert doc.confidence is Confidence.HIGH, "텍스트 레이어가 있는 PDF는 신뢰도 상이다"
    assert len(doc.raw_text) > 200


# --- ② 영속 --------------------------------------------------------------------


def make_profile(**overrides) -> ProfileJSON:
    base = {
        "competencies": [
            CompetencyRecord(
                comp_id="pf-01",
                category=BACKEND_REQUIRED[0].category,
                name=BACKEND_REQUIRED[0].name,
                importance=Importance.REQUIRED,
                level=Level.OPERATED,
            )
        ],
        "level_coordinates": {"pf-01": Level.OPERATED},
        "coverage": {BACKEND_REQUIRED[0].category: 0.5},
        "built_at": date(2026, 8, 11),
    }
    return ProfileJSON(**{**base, **overrides})


def test_download_upload_roundtrip_preserves_everything():
    """**다운로드 → 업로드가 무손실이어야** 카드의 완료 조건이 성립한다."""
    profile = make_profile()

    restored = ProfileJSON.model_validate_json(profile_bytes(profile))

    assert restored == profile


def test_saved_and_downloaded_bytes_are_identical(tmp_path):
    """두 export 경로가 갈리면 "내려받은 건 되는데 저장된 건 안 되는" 사고가 난다."""
    profile = make_profile()

    path = save_profile(profile, directory=tmp_path)

    assert path.read_bytes() == profile_bytes(profile)


def test_saving_twice_on_the_same_day_does_not_overwrite(tmp_path):
    """이름에 내용 해시가 붙는 이유 — 같은 날 두 번 만들어도 앞엣것이 안 사라진다."""
    first = save_profile(make_profile(), directory=tmp_path)
    second = save_profile(make_profile(coverage={}), directory=tmp_path)

    assert first != second
    assert first.exists() and second.exists()


def test_load_saved_profile_picks_the_newest(tmp_path):
    old = make_profile()
    new = make_profile(level_coordinates={"pf-01": Level.LED})

    save_profile(old, directory=tmp_path)
    saved_new = save_profile(new, directory=tmp_path)
    # mtime 해상도가 낮은 파일 시스템에서도 순서가 서게 못 박는다.
    import os

    os.utime(saved_new, (1e9 + 100, 1e9 + 100))
    os.utime(next(p for p in tmp_path.glob("profile_*.json") if p != saved_new), (1e9, 1e9))

    assert load_saved_profile(directory=tmp_path) == new


def test_load_saved_profile_skips_corrupt_files(tmp_path):
    """깨진 파일 하나가 회사 분석 화면 전체를 막으면 안 된다."""
    good = save_profile(make_profile(), directory=tmp_path)
    broken = tmp_path / "profile_20991231_deadbeef.json"
    broken.write_text("{ 이건 JSON이 아니다", encoding="utf-8")
    import os

    os.utime(broken, (1e9 + 100, 1e9 + 100))
    os.utime(good, (1e9, 1e9))

    assert load_saved_profile(directory=tmp_path) is not None


def test_load_saved_profile_returns_none_when_empty(tmp_path):
    assert load_saved_profile(directory=tmp_path / "없는폴더") is None


def test_download_name_carries_the_build_date():
    assert download_filename(make_profile()) == "내프로필_20260811.json"


def test_uploaded_profile_wins_over_the_saved_one(monkeypatch, tmp_path):
    """우선순위는 **방금 지정한 것 → 직전 산출 → 샘플**이다.

    거꾸로 두면 사용자가 파일을 골라 올렸는데도 예전 프로필로 분석이 돈다 —
    화면에는 아무 표시도 안 나므로 조용히 틀린 결과가 나가는 종류의 결함이다.
    """
    from app.main import load_profile

    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")
    save_profile(make_profile(), directory=tmp_path / "profiles")

    wanted = make_profile(level_coordinates={"pf-01": Level.LED})

    class FakeUpload:
        def getvalue(self):
            return profile_bytes(wanted)

    assert load_profile(FakeUpload(), use_sample=True, use_saved=True) == wanted


def test_save_failure_does_not_lose_the_profile(monkeypatch):
    """저장이 막혀도 프로필은 상태에 남는다 — 정본 export는 다운로드다."""

    def boom(*args, **kwargs):
        raise OSError("디스크가 가득 찼다")

    monkeypatch.setattr(profile_mod, "save_profile", boom)
    profile = make_profile()

    assert build_profile_node({"profile": profile})["profile"] is profile


def test_build_profile_node_does_not_invent_an_empty_profile():
    assert build_profile_node({}) == {}


# --- ③ 그래프 ------------------------------------------------------------------


def test_graph_requires_a_checkpointer():
    """H2가 `interrupt()`로 멈추므로 저장할 곳 없이는 재개가 성립하지 않는다."""
    with pytest.raises(ValueError, match="checkpointer"):
        build_profile_graph(None)


def test_wired_nodes_are_the_four_the_card_specifies():
    """**목록을 다시 하드코딩하지 말 것**(§2-1 D59) — T22b가 H4를 사이에 끼웠다.

    이 카드가 거는 것은 "이 넷이 이 순서로 있다"이지 "이 넷뿐이다"가 아니다.
    전제를 배선에서 유도해 두면 다음 노드가 끼어도 안 깨진다.
    """
    card_nodes = ["parse_resume", "extract", "level_survey", "build_profile"]

    assert [n for n in PROFILE_NODE_NAMES if n in card_nodes] == card_nodes


def test_collection_tools_are_not_wired_into_the_profile_graph():
    """**카드 불변식** — 사용자 프로필과 수집 결과를 혼합하지 않는다.

    "조심한다"가 아니라 **배선의 부재**로 지킨다. 회사 수집 노드가 하나라도
    끼면 이력서 아닌 문서가 `source_docs`에 들어와 "내 역량"에 타사 JD 문장이
    유입된다.
    """
    forbidden = {"collect", "discover_jobs", "select_job", "ingest_pasted_jd", "verify"}

    assert forbidden.isdisjoint(PROFILE_NODE_NAMES)


def test_graph_pauses_at_the_level_survey_and_builds_a_profile(monkeypatch, profile_graph):
    """관통 — 이력서 → 추출 → H2 중단 → 답변 → `ProfileJSON`."""
    install_profile_stubs(monkeypatch)

    status = resume_or_start(
        profile_graph,
        "t-uc1",
        initial_input=initial_profile_state("/tmp/resume.pdf", role="백엔드 엔지니어"),
    )
    assert status.phase is ThreadPhase.INTERRUPTED, "레벨 측정에서 멈춰야 한다"

    prompt = normalize_prompt(status.questions[0].payload)
    assert prompt.kind == KIND

    status = resume_or_start(profile_graph, "t-uc1", resume=survey_answers(status))

    assert status.phase is ThreadPhase.COMPLETE
    profile = status.values["profile"]
    assert isinstance(profile, ProfileJSON)
    assert profile.competencies, "프로필이 비어 있다"
    assert profile.level_coordinates, "레벨 좌표가 안 찍혔다"


def test_resume_extraction_runs_once_even_after_the_survey(monkeypatch, profile_graph):
    """**T12의 불변식을 UC-1에서.** 재개가 이력서를 다시 읽지 않는다.

    여기서 세는 것이 비용이다 — 스캔본이면 `parse_resume`이 OCR(T22)을 타고,
    그건 호출마다 돈이 나가는 경로다(D70). 재개할 때마다 다시 읽으면 사용자가
    설문을 고쳐 제출할 때마다 과금된다.
    """
    calls = install_profile_stubs(monkeypatch)

    status = resume_or_start(
        profile_graph, "t-once", initial_input=initial_profile_state("/tmp/resume.pdf")
    )
    resume_or_start(profile_graph, "t-once", resume=survey_answers(status))

    assert calls["parse_resume"] == 1, "재개가 이력서를 다시 읽었다"
    assert calls["extract_competencies"] == 1, "재개가 추출을 다시 돌렸다"


def test_the_survey_prefills_levels_from_the_resume(monkeypatch, profile_graph):
    """이력서에서 읽은 역량은 **미리 골라진 채로** 뜬다 (T23 사전 채움).

    사전 채움이 죽으면 사용자는 26문항을 맨손으로 채워야 하는데, 화면은 똑같이
    떠서 관통 테스트로는 안 잡힌다 — 여기서 개수를 하나 건다(D65 M21과 같은 이유).
    """
    install_profile_stubs(monkeypatch)

    status = resume_or_start(
        profile_graph, "t-prefill", initial_input=initial_profile_state("/tmp/resume.pdf")
    )

    prompt = normalize_prompt(status.questions[0].payload)
    prefilled = [q for q in prompt.questions if q.default]
    assert prefilled, "이력서에서 읽은 역량이 하나도 미리 골라지지 않았다"


def test_survey_without_a_resume_still_produces_a_profile(monkeypatch, profile_graph):
    """폭 보완만으로도 프로필은 나온다 — 그리고 **LLM을 부르지 않는다.**"""
    calls = install_profile_stubs(monkeypatch)

    status = resume_or_start(profile_graph, "t-noresume", initial_input=initial_profile_state(""))
    assert status.phase is ThreadPhase.INTERRUPTED

    status = resume_or_start(profile_graph, "t-noresume", resume=survey_answers(status))

    assert status.phase is ThreadPhase.COMPLETE
    assert status.values["profile"].level_coordinates
    assert calls["parse_resume"] == 0
    assert calls["extract_competencies"] == 1, "문서가 없어도 노드는 지나간다"


def test_both_graphs_share_one_sqlite_file_without_colliding(monkeypatch, tmp_path):
    """**실행에서만 나는 자리다.** 화면 테스트는 checkpointer를 인메모리로 갈아끼우므로
    (`run_app`) 두 그래프가 sqlite 파일 하나를 함께 쓰는 경로를 한 번도 안 탄다.

    실제 실행에서는 둘 다 `.jobprep/checkpoints.sqlite`를 연다. 스레드 id가
    다르기만 하면 안전하다는 것을 여기서 못 박는다 — 커넥션이 둘이어도 되는지,
    한쪽 스레드가 다른 쪽 그래프에 보이지는 않는지가 걸린다.
    """
    install_profile_stubs(monkeypatch)
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")

    db = tmp_path / "checkpoints.sqlite"
    uc1 = build_profile_graph(build_checkpointer(db))
    uc2 = build_analysis_graph(build_checkpointer(db), interactive=True)

    status = resume_or_start(uc1, "th-1", initial_input=initial_profile_state(""))
    assert status.phase is ThreadPhase.INTERRUPTED

    # 같은 파일을 보는 다른 그래프·다른 스레드는 **미시작**이어야 한다.
    assert inspect_thread(uc2, "th-2").phase is ThreadPhase.NOT_STARTED

    status = resume_or_start(uc1, "th-1", resume=survey_answers(status))
    assert status.phase is ThreadPhase.COMPLETE
    assert isinstance(status.values["profile"], ProfileJSON)


def test_the_finished_profile_lands_on_disk(monkeypatch, profile_graph, isolated_profile_dir):
    """`build_profile` 노드가 실제로 파일을 떨구는가 — §10-3의 "산출은 파일이다"."""
    install_profile_stubs(monkeypatch)

    status = resume_or_start(
        profile_graph, "t-disk", initial_input=initial_profile_state("/tmp/resume.pdf")
    )
    status = resume_or_start(profile_graph, "t-disk", resume=survey_answers(status))

    saved = load_saved_profile(directory=isolated_profile_dir)
    assert saved == status.values["profile"]


# --- ④ 완료 조건: UC-1 산출이 UC-2의 질문 수를 줄인다 ---------------------------


def analysis_input(profile: ProfileJSON | None) -> dict:
    return {
        "mode": "analysis",
        "company": JD.company,
        "role": "백엔드 엔지니어",
        "target_date": TARGET_DATE,
        "profile": profile,
        "raw_jd_input": JD.raw_text,
    }


def build_uc1_profile(monkeypatch) -> ProfileJSON:
    """UC-1을 실제로 돌려 프로필을 만든다 — 손으로 조립하지 않는다."""
    install_profile_stubs(monkeypatch)
    graph = build_profile_graph(InMemorySaver())

    status = resume_or_start(
        graph, "t-uc1-src", initial_input=initial_profile_state("/tmp/resume.pdf")
    )
    status = resume_or_start(graph, "t-uc1-src", resume=survey_answers(status))
    return status.values["profile"]


def open_questions(graph, thread: str, profile: ProfileJSON) -> int:
    status = resume_or_start(graph, thread, initial_input=analysis_input(profile))
    return len(status.values.get("pending_questions") or [])


def test_uc1_profile_cuts_uc2_interview_questions(monkeypatch, capsys):
    """**T24 = 카드 완료 조건.** 프로필 다운로드 → 업로드 → 질문 수가 줄어든다.

    비교 대상은 "프로필 없음"이 아니라 **빈 프로필**이다. `verify` 노드는
    프로필이 `None`이면 아예 판정을 안 하고 질문도 안 만들어(0건) 비교가
    뒤집히기 때문이다 — 줄어들 것이 있으려면 판정이 돌아야 한다.

    프로필은 **직렬화를 실제로 왕복시킨 뒤** 넣는다. 그래야 재는 것이
    "메모리 안의 객체"가 아니라 카드가 말한 **다운로드 → 업로드 경로**가 된다.
    """
    uc1_profile = build_uc1_profile(monkeypatch)

    # 다운로드 → 업로드.
    uploaded = ProfileJSON.model_validate_json(profile_bytes(uc1_profile))

    empty = ProfileJSON(
        competencies=[], level_coordinates={}, coverage={}, built_at=date.today()
    )

    install_analysis_stubs(monkeypatch)
    graph = build_analysis_graph(InMemorySaver(), interactive=True)

    before = open_questions(graph, "t-before", empty)
    after = open_questions(graph, "t-after", uploaded)

    print("\n[델타 인터뷰 질문 수 — 프로필 업로드 전후]")
    print(f"  빈 프로필      {before}건")
    print(f"  UC-1 프로필    {after}건  (내려받아 다시 올린 것)")

    assert before > 0, "빈 프로필이면 물어볼 것이 있어야 비교가 성립한다"
    assert after < before, "UC-1 프로필이 질문 수를 줄이지 못했다"

    # **개수를 하나 건다** — "줄기만 하면 통과"면 한 건만 줄어도 초록불이다.
    # 이력서가 덮은 역량의 기준은 전부 판정돼야 한다.
    covered = sum(len(CRITERIA_ALL[comp_id]) for comp_id in COVERED_COMP_IDS)
    assert before - after == covered


def test_the_uploaded_profile_actually_fills_my_level(monkeypatch):
    """질문만 줄고 판정이 안 붙으면 반쪽이다 — 후보쌍이 `my_level`까지 가는지 본다."""
    uc1_profile = build_uc1_profile(monkeypatch)
    install_analysis_stubs(monkeypatch)
    graph = build_analysis_graph(InMemorySaver(), interactive=True)

    status = resume_or_start(graph, "t-level", initial_input=analysis_input(uc1_profile))
    while status.phase is ThreadPhase.INTERRUPTED:
        prompt = normalize_prompt(status.questions[0].payload)
        status = resume_or_start(
            graph,
            "t-level",
            resume={q.key: "직접 담당했습니다." for q in prompt.questions},
        )

    leveled = [m for m in status.values["match_results"] if m.my_level is not None]
    assert leveled, "UC-1 프로필이 `my_level`을 하나도 못 채웠다"


# --- ⑤ 화면 --------------------------------------------------------------------


def test_every_wired_node_has_a_user_facing_label():
    """배선된 노드는 전부 UC-1 시트에 있어야 한다 — 하나라도 빠지면 화면이 흐려진다."""
    labels = load_labels(PROFILE_LABELS_PATH)
    missing = [node for node in PROFILE_NODE_NAMES if node not in labels.nodes]

    assert missing == [], f"presets/profile_labels.yaml에 없는 노드: {missing}"


def test_the_profile_sheet_has_its_own_titles():
    """**시트를 가른 이유가 이것이다** — 프로필을 만드는데 "회사 분석 진행 중"은 틀렸다."""
    analysis = load_labels()
    profile = load_labels(PROFILE_LABELS_PATH)

    assert profile.title != analysis.title
    assert "분석" not in profile.title


def test_internal_node_names_never_reach_the_profile_screen():
    tracker = ProgressTracker(
        PROFILE_NODE_NAMES, labels=load_labels(PROFILE_LABELS_PATH), clock=lambda: 0.0
    )
    rendered = tracker.as_markdown()

    for node in PROFILE_NODE_NAMES:
        assert node not in rendered, f"내부 함수명 '{node}'이 화면에 노출됐다"


def run_app(monkeypatch):
    import graphs.session as session

    monkeypatch.setattr(session, "build_checkpointer", lambda *a, **kw: InMemorySaver())
    return AppTest.from_file(APP, default_timeout=60).run()


def mode_radio(at):
    """모드 라디오는 **키로 찾는다.**

    `at.radio[0]`으로 잡으면 안 된다 — 사이드바 위젯이 본문보다 뒤에 오므로
    설문이 떠 있는 동안에는 0번이 첫 문항이다. 위치로 잡는 순간 이 파일의
    화면 테스트가 화면 구성에 따라 다른 것을 누른다.
    """
    return next(w for w in at.radio if w.key == MODE_KEY)


def survey_radios(at):
    """레벨 측정 문항만. 모드 라디오를 같이 세면 문항 수 회귀가 안 잡힌다."""
    return [w for w in at.radio if w.key != MODE_KEY]


def switch_mode(at, mode: str):
    mode_radio(at).set_value(MODE_LABELS[mode])
    return at.run()


def switch_to_profile_mode(at):
    return switch_mode(at, MODE_PROFILE)


def test_app_defaults_to_the_analysis_mode(monkeypatch):
    """**기존 화면이 그대로여야 한다** — 모드가 둘이 됐다고 UC-2가 밀리면 안 된다."""
    at = run_app(monkeypatch)

    assert not at.exception
    assert any(w.label == "JD 본문" for w in at.text_area), "회사 분석이 기본 화면이 아니다"


def test_profile_mode_uploader_sits_outside_the_form(monkeypatch):
    """**D48** — 폼 안이면 파일이 서버에 닿는 순간이 곧 제출이라 확인할 틈이 없다.

    `AppTest`는 폼 배칭을 모사하지 않아 동작으로는 이 결함을 못 잡는다(D41).
    그래서 배치 자체를 고정한다 — 업로더의 `form_id`가 비어야 한다.
    """
    at = switch_to_profile_mode(run_app(monkeypatch))

    assert not at.exception
    assert at.file_uploader[0].form_id == "", "이력서 업로더가 폼 안에 있다"


def test_app_walks_uc1_from_upload_to_download(monkeypatch, tmp_path):
    """**화면 관통** — 모드 전환 → 이력서 업로드 → 설문 → ProfileJSON 다운로드."""
    install_profile_stubs(monkeypatch)
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")

    at = switch_to_profile_mode(run_app(monkeypatch))
    at.file_uploader[0].set_value(("resume.pdf", b"%PDF-1.4 fake", "application/pdf"))
    at = at.run()

    at.text_input[0].set_value("백엔드 엔지니어")
    at = at.button[0].click().run()
    assert not at.exception

    # ① H2 — 설문이 떴고 아직 결과는 없다.
    questions = survey_radios(at)
    assert questions, "레벨 측정 폼이 안 떴다"
    assert not at.download_button
    assert at.subheader[0].value == "해본 만큼만 골라주세요"
    assert at.button[0].label == "프로필 만들기"

    survey = load_survey()
    # **개수 회귀** — 한 문항만 떠도 흐름은 끝까지 간다(D65 M21).
    assert len(questions) == len(survey.axes)

    for widget in questions:
        widget.set_value(widget.options[0])
    at = at.button[0].click().run()
    assert not at.exception

    # ② 완료 — 다운로드 버튼과 저장 파일이 함께 있어야 한다.
    assert at.download_button, "프로필 다운로드 버튼이 안 떴다"
    assert at.download_button[0].label == "ProfileJSON 다운로드"

    saved = load_saved_profile(directory=tmp_path / "profiles")
    assert saved is not None, "완성된 프로필이 디스크에 안 남았다"
    assert len(saved.level_coordinates) == len(survey.axes), "고른 답이 다 안 실렸다"


def test_switching_modes_keeps_both_threads(monkeypatch, tmp_path):
    """모드를 오가도 각자의 스레드가 그대로 서 있다 — 네임스페이스가 갈려 있어서다."""
    install_profile_stubs(monkeypatch)
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")

    at = switch_to_profile_mode(run_app(monkeypatch))
    at = at.button[0].click().run()  # 이력서 없이 설문만 — H2에서 멈춘다
    assert at.subheader[0].value == "해본 만큼만 골라주세요"

    at = switch_mode(at, MODE_ANALYSIS)
    assert any(w.label == "JD 본문" for w in at.text_area), "회사 분석이 입력 화면으로 돌아와야 한다"

    at = switch_mode(at, MODE_PROFILE)
    assert at.subheader[0].value == "해본 만큼만 골라주세요", "프로필 설문이 사라졌다"


def test_uc1_keeps_its_session_inside_its_namespace(monkeypatch, tmp_path):
    """**두 모드가 서로의 상태를 밟지 않는다** — 스레드·그래프·트래커가 전부 안쪽에 산다.

    이 결함은 **화면 동작으로는 안 보인다.** 테스트가 모드마다 새 checkpointer를
    쥐여 주기 때문에(`run_app`) 스레드 id가 겹쳐도 서로 다른 저장소를 봐서
    아무 일도 안 일어난다 — 실제 실행에서는 sqlite 파일 **하나**를 공유하므로
    UC-1의 중단이 회사 분석 화면에 뜬다. 그래서 결과가 아니라 **키가 어디에
    사는지**를 직접 건다(D49와 같은 종류의 맹점이다).
    """
    install_profile_stubs(monkeypatch)
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")

    at = switch_to_profile_mode(run_app(monkeypatch))
    at = at.button[0].click().run()

    ns = at.session_state[PROFILE_NS]
    for key in (THREAD_KEY, GRAPH_KEY, TRACKER_KEY):
        assert key in ns, f"UC-1의 '{key}'가 네임스페이스 밖에 있다"

    # UC-2도 자기 스레드·그래프를 갖고 있다(기본 화면이라 이미 만들어졌다).
    # **같은 것을 가리키면 안 된다** — 그 순간 UC-1의 중단이 회사 분석 화면에 뜬다.
    assert at.session_state[THREAD_KEY] != ns[THREAD_KEY]
    assert at.session_state[GRAPH_KEY] is not ns[GRAPH_KEY]


def test_the_profile_panel_uses_the_profile_sheet(monkeypatch, tmp_path):
    """진행 표시가 **UC-1 시트**로 그려지는가.

    시트를 안 갈아 끼우면 프로필을 만드는 내내 "회사 분석 — 답변 대기"가 뜨고
    노드 줄은 전부 fallback("처리 중")이 된다. 흐름은 똑같이 끝까지 가므로
    관통 테스트로는 안 잡힌다.
    """
    install_profile_stubs(monkeypatch)
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")

    at = switch_to_profile_mode(run_app(monkeypatch))
    at = at.button[0].click().run()

    sheet = load_labels(PROFILE_LABELS_PATH)
    assert at.status, "진행 표시 패널이 떠야 한다"
    assert at.status[0].label == sheet.paused_title

    body = "\n".join(block.value for block in at.markdown)
    assert f"⏸ {sheet.node('level_survey').label}" in body
    assert sheet.fallback not in body, "노드 줄이 fallback으로 떨어졌다 — 시트가 안 맞는다"
    for node in PROFILE_NODE_NAMES:
        assert node not in body, f"내부 함수명 '{node}'이 화면에 노출됐다"


def test_saved_profile_is_never_used_unless_asked(monkeypatch, tmp_path):
    """**기본이 꺼짐이다** — 예전 프로필이 사용자도 모르게 판정 근거가 되면 안 된다."""
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")
    save_profile(make_profile(), directory=tmp_path / "profiles")

    at = run_app(monkeypatch)
    checkbox = next(c for c in at.checkbox if "직전에 만든" in c.label)

    assert checkbox.value is False


def test_the_survey_notice_is_not_the_delta_interview_one(monkeypatch, tmp_path):
    """중단점마다 안내가 달라야 한다 (HITL 규약 ④ · D63).

    T12의 문구("판정 근거가 부족한 항목이…")가 레벨 측정 화면에 뜨면 사용자는
    자기가 무엇을 하고 있는지 알 수 없다.
    """
    install_profile_stubs(monkeypatch)
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")

    at = switch_to_profile_mode(run_app(monkeypatch))
    at = at.button[0].click().run()

    notices = " ".join(info.value for info in at.info)
    assert "해본" in notices or "골라" in notices
    assert "판정 근거가 부족" not in notices


def test_skip_label_is_offered_for_every_axis(monkeypatch, tmp_path):
    """"해당 없음"은 레벨 0이 아니라 **모르는 축**이다 — 고를 수 있어야 성립한다."""
    install_profile_stubs(monkeypatch)
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")

    at = switch_to_profile_mode(run_app(monkeypatch))
    at = at.button[0].click().run()

    questions = survey_radios(at)
    assert questions
    for widget in questions:
        assert SKIP_LABEL in widget.options

"""T22b · H4 OCR 보정 (사용자 직접 수정).

카드의 완료 조건은 한 문장이다 — **스캔 이력서 → 인식 결과가 채워진 채로 표시 →
사용자가 고침 → 고친 텍스트가 `extract`로 흘러간다.** 마지막 절이 핵심이라
관통 테스트는 "끝까지 갔다"가 아니라 **`extract`가 실제로 받은 텍스트**를 본다
(D65 M21 — 양이 틀린 회귀를 놓치지 않게 개수도 함께 건다).

층은 다섯이다:
    ① 판별·페이로드 — 언제 멈추고 무엇을 실어 보내는가 (순수)
    ② 보정 적용     — 고친 텍스트가 문서에 얹히는가 (순수)
    ③ 그래프       — 멈추고, 재개하고, 고친 것이 하류로 흐르는가
    ④ 화면         — `AppTest`로 업로드 → 보정 → 설문까지 관통
    ⑤ 배선         — H4가 `parse_resume`과 `extract` 사이에 있는가 (§2-1)

**LLM·임베딩·OCR은 한 번도 타지 않는다.** `parse_resume`(T21·T22)·
`extract_competencies`(T04)·`retrieve_candidates`(T14) 셋을 전부 갈아끼운다 —
특히 마지막은 `nodes/level_survey.py`가 사전 채움에 쓰는 것이라 다른 모듈에
걸면 그대로 샌다(D71).
"""

from __future__ import annotations

import contextlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from app.hitl import KIND_TITLES, SUBMIT_LABELS, normalize_prompt
from app.main import MODE_KEY, MODE_LABELS, MODE_PROFILE, RESUME_NOTICES
from app.progress import load_labels
from contracts.enums import Confidence, Importance, Level, SourceType
from contracts.models import CompetencyRecord, ProfileJSON, SourceDocument
from graphs.profile_graph import (
    PROFILE_NODE_NAMES,
    RESUME_DOC_ID,
    build_profile_graph,
    initial_profile_state,
)
from graphs.session import ThreadPhase, resume_or_start
from nodes import analysis_nodes, level_survey as level_survey_mod
from nodes.level_survey import KIND as SURVEY_KIND
from nodes.ocr_review import (
    CORRECTED_CONFIDENCE,
    KIND,
    apply_corrections,
    build_payload,
    docs_needing_review,
    ocr_review,
)
from tools.parse_resume import needs_manual_correction

import graphs.profile_graph as profile_mod

FIXTURES = Path(__file__).parent.parent / "fixtures"
APP = str(Path(__file__).parent.parent / "app" / "main.py")
PROFILE_LABELS_PATH = Path(__file__).parent.parent / "presets" / "profile_labels.yaml"

ALL_REQUIRED = [
    CompetencyRecord.model_validate(item)
    for item in json.loads((FIXTURES / "competencies_required.json").read_text("utf-8"))
]
BACKEND_REQUIRED = [c for c in ALL_REQUIRED if c.comp_id.startswith("req-be-")]

# 이력서가 증명해 주는 역량 — **이름을 지어내지 않는다**(R5). 골든 픽스처에서 온
# 이름이라야 "고친 텍스트가 흘러갔다"를 역량 개수로 잴 수 있다.
COVERED_NAMES = [c.name for c in BACKEND_REQUIRED[:2]]

# OCR이 실패한 모습. 사람이 읽을 수 없고, 역량 이름도 한 건 안 들어 있다.
GARBLED = "ìê°ìê° ??? \n\n ¿¿¿"
CORRECTED = "\n".join(["경력기술서", *COVERED_NAMES, "위 업무를 직접 담당했습니다."])


def make_doc(text: str, confidence: Confidence, *, doc_id: str = RESUME_DOC_ID) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        source_type=SourceType.OTHER,
        company="",
        title="이력서",
        collected_at=date.today(),
        raw_text=text,
        confidence=confidence,
    )


# --- 스텁 ---------------------------------------------------------------------


def install_stubs(monkeypatch, *, parsed: tuple[str, Confidence]) -> tuple[Counter, list]:
    """UC-1이 네트워크·파일을 타는 지점 셋을 갈아끼운다.

    반환은 (호출 횟수, `extract`가 받은 문서 묶음들)이다. 뒤엣것이 이 카드의
    완료 조건을 재는 자다 — **고친 텍스트가 정말 `extract`까지 갔는지**는
    거기서만 보인다.
    """
    calls: Counter = Counter()
    seen_docs: list[list[SourceDocument]] = []

    def fake_parse(file_path):
        calls["parse_resume"] += 1
        return parsed

    def fake_extract(docs, role):
        calls["extract_competencies"] += 1
        seen_docs.append(list(docs))
        text = "\n".join(d.raw_text for d in docs)
        return [
            CompetencyRecord(
                comp_id=f"req-{RESUME_DOC_ID}-{i:02d}",
                category=comp.category,
                name=comp.name,
                importance=Importance.REQUIRED,
                level=Level.USED,
            )
            for i, comp in enumerate(
                (c for c in BACKEND_REQUIRED if c.name in text), start=1
            )
        ]

    def fake_bind(axes, owned, top_k=3, embed=None):
        calls["retrieve_candidates"] += 1
        return [(axis.comp_id, own.comp_id) for axis, own in zip(axes, owned)]

    monkeypatch.setattr(profile_mod, "parse_resume", fake_parse)
    monkeypatch.setattr(analysis_nodes, "extract_competencies", fake_extract)
    monkeypatch.setattr(level_survey_mod, "retrieve_candidates", fake_bind)
    return calls, seen_docs


@pytest.fixture
def profile_graph():
    return build_profile_graph(InMemorySaver())


@pytest.fixture(autouse=True)
def isolated_profile_dir(tmp_path, monkeypatch):
    """저장 위치를 테스트마다 갈아끼운다 — 안 하면 저장소에 파일이 쌓인다(T24 전례)."""
    monkeypatch.setattr(profile_mod, "PROFILE_DIR", tmp_path / "profiles")
    return tmp_path / "profiles"


def survey_answers(status) -> dict[str, str]:
    prompt = normalize_prompt(status.questions[0].payload)
    return {
        q.key: (q.default or q.options[0]) for q in prompt.questions if q.options
    }


# --- ① 판별·페이로드 ------------------------------------------------------------


def test_a_readable_resume_never_pauses():
    """잘 읽힌 이력서는 **그냥 지나간다.**

    `interrupt()`를 부르면 그래프 밖에서는 예외가 난다 — 이 테스트가 통과한다는
    사실 자체가 "안 멈췄다"의 증거다.
    """
    state = {"source_docs": [make_doc("멀쩡한 본문입니다." * 30, Confidence.HIGH)]}

    assert ocr_review(state) == {}


def test_no_resume_means_nothing_to_review():
    """이력서 없이 설문만 도는 경로 — 물어볼 것이 없다."""
    assert ocr_review({}) == {}
    assert ocr_review({"source_docs": []}) == {}


def test_an_empty_text_needs_review_even_at_high_confidence():
    """판별자는 신뢰도만 보지 않는다 — 건진 것이 없으면 고칠 거리를 줘야 한다."""
    assert docs_needing_review([make_doc("", Confidence.HIGH)])


def test_only_the_unreadable_documents_are_asked_about():
    good = make_doc("멀쩡한 본문입니다." * 30, Confidence.HIGH, doc_id="ok-1")
    bad = make_doc(GARBLED, Confidence.LOW, doc_id="bad-1")

    assert [d.doc_id for d in docs_needing_review([good, bad])] == ["bad-1"]


def test_the_node_only_asks_about_what_it_could_not_read(monkeypatch):
    """멀쩡한 문서까지 보정 화면에 올리면 안 된다.

    `docs_needing_review`를 따로 재는 것만으로는 부족하다 — 노드가 그 결과를
    쓰지 않고 문서 전부를 실어 보내도 지금 그래프(문서 1건)에서는 아무 테스트도
    소리를 내지 않는다(뮤테이션 M10). 그래서 **노드가 실제로 실어 보낸 것**을 본다.
    """
    sent: list = []

    def fake_interrupt(payload):
        sent.append(payload)
        return {}

    monkeypatch.setattr("nodes.ocr_review.interrupt", fake_interrupt)

    good = make_doc("멀쩡한 본문입니다." * 30, Confidence.HIGH, doc_id="ok-1")
    bad = make_doc(GARBLED, Confidence.LOW, doc_id="bad-1")

    ocr_review({"source_docs": [good, bad]})

    asked = [q["question_id"] for q in sent[0]["questions"]]
    assert asked == ["bad-1"], "잘 읽힌 문서까지 고치라고 띄웠다"


def test_payload_carries_the_recognized_text_so_the_form_can_show_it():
    """**카드의 목적이 여기 걸린다** — 빈 칸이 뜨면 "확인하고 고친다"가 성립하지 않는다."""
    prompt = normalize_prompt(build_payload([make_doc(GARBLED, Confidence.LOW)]))

    assert prompt.kind == KIND
    assert prompt.title == KIND_TITLES[KIND], "제목이 기본값으로 떨어졌다"

    question = prompt.questions[0]
    assert question.key == RESUME_DOC_ID, "답의 키는 문서 id다 — 위치로 잇지 않는다"
    assert question.value == GARBLED, "인식 결과가 위젯에 안 실렸다"
    assert question.options is None, "보정은 고르는 화면이 아니라 고치는 화면이다"


def test_an_empty_recognition_still_gets_a_writable_box():
    """한 글자도 못 건져도 사용자가 직접 채울 칸은 떠야 한다 (불변식 ②)."""
    prompt = normalize_prompt(build_payload([make_doc("", Confidence.LOW)]))

    assert len(prompt.questions) == 1
    assert prompt.questions[0].options is None


# --- ② 보정 적용 ----------------------------------------------------------------


def test_correction_replaces_the_text_and_lifts_the_confidence():
    docs = [make_doc(GARBLED, Confidence.LOW)]

    fixed = apply_corrections(docs, {RESUME_DOC_ID: CORRECTED})

    assert fixed[0].raw_text == CORRECTED
    assert fixed[0].confidence is CORRECTED_CONFIDENCE


def test_a_corrected_document_no_longer_asks_to_be_corrected():
    """고쳤는데도 판별자가 계속 참이면 하류(T25 게이트)가 "못 읽은 문서"로 센다."""
    fixed = apply_corrections([make_doc(GARBLED, Confidence.LOW)], {RESUME_DOC_ID: CORRECTED})

    assert not needs_manual_correction(fixed[0].raw_text, fixed[0].confidence)


def test_a_blank_answer_keeps_what_was_salvaged():
    """**불변식 ②** — 못 고쳤다고 건진 텍스트까지 버리지 않는다."""
    docs = [make_doc(GARBLED, Confidence.LOW)]

    assert apply_corrections(docs, {}) == docs
    assert apply_corrections(docs, {RESUME_DOC_ID: "   "}) == docs


def test_untouched_documents_survive_the_update():
    """목록을 통째로 돌려주는 이유 — `source_docs`에는 reducer가 없다.

    고친 문서만 반환하면 나머지가 조용히 사라진다. 여기서 안 걸면 문서가 둘 이상인
    경로가 생겼을 때 알아채지 못한다.
    """
    good = make_doc("멀쩡한 본문입니다." * 30, Confidence.HIGH, doc_id="ok-1")
    bad = make_doc(GARBLED, Confidence.LOW, doc_id="bad-1")

    fixed = apply_corrections([good, bad], {"bad-1": CORRECTED})

    assert [d.doc_id for d in fixed] == ["ok-1", "bad-1"]
    assert fixed[0] == good, "손대지 않은 문서가 바뀌었다"
    assert fixed[1].raw_text == CORRECTED


# --- ②-b 재개 값의 모양 (`app/hitl.py`에서 고친 자리) ---------------------------


class FakeStreamlit:
    """`render_prompt_form`이 부르는 것만 흉내 낸다 (T23 `FakeStreamlit`과 같은 장치).

    답은 위젯 key로 미리 정해 둔다 — 화면 없이 **폼이 무엇을 돌려주는지**만 본다.
    """

    def __init__(self, typed: dict[str, str]):
        self.typed = typed

    @contextlib.contextmanager
    def form(self, key):
        yield

    def subheader(self, *a, **kw):
        pass

    caption = markdown = subheader

    def text_area(self, label, value=None, key=None, **kw):
        return self.typed.get(key, value or "")

    def form_submit_button(self, label, **kw):
        return True


def form_result(payload, typed: dict[str, str], monkeypatch) -> dict[str, str]:
    from app import hitl

    prompt = normalize_prompt(payload)
    monkeypatch.setattr(
        hitl, "st", FakeStreamlit({f"h4__{key}": text for key, text in typed.items()})
    )
    return hitl.render_prompt_form(prompt, form_key="h4")


def test_the_form_sends_only_the_boxes_that_were_touched(monkeypatch):
    """공백 답변은 **키째로 빠진다** (D28) — "무엇에 답했나"가 재개 값만 봐도 드러난다."""
    payload = build_payload(
        [make_doc(GARBLED, Confidence.LOW, doc_id="bad-1"), make_doc("", Confidence.LOW, doc_id="bad-2")]
    )

    answers = form_result(payload, {"bad-1": CORRECTED, "bad-2": "   "}, monkeypatch)

    assert answers == {"bad-1": CORRECTED}


def test_an_all_blank_form_still_sends_something_to_resume_with(monkeypatch):
    """전부 공백이어도 **빈 dict는 안 나간다** — 나가면 재개가 통째로 건너뛰어진다.

    받는 쪽 동작은 `{}`와 같다(노드가 공백을 버린다). 달라지는 것은 하나,
    재개가 실제로 일어난다는 것이다.
    """
    payload = build_payload([make_doc("", Confidence.LOW, doc_id="bad-1")])

    answers = form_result(payload, {"bad-1": "  "}, monkeypatch)

    assert answers == {"bad-1": ""}
    assert not apply_corrections([make_doc("", Confidence.LOW, doc_id="bad-1")], answers)[0].raw_text


# --- ③ 그래프 -------------------------------------------------------------------


def test_a_scanned_resume_pauses_and_the_fix_reaches_extract(monkeypatch, profile_graph):
    """**카드의 완료 조건.** 못 읽음 → 채워진 채로 표시 → 고침 → `extract`가 그것을 받는다."""
    calls, seen_docs = install_stubs(monkeypatch, parsed=(GARBLED, Confidence.LOW))

    status = resume_or_start(
        profile_graph, "t-h4", initial_input=initial_profile_state("/tmp/scan.pdf")
    )

    assert status.phase is ThreadPhase.INTERRUPTED
    prompt = normalize_prompt(status.questions[0].payload)
    assert prompt.kind == KIND, "H2가 아니라 H4에서 먼저 멈춰야 한다"
    assert prompt.questions[0].value == GARBLED, "읽어 온 내용이 화면에 안 실렸다"
    assert not seen_docs, "고치기도 전에 `extract`가 돌았다"

    status = resume_or_start(profile_graph, "t-h4", resume={RESUME_DOC_ID: CORRECTED})

    # 다음 중단은 H2다 — 고친 뒤에는 흐름이 그대로 이어진다.
    assert status.phase is ThreadPhase.INTERRUPTED
    assert normalize_prompt(status.questions[0].payload).kind == SURVEY_KIND

    # **여기가 완료 조건이다.** `extract`는 고친 텍스트를 받았다.
    assert len(seen_docs) == 1
    assert [d.raw_text for d in seen_docs[0]] == [CORRECTED]
    assert seen_docs[0][0].confidence is CORRECTED_CONFIDENCE

    # **개수 회귀** — 텍스트가 흘렀는지는 산출로도 걸어야 한다. 깨진 원문에는
    # 역량 이름이 한 건도 없으므로, 안 흘렀으면 이 수가 0이 된다(D65 M21).
    assert len(status.values["required"]) == len(COVERED_NAMES)
    assert calls["parse_resume"] == 1, "보정 재개가 이력서를 다시 읽었다(OCR은 돈이 든다)"


def test_a_readable_resume_goes_straight_to_the_survey(monkeypatch, profile_graph):
    """잘 읽힌 이력서에는 H4가 안 뜬다 — 안 그러면 모든 사용자가 한 화면을 더 본다."""
    _, seen_docs = install_stubs(monkeypatch, parsed=(CORRECTED * 20, Confidence.HIGH))

    status = resume_or_start(
        profile_graph, "t-clean", initial_input=initial_profile_state("/tmp/resume.pdf")
    )

    assert normalize_prompt(status.questions[0].payload).kind == SURVEY_KIND
    assert seen_docs, "H4가 안 떴으면 `extract`는 이미 돌았어야 한다"


def test_an_unfixed_resume_still_produces_a_profile(monkeypatch, profile_graph):
    """**불변식 ②** — OCR 실패가 전체 실패가 되면 안 된다. 그냥 제출해도 끝까지 간다."""
    install_stubs(monkeypatch, parsed=("", Confidence.LOW))

    status = resume_or_start(
        profile_graph, "t-skip", initial_input=initial_profile_state("/tmp/scan.pdf")
    )
    assert normalize_prompt(status.questions[0].payload).kind == KIND

    # 빈 칸 그대로 제출 — 폼이 내보내는 모양 그대로다(키는 있고 값이 빈 문자열).
    status = resume_or_start(profile_graph, "t-skip", resume={RESUME_DOC_ID: ""})
    assert normalize_prompt(status.questions[0].payload).kind == SURVEY_KIND

    status = resume_or_start(profile_graph, "t-skip", resume=survey_answers(status))

    assert status.phase is ThreadPhase.COMPLETE
    profile = status.values["profile"]
    assert isinstance(profile, ProfileJSON)
    assert profile.level_coordinates, "설문만으로도 프로필은 나와야 한다"


def test_the_correction_survives_all_the_way_into_the_profile(monkeypatch, profile_graph):
    """고친 텍스트에서 나온 역량이 **최종 산출까지** 남는가.

    `extract`가 받았는지만 보면 중간에서 덮여도 초록불이다 — 프로필의 역량 이름을
    직접 건다.
    """
    install_stubs(monkeypatch, parsed=(GARBLED, Confidence.LOW))

    status = resume_or_start(
        profile_graph, "t-end", initial_input=initial_profile_state("/tmp/scan.pdf")
    )
    status = resume_or_start(profile_graph, "t-end", resume={RESUME_DOC_ID: CORRECTED})
    status = resume_or_start(profile_graph, "t-end", resume=survey_answers(status))

    assert status.phase is ThreadPhase.COMPLETE
    names = {c.name for c in status.values["profile"].competencies}
    assert set(COVERED_NAMES) <= names, "고친 이력서의 역량이 프로필에 안 남았다"


def test_an_empty_answer_map_would_silently_do_nothing(monkeypatch, profile_graph):
    """**왜 폼이 빈 dict를 안 내보내는가** (`app/hitl.py` 반환문의 근거).

    langgraph 1.2.10은 `Command(resume={})`를 "중단 id별 답변 맵인데 항목이 없는 것"
    으로 읽어 재개를 아예 하지 않는다 — 화면에서는 제출을 눌러도 같은 폼이 다시
    뜨고 이유가 안 보인다. 이 테스트는 그 동작을 **못 박아 둔다**: 여기가 깨지면
    (버전이 올라 `{}`도 재개가 되면) `app/hitl.py`의 우회를 걷어내도 된다.
    """
    install_stubs(monkeypatch, parsed=("", Confidence.LOW))

    resume_or_start(
        profile_graph, "t-empty", initial_input=initial_profile_state("/tmp/scan.pdf")
    )
    status = resume_or_start(profile_graph, "t-empty", resume={})

    assert normalize_prompt(status.questions[0].payload).kind == KIND, (
        "빈 dict로도 재개가 됐다 — `render_prompt_form`의 우회를 걷어낼 수 있다"
    )


# --- ④ 화면 ---------------------------------------------------------------------


def run_app(monkeypatch):
    import graphs.session as session

    monkeypatch.setattr(session, "build_checkpointer", lambda *a, **kw: InMemorySaver())
    return AppTest.from_file(APP, default_timeout=60).run()


def switch_to_profile_mode(at):
    """모드 라디오는 **키로 찾는다** — 위치로 잡으면 화면 구성에 따라 다른 것을 누른다(D73)."""
    next(w for w in at.radio if w.key == MODE_KEY).set_value(MODE_LABELS[MODE_PROFILE])
    return at.run()


def start_with_a_scanned_resume(monkeypatch, at):
    at.file_uploader[0].set_value(("scan.pdf", b"%PDF-1.4 fake", "application/pdf"))
    at = at.run()
    return at.button[0].click().run()


def test_app_shows_the_recognized_text_prefilled_and_carries_the_fix(monkeypatch):
    """**화면 관통** — 업로드 → 채워진 칸 → 고쳐 제출 → 설문으로 이어진다."""
    _, seen_docs = install_stubs(monkeypatch, parsed=(GARBLED, Confidence.LOW))

    at = start_with_a_scanned_resume(monkeypatch, switch_to_profile_mode(run_app(monkeypatch)))
    assert not at.exception

    # ① H4 — 인식 결과가 **채워진 채로** 떴다.
    assert at.subheader[0].value == KIND_TITLES[KIND]
    assert len(at.text_area) == 1, "보정 칸은 문서 수만큼만 떠야 한다"
    assert at.text_area[0].value == GARBLED, "빈 칸이 떴다 — 고칠 거리가 없다"
    assert at.button[0].label == SUBMIT_LABELS[KIND]

    notices = " ".join(info.value for info in at.info)
    assert "읽지 못했" in notices
    assert "판정 근거가 부족" not in notices, "델타 인터뷰 문구가 H4 화면에 떴다(D63)"

    # ② 고쳐서 제출 → H2로 넘어간다.
    at.text_area[0].set_value(CORRECTED)
    at = at.button[0].click().run()
    assert not at.exception

    assert at.subheader[0].value == KIND_TITLES[SURVEY_KIND], "설문으로 안 넘어갔다"
    assert seen_docs and seen_docs[0][0].raw_text == CORRECTED, "화면에서 고친 것이 상태에 안 남았다"


def test_submitting_without_fixing_anything_moves_on(monkeypatch):
    """**불변식 ②를 화면에서.** 한 글자도 못 건진 채 그냥 제출해도 흐름이 이어진다.

    이 자리가 조용히 막혔던 곳이다 — 폼이 빈 dict를 돌려주면 langgraph가 재개를
    건너뛰어 **같은 화면이 계속 뜬다**. 그래프 테스트로는 안 잡힌다(거기서는 재개
    값을 손으로 만든다). 화면을 실제로 눌러 봐야 나오는 결함이다.
    """
    install_stubs(monkeypatch, parsed=("", Confidence.LOW))

    at = start_with_a_scanned_resume(monkeypatch, switch_to_profile_mode(run_app(monkeypatch)))
    assert at.subheader[0].value == KIND_TITLES[KIND]

    at = at.button[0].click().run()  # 아무것도 안 채우고 제출

    assert not at.exception
    assert at.subheader[0].value == KIND_TITLES[SURVEY_KIND], "제출했는데 같은 화면이 다시 떴다"


def test_the_paused_panel_names_the_review_step(monkeypatch):
    """진행 표시가 H4를 **사람 말로** 가리키는가 — 라벨 시트가 비면 fallback으로 떨어진다."""
    install_stubs(monkeypatch, parsed=(GARBLED, Confidence.LOW))

    at = start_with_a_scanned_resume(monkeypatch, switch_to_profile_mode(run_app(monkeypatch)))

    sheet = load_labels(PROFILE_LABELS_PATH)
    body = "\n".join(block.value for block in at.markdown)
    assert f"⏸ {sheet.node('ocr_review').label}" in body
    assert "ocr_review" not in body, "내부 함수명이 화면에 노출됐다"


def test_a_readable_resume_shows_no_review_screen(monkeypatch):
    """화면에서도 확인한다 — 멀쩡한 이력서에 보정 칸이 뜨면 안 된다."""
    install_stubs(monkeypatch, parsed=(CORRECTED * 20, Confidence.HIGH))

    at = start_with_a_scanned_resume(monkeypatch, switch_to_profile_mode(run_app(monkeypatch)))

    assert at.subheader[0].value == KIND_TITLES[SURVEY_KIND]
    assert not at.text_area, "보정 칸이 떴다"


# --- ⑤ 배선 (§2-1) ---------------------------------------------------------------


def test_h4_sits_between_parsing_and_extraction():
    """자리가 곧 계약이다 — 뒤에 두면 고친 텍스트가 `extract`에 안 닿는다."""
    order = PROFILE_NODE_NAMES

    assert "ocr_review" in order, "H4가 배선되지 않았다"
    assert order.index("parse_resume") < order.index("ocr_review") < order.index("extract")


def test_the_wired_node_has_a_user_facing_label():
    """배선한 카드가 라벨을 갖는다(D59) — 없으면 진행 표시가 fallback으로 떨어진다."""
    assert "ocr_review" in load_labels(PROFILE_LABELS_PATH).nodes


def test_the_interrupt_has_its_own_title_and_notice():
    """중단점마다 제목·안내가 다르다 (HITL 규약 ④)."""
    assert KIND in KIND_TITLES
    assert KIND in RESUME_NOTICES
    assert RESUME_NOTICES[KIND] != RESUME_NOTICES["delta_interview"]
    assert RESUME_NOTICES[KIND] != RESUME_NOTICES["level_survey"]

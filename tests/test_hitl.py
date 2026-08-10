"""T12 · Streamlit 재개 루프.

완료 조건은 하나다 — **폼 제출 시 그래프가 처음부터 다시 돌지 않는다.**
그래서 이 파일의 중심은 `test_resume_does_not_rerun_upstream_nodes`이며, 증명 방식은
카드가 요구한 대로 **재개 전후 LLM 호출 횟수 비교**다. LLM을 타는 도구 4종을
호출 횟수를 세는 스텁으로 갈아끼우고, 반환값은 전부 `fixtures/`의 골든 데이터를
쓴다 (R5 — 자기 구현을 흉내낸 mock으로 통과시키지 않는다).

층은 셋이다:
    ① 그래프 — 중단·재개가 상류 노드를 다시 돌리지 않는가
    ② `app/hitl.py` — 어떤 모양의 페이로드든 폼으로 펼쳐지는가 (순수 함수)
    ③ `app/main.py` — 실제 Streamlit 앱을 `AppTest`로 구동해 관통하는가
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from app.hitl import SKIP_LABEL, FormQuestion, normalize_prompt
from contracts.enums import VerdictState
from contracts.models import (
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    ProfileJSON,
    Question,
    SourceDocument,
    StrategyBrief,
)
from graphs.analysis_graph import INTERVIEW_NODE, NODE_NAMES, build_analysis_graph, node_sequence
from graphs.session import ThreadPhase, inspect_thread, resume_or_start
from nodes import analysis_nodes, interview
from nodes.interview import MAX_ROUNDS, ROUND_LIMIT_RATIONALE
from tools.verify import question_id_for

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
VERDICTS_BY_ID = {
    v["criterion_id"]: CriterionVerdict.model_validate(v)
    for v in json.loads((FIXTURES / "verdicts_all_met.json").read_text("utf-8"))
}

SCORED_COMP_IDS = sorted(set(CRITERIA_ALL) & {c.comp_id for c in BACKEND_REQUIRED})

# 첫 역량의 기준들만 "근거를 못 찾아 질문으로 승격된" 것으로 둔다. 전부 미결로
# 만들면 집계할 게 없어져 브리프가 비고, 하나도 없으면 그래프가 멈추지 않는다.
OPEN_CRITERION_IDS = {c.criterion_id for c in CRITERIA_ALL[SCORED_COMP_IDS[0]]}

TARGET_DATE = date.today() + timedelta(days=90)


def initial_state() -> dict:
    return {
        "mode": "analysis",
        "company": JD.company,
        "role": "백엔드 엔지니어",
        "target_date": TARGET_DATE,
        "profile": PROFILE,
        "raw_jd_input": JD.raw_text,
    }


# --- 스텁 ---------------------------------------------------------------------


def install_stubs(monkeypatch, *, answers_resolve: bool = True) -> Counter:
    """LLM 도구 4종을 호출 횟수를 세는 픽스처 스텁으로 대체한다.

    `answers_resolve=False`면 무슨 답을 줘도 판정이 안 서는 상황을 흉내낸다 —
    라운드 상한이 실제로 루프를 끊는지 보려면 이 경우가 필요하다.

    `verify_criteria`는 **두 모듈이 각자 import 해 간다**(`analysis_nodes`의 verify
    노드, `interview`의 재판정). 둘 다 갈아끼워야 카운터가 전체 호출을 센다.
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
        decided: list[CriterionVerdict] = []
        questions: list[Question] = []

        for criterion in criteria:
            qid = question_id_for(criterion.criterion_id)
            answered = qid in answers
            open_now = criterion.criterion_id in OPEN_CRITERION_IDS
            if not open_now or (answered and answers_resolve):
                decided.append(VERDICTS_BY_ID[criterion.criterion_id])
            else:
                questions.append(
                    Question(
                        question_id=qid,
                        criterion_id=criterion.criterion_id,
                        text=f"{criterion.text} — 해당 경험이 있나요?",
                    )
                )
        return decided, questions

    def fake_fill(brief):
        calls["fill_brief_slots"] += 1
        return brief.model_copy(update={"summary_line": "슬롯이 채워졌다"})

    def fake_retrieve(required, owned, top_k=3):
        # 임베딩도 네트워크를 탄다(T14b). 갈아끼우지 않으면 오프라인 실행이 실제
        # API를 왕복해 §4의 "pytest -q는 LLM 없이 통과한다"가 깨진다.
        calls["retrieve_candidates"] += 1
        return [(r.comp_id, owned[0].comp_id) for r in required] if owned else []

    monkeypatch.setattr(analysis_nodes, "extract_competencies", fake_extract)
    monkeypatch.setattr(analysis_nodes, "decompose_criteria", fake_decompose)
    monkeypatch.setattr(analysis_nodes, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(analysis_nodes, "verify_criteria", fake_verify)
    monkeypatch.setattr(analysis_nodes, "fill_brief_slots", fake_fill)
    monkeypatch.setattr(interview, "verify_criteria", fake_verify)
    return calls


@pytest.fixture
def hitl_graph():
    return build_analysis_graph(InMemorySaver(), interactive=True)


def answers_for(status) -> dict[str, str]:
    """중단 페이로드에 실린 질문 전부에 답한다."""
    prompt = normalize_prompt(status.questions[0].payload)
    return {q.key: "직전 프로젝트에서 직접 담당했습니다." for q in prompt.questions}


# --- ① 완료 조건: 재개가 상류 노드를 다시 돌리지 않는다 -------------------------


def test_resume_does_not_rerun_upstream_nodes(monkeypatch, hitl_graph, capsys):
    """**T12 = P1 완료 조건.** 폼 제출 시 그래프가 처음부터 다시 돌지 않는다."""
    calls = install_stubs(monkeypatch)

    status = resume_or_start(hitl_graph, "t-resume", initial_input=initial_state())
    assert status.phase is ThreadPhase.INTERRUPTED, "미결 기준이 있으면 멈춰야 한다"
    before = Counter(calls)

    status = resume_or_start(hitl_graph, "t-resume", resume=answers_for(status))
    assert status.phase is ThreadPhase.COMPLETE
    after = Counter(calls)

    # 카드가 요구한 로그 — `-s`로 돌리면 그대로 보인다.
    print("\n[재개 전후 LLM 도구 호출 횟수]")
    for name in ("extract_competencies", "decompose_criteria", "verify_criteria", "fill_brief_slots"):
        print(f"  {name:<22} 재개 전 {before[name]} → 재개 후 {after[name]}")

    # 재개가 처음부터 돌았다면 이 셋이 전부 2가 된다.
    assert after["extract_competencies"] == 1
    assert after["decompose_criteria"] == 1
    # verify만 2 — verify 노드 1회 + 델타 인터뷰의 재판정 1회. 재실행이 아니라 재판정이다.
    assert after["verify_criteria"] == before["verify_criteria"] + 1
    assert after["fill_brief_slots"] == 1

    assert isinstance(status.values["brief"], StrategyBrief)


def test_answers_actually_change_the_verdicts(monkeypatch, hitl_graph):
    """재개가 싸게 끝나는 것만으로는 부족하다 — 답변이 판정에 반영돼야 의미가 있다."""
    install_stubs(monkeypatch)

    status = resume_or_start(hitl_graph, "t-verdict", initial_input=initial_state())
    open_ids = {q.criterion_id for q in status.values["pending_questions"]}
    assert open_ids == OPEN_CRITERION_IDS

    status = resume_or_start(hitl_graph, "t-verdict", resume=answers_for(status))

    decided = {v.criterion_id for v in status.values["verdicts"]}
    assert OPEN_CRITERION_IDS <= decided, "답한 기준이 판정 목록에 들어와야 한다"
    assert not status.values["pending_questions"]
    assert status.values["interview_round"] == 1


def test_download_rerun_does_not_run_the_graph_again(monkeypatch, hitl_graph):
    """완료된 스레드는 몇 번을 조회해도 다시 돌지 않는다 (D27 — 다운로드 버튼 rerun)."""
    calls = install_stubs(monkeypatch)

    status = resume_or_start(hitl_graph, "t-done", initial_input=initial_state())
    resume_or_start(hitl_graph, "t-done", resume=answers_for(status))
    settled = Counter(calls)

    for _ in range(3):
        status = resume_or_start(hitl_graph, "t-done", initial_input=initial_state())
        assert status.phase is ThreadPhase.COMPLETE

    assert Counter(calls) == settled


def test_pending_form_without_submit_leaves_thread_interrupted(monkeypatch, hitl_graph):
    """폼을 그리기만 하고 제출하지 않으면 그래프는 중단 지점에 그대로 서 있다."""
    calls = install_stubs(monkeypatch)

    resume_or_start(hitl_graph, "t-idle", initial_input=initial_state())
    frozen = Counter(calls)

    for _ in range(3):  # rerun 3번 = 조회 3번
        status = inspect_thread(hitl_graph, "t-idle")
        assert status.is_interrupted

    assert Counter(calls) == frozen


# --- ② 라운드 상한 -------------------------------------------------------------


def test_round_limit_ends_the_loop(monkeypatch, hitl_graph):
    """답이 판정을 못 세워도 루프가 끝난다 — 3라운드는 없다 (T11 계약)."""
    install_stubs(monkeypatch, answers_resolve=False)

    status = resume_or_start(hitl_graph, "t-limit", initial_input=initial_state())

    rounds = 0
    while status.is_interrupted:
        rounds += 1
        assert rounds <= MAX_ROUNDS, "상한을 넘겨서 물으면 안 된다"
        status = resume_or_start(hitl_graph, "t-limit", resume=answers_for(status))

    assert status.phase is ThreadPhase.COMPLETE
    assert rounds == MAX_ROUNDS
    assert status.values["interview_round"] == MAX_ROUNDS

    # 못 푼 기준은 조용히 사라지지 않고 판단보류로 확정된다.
    unknown = [v for v in status.values["verdicts"] if v.state is VerdictState.UNKNOWN]
    assert {v.criterion_id for v in unknown} == OPEN_CRITERION_IDS
    assert all(v.rationale == ROUND_LIMIT_RATIONALE for v in unknown)
    assert isinstance(status.values["brief"], StrategyBrief)


def test_graph_does_not_stop_when_nothing_is_open(monkeypatch, hitl_graph):
    """물을 게 없으면 중단 없이 한 번에 끝난다 (질문 없으면 안 묻는다)."""
    calls = install_stubs(monkeypatch)
    monkeypatch.setattr(
        analysis_nodes,
        "verify_criteria",
        lambda criteria, profile, answers=None: (
            [VERDICTS_BY_ID[c.criterion_id] for c in criteria],
            [],
        ),
    )

    status = resume_or_start(hitl_graph, "t-clean", initial_input=initial_state())

    assert status.phase is ThreadPhase.COMPLETE
    assert calls["verify_criteria"] == 0  # 위에서 다시 갈아끼운 스텁이 돌았다
    assert isinstance(status.values["brief"], StrategyBrief)


# --- ③ 배선 불변식 -------------------------------------------------------------


def test_interview_sits_between_verify_and_aggregate():
    names = [name for name, _ in node_sequence(interactive=True)]
    assert names.index("verify") + 1 == names.index(INTERVIEW_NODE)
    assert names.index(INTERVIEW_NODE) + 1 == names.index("aggregate")


def test_p0_sequence_is_untouched():
    """`interactive=False`는 T08이 고정한 직선 그대로다 — P0 경로가 살아 있어야 한다."""
    assert [name for name, _ in node_sequence()] == NODE_NAMES
    assert INTERVIEW_NODE not in NODE_NAMES


# --- ④ app/hitl.py — 질문 렌더링 일반화 ----------------------------------------


def test_delta_interview_payload_normalizes():
    payload = interview.build_interview_payload(
        [Question(question_id="q-c1", criterion_id="c1", text="쿠버네티스를 운영했나요?")], 1
    )
    prompt = normalize_prompt(payload)

    assert prompt.kind == "delta_interview"
    assert prompt.title == "몇 가지만 더 알려주세요"
    assert prompt.progress_caption == "1 / 2 라운드 · 질문 1건"
    assert prompt.questions == [FormQuestion(key="q-c1", text="쿠버네티스를 운영했나요?")]


def test_choice_questions_keep_their_options():
    """H1(공고 선택)·H4(OCR 보정)가 쓸 선택지형 질문 — 유형별 코드를 새로 만들지 않는다."""
    prompt = normalize_prompt(
        {
            "kind": "job_selection",
            "questions": [{"question_id": "j1", "text": "어느 공고?", "options": ["A", "B"]}],
        }
    )
    assert prompt.title == "어느 공고로 분석할까요"
    assert prompt.questions[0].options == ["A", "B"]


@pytest.mark.parametrize(
    "payload, expected_keys, expected_texts",
    [
        ({"text": "한 줄만 묻는 경우"}, ["q0"], ["한 줄만 묻는 경우"]),
        ("문자열 하나", ["q0"], ["문자열 하나"]),
        ([{"criterion_id": "c9", "text": "목록만 온 경우"}], ["c9"], ["목록만 온 경우"]),
        (
            [Question(question_id="q-c9", criterion_id="c9", text="모델이 온 경우")],
            ["q-c9"],
            ["모델이 온 경우"],
        ),
    ],
)
def test_unknown_payload_shapes_still_render(payload, expected_keys, expected_texts):
    """모르는 모양이 와도 폼은 그려진다 — 여기서 예외가 나면 재개 루프가 막힌다."""
    prompt = normalize_prompt(payload)
    assert [q.key for q in prompt.questions] == expected_keys
    assert [q.text for q in prompt.questions] == expected_texts
    assert prompt.title == "추가 입력이 필요합니다"  # 모르는 kind는 기본 제목


# --- ⑤ 실제 앱 관통 (AppTest) --------------------------------------------------


def run_app(monkeypatch):
    """실제 `app/main.py`를 구동한다. 체크포인터만 인메모리로 갈아끼운다.

    `main.py`가 `from graphs.session import build_checkpointer`를 **스크립트 실행
    시점에** 하므로 원본 모듈 속성을 갈아끼우면 그대로 걸린다.
    """
    import graphs.session as session

    monkeypatch.setattr(session, "build_checkpointer", lambda *a, **kw: InMemorySaver())
    at = AppTest.from_file(APP, default_timeout=30)
    return at.run()


def fill_input_form(at):
    at.text_input[0].set_value("테크노베이션")
    at.text_input[1].set_value("백엔드 엔지니어")
    at.text_area[0].set_value(JD.raw_text)
    return at.button[0].click().run()


def test_app_pauses_then_resumes_without_rerunning(monkeypatch):
    """카드의 수동 검증을 자동화한 것 — 폼 → 중단 → 답변 → 결과 관통."""
    calls = install_stubs(monkeypatch)
    at = run_app(monkeypatch)

    assert not at.exception
    at = fill_input_form(at)
    assert not at.exception

    # 중단 국면 — 질문 폼이 떠 있고 결과·다운로드는 아직 없다.
    assert at.subheader[0].value == "몇 가지만 더 알려주세요"
    assert len(at.text_area) == len(OPEN_CRITERION_IDS)
    assert not at.download_button
    before = Counter(calls)

    for widget in at.text_area:
        widget.set_value("사내 서비스에서 직접 담당했습니다.")
    at = at.button[0].click().run()
    assert not at.exception

    # 완료 국면 — 결과와 다운로드 버튼이 떴고, 상류 노드는 다시 돌지 않았다.
    assert at.download_button, "브리프 .md 다운로드 버튼이 떠야 한다"
    assert at.download_button[0].label == "전략 브리프 .md 다운로드"
    assert calls["extract_competencies"] == before["extract_competencies"] == 1
    assert calls["decompose_criteria"] == before["decompose_criteria"] == 1

    # `AppTest`의 download_button.value는 클릭 여부(bool)라 파일 바이트를 못 본다.
    # `.md` 본문은 test_markdown.py가 이미 고정하므로, 여기서는 결과 화면이 실제로
    # 그려졌는지(= 브리프가 상태에 있는지)를 요약 지표로 확인한다.
    assert at.metric, "요약 지표가 떠야 결과가 그려진 것이다"
    assert {m.label for m in at.metric} == {"남은 기간", "소스 충족률", "신뢰등급", "카드"}
    assert at.metric[0].value == "90일"


def test_app_new_analysis_starts_a_fresh_thread(monkeypatch):
    """완료된 스레드를 재사용하지 않는다 — [새 분석 시작]은 새 thread_id를 뽑는다."""
    install_stubs(monkeypatch)
    at = fill_input_form(run_app(monkeypatch))

    for widget in at.text_area:
        widget.set_value("담당했습니다.")
    at = at.button[0].click().run()
    assert at.download_button

    at = at.button[-1].click().run()  # [새 분석 시작]

    assert not at.exception
    assert not at.download_button
    assert at.text_input, "입력 폼으로 돌아와야 한다"


def test_skip_choice_is_not_sent_as_an_answer():
    """선택지형 질문에서 '답하지 않음'은 답변으로 넘기지 않는다 (D28 부수결정 2)."""
    prompt = normalize_prompt(
        {"questions": [{"question_id": "q1", "text": "?", "options": ["예", "아니오"]}]}
    )
    assert SKIP_LABEL not in (prompt.questions[0].options or [])

"""취업준비 Helper Agent — JD 붙여넣기 → (필요하면 되묻고) → 3트랙 브리프 `.md`.

**T12에서 P0의 `invoke` 1회가 HITL 재개 루프로 바뀌었다.** 화면은 이제 스레드의
국면 하나를 그리는 상태기계다:

```
미시작  → 입력 폼          → resume_or_start(initial_input=…)
중단됨  → 질문 폼(app/hitl) → resume_or_start(resume=답변)
완료    → 요약 + .md 다운로드   (실행하지 않는다)
```

**이 파일에서 `graph.invoke()`를 직접 부르지 않는다** (`graphs/session.py`의 유일한
규칙, D27). Streamlit은 위젯을 건드릴 때마다 스크립트를 통째로 재실행하므로,
`invoke`를 직접 부르면 사용자가 답변할 때마다·다운로드 버튼을 누를 때마다 분석이
처음부터 다시 돈다. `resume_or_start()`는 실행 전에 반드시 상태를 조회한다.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

# `streamlit run app/main.py`는 저장소 루트를 sys.path에 넣지 않는다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.hitl import normalize_prompt, render_pending  # noqa: E402
from app.progress import (  # noqa: E402
    ProgressTracker,
    close_status,
    live_status,
    load_labels,
    panel_slot,
    render_panel,
)
from contracts.models import ProfileJSON  # noqa: E402
from contracts.state import GraphState  # noqa: E402
from graphs.analysis_graph import build_analysis_graph, node_sequence  # noqa: E402
from graphs.profile_graph import (  # noqa: E402
    build_profile_graph,
    download_filename,
    initial_profile_state,
    load_saved_profile,
    profile_bytes,
    profile_node_sequence,
)
from graphs.session import (  # noqa: E402
    THREAD_KEY,
    RunStatus,
    ThreadPhase,
    build_checkpointer,
    get_or_create_thread,
    inspect_thread,
    resume_or_start,
)
from llm.client import LLMError  # noqa: E402
from llm.vision import extract_text_from_image  # noqa: E402
from render.cards import render_brief  # noqa: E402
from render.markdown import filename_for, render_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PROFILE = ROOT / "fixtures" / "profile_sample.json"
PROFILE_LABELS = ROOT / "presets" / "profile_labels.yaml"

GRAPH_KEY = "analysis_graph"
TRACKER_KEY = "progress_tracker"
ERROR_KEY = "run_error"

# --- 두 모드 (UC-1 / UC-2) -----------------------------------------------------
#
# 화면 하나에 그래프가 둘이다. **서로의 상태를 밟지 않는 것이 유일한 규칙**이며,
# 그것을 이름 규칙이 아니라 **네임스페이스**로 만든다 — UC-1의 그래프·스레드·
# 트래커·오류는 전부 `st.session_state[PROFILE_NS]` 안쪽 dict에 산다. 키를
# `uc1_graph`처럼 접두사로 가르면 새 키를 더할 때마다 접두사를 잊을 자리가 생기고,
# 그때 두 모드가 같은 스레드를 공유해 분석 결과가 프로필 화면에 뜬다.
#
# `get_or_create_thread()`가 `MutableMapping`이면 무엇이든 받게 돼 있어서(T10)
# 안쪽 dict를 그대로 넘기면 세션 계층은 고칠 게 없다.
MODE_ANALYSIS = "analysis"
MODE_PROFILE = "profile"
MODE_KEY = "app_mode"
MODE_LABELS = {
    MODE_ANALYSIS: "🎯 회사 분석 (UC-2)",
    MODE_PROFILE: "👤 내 프로필 만들기 (UC-1)",
}
MODE_BY_LABEL = {label: mode for mode, label in MODE_LABELS.items()}

PROFILE_NS = "uc1_session"

# 업로드된 이력서가 놓일 자리. `parse_resume(file_path)`가 **경로**를 받으므로
# 어딘가에 떨어뜨려야 한다. 체크포인트와 같은 폴더 아래라 `.gitignore`가 덮는다.
UPLOAD_DIR = Path(".jobprep") / "uploads"
RESUME_PATH_KEY = "resume_path"
RESUME_NAME_KEY = "resume_name"

# JD 본문 `text_area`의 위젯 키. 이미지 추출 결과를 **위젯 생성 전에** 이 키로 넣어
# 두면 폼이 그 값으로 그려진다 — 사용자가 그대로 고칠 수 있는 상태다.
JD_TEXT_KEY = "jd_body"
IMAGE_DIGEST_KEY = "jd_image_digest"
IMAGE_ERROR_KEY = "jd_image_error"


# --- 그래프 (세션당 1개) -------------------------------------------------------


def get_graph():
    """컴파일된 대화형 그래프를 세션에 한 번만 만들어 재사용한다.

    `st.cache_resource`가 아니라 `st.session_state`에 두는 이유는 두 가지다.
    ① 체크포인터가 여는 sqlite 커넥션의 수명이 세션과 같아진다. ② 테스트가
    세션마다 깨끗한 상태에서 시작한다 — 프로세스 전역 캐시면 앞 테스트의
    스레드가 다음 테스트로 샌다.

    rerun마다 다시 만들면 sqlite 커넥션이 계속 쌓이므로 반드시 재사용해야 한다.
    """
    if GRAPH_KEY not in st.session_state:
        st.session_state[GRAPH_KEY] = build_analysis_graph(
            build_checkpointer(), interactive=True
        )
    return st.session_state[GRAPH_KEY]


def get_tracker() -> ProgressTracker:
    """진행 표시도 그래프와 같은 수명을 갖는다 — 세션에 하나.

    줄 목록은 `NODE_SEQUENCE`가 아니라 **`node_sequence(interactive=True)`**에서
    받는다. 전자에는 델타 인터뷰가 없어 화면과 실제 배선이 어긋난다(D29).

    실행마다 새로 만들면 안 된다. 재개는 이미 끝난 노드를 다시 흘리지 않으므로
    (D27 — 상류 노드는 재실행되지 않는다) 앞 라운드에서 ✅였던 줄이 ⬜로 되돌아간다.
    """
    if TRACKER_KEY not in st.session_state:
        names = [name for name, _ in node_sequence(interactive=True)]
        st.session_state[TRACKER_KEY] = ProgressTracker(names)
    return st.session_state[TRACKER_KEY]


def load_profile(uploaded, use_sample: bool, use_saved: bool = False) -> ProfileJSON | None:
    """분석에 쓸 ProfileJSON을 고른다. 설계 §10-3 — UC-1 산출물을 업로드로 받는다.

    우선순위는 **사용자가 방금 지정한 것 → 직전 산출 → 시연용 샘플**이다.
    `use_saved`는 UC-1이 디스크에 남긴 프로필을 다운로드·업로드 없이 바로 집는
    지름길이며, 기본값이 `False`인 것이 중요하다 — 기본으로 켜면 예전에 만들어 둔
    프로필이 사용자도 모르게 분석의 근거가 된다.
    """
    if uploaded is not None:
        return ProfileJSON.model_validate_json(uploaded.getvalue())
    if use_saved:
        saved = load_saved_profile()
        if saved is not None:
            return saved
    if use_sample and SAMPLE_PROFILE.exists():
        return ProfileJSON.model_validate_json(SAMPLE_PROFILE.read_bytes())
    return None


def initial_state(
    *,
    company: str,
    role: str,
    target_date: date,
    raw_jd: str,
    profile: ProfileJSON | None,
) -> GraphState:
    return {
        "mode": "analysis",
        "company": company,
        "role": role,
        "target_date": target_date,
        "profile": profile,
        "raw_jd_input": raw_jd,
    }


def advance(
    graph, thread_id: str, *, slot=None, tracker=None, store=None, **kwargs
) -> RunStatus | None:
    """`resume_or_start()`를 감싸 **진행을 보여주고** 실패를 화면에 남긴다.

    실행하는 동안 노드 이벤트가 `st.status` 패널로 흐른다(T13). 콜백은
    `resume_or_start(on_event=)`로 넘긴다 — 여기서 `graph.stream()`을 직접 부르면
    국면 조회를 건너뛰는 경로가 생겨 완료된 스레드가 다시 도는 결함이 되살아난다(D27).

    LLM 호출이 실패하면(크레딧 소진·타임아웃) 그래프는 중간에서 멈추고, 그 스레드는
    `interrupts`가 비어 있어 국면 판별상 "완료"로 보인다. 예외를 삼키면 사용자는
    까닭 모를 빈 결과를 보게 되므로 사유를 세션에 남겨 그대로 띄운다.

    `tracker`·`store`는 UC-1이 같은 실행 경로를 쓰기 위한 자리다(T24). 안 주면
    UC-2의 세션 전역을 그대로 쓴다 — **실행 경로를 두 벌로 만들지 않는 것**이
    핵심이고(D27의 규칙은 그래프가 몇 개든 같다), 그래서 여기서 갈리는 것은
    "어느 트래커·어느 저장소인가"뿐이다.
    """
    store = st.session_state if store is None else store
    store.pop(ERROR_KEY, None)
    tracker = get_tracker() if tracker is None else tracker
    tracker.begin()  # 중단 중 흘러간 사용자 대기 시간을 노드 시간에 섞지 않는다
    box, on_event = live_status(tracker, slot=slot)

    try:
        return resume_or_start(graph, thread_id, on_event=on_event, **kwargs)
    except Exception as exc:  # noqa: BLE001 — 무엇이 터지든 화면에는 나와야 한다
        reason = f"{type(exc).__name__}: {exc}"
        store[ERROR_KEY] = reason
        tracker.fail(reason)
        return None
    finally:
        close_status(box, tracker)


def reset_thread() -> None:
    """새 분석 = 새 스레드. 완료된 스레드는 재실행하지 않는 게 원칙이다(D27).

    JD 본문과 이미지 표식도 함께 지운다 — 안 지우면 새 분석 폼에 앞 공고의 본문이
    남아 있고, 같은 스크린샷을 다시 올려도 "이미 시도함"으로 걸려 추출이 안 돈다.
    """
    for key in (
        THREAD_KEY,
        ERROR_KEY,
        TRACKER_KEY,
        JD_TEXT_KEY,
        IMAGE_DIGEST_KEY,
        IMAGE_ERROR_KEY,
    ):
        st.session_state.pop(key, None)


# --- 화면: 미시작 -------------------------------------------------------------


def ingest_jd_image(uploaded, state) -> None:
    """업로드된 이미지를 텍스트로 바꿔 `state`에 넣는다.

    **Streamlit 위젯을 부르지 않는 순수 계층이다** — spinner·에러 표시는 호출부가
    맡는다. 그래야 "같은 이미지를 두 번 추출하지 않는다" 같은 규칙을 위젯 없이
    직접 검증할 수 있다.

    성공하면 `JD_TEXT_KEY`에 본문이, 실패하면 `IMAGE_ERROR_KEY`에 사유가 남는다.
    둘 중 무엇이든 `IMAGE_DIGEST_KEY`(시도 표식)는 **호출 전에** 먼저 찍는다.
    Streamlit은 위젯을 건드릴 때마다 스크립트를 통째로 다시 돌리므로, 표식이 없으면
    업로드된 이미지가 화면에 붙어 있는 내내 같은 API를 왕복한다 — 실패하는 경우에도
    마찬가지라 크레딧이 조용히 녹는다. 재시도는 표식을 지우는 버튼이 맡는다.
    """
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if state.get(IMAGE_DIGEST_KEY) == digest:
        return

    state[IMAGE_DIGEST_KEY] = digest
    try:
        text = extract_text_from_image(data, uploaded.type or "")
    except LLMError as exc:
        # `LLMError`만 삼킨다. 그 밖의 예외는 코드 결함이므로 전파해서 드러낸다(D20).
        state[IMAGE_ERROR_KEY] = str(exc)
        return

    state.pop(IMAGE_ERROR_KEY, None)
    state[JD_TEXT_KEY] = text


def retry_jd_image() -> None:
    """시도 표식을 지운다 — 다음 rerun에서 같은 이미지를 한 번 더 추출한다."""
    for key in (IMAGE_DIGEST_KEY, IMAGE_ERROR_KEY):
        st.session_state.pop(key, None)


def render_jd_image_input() -> None:
    """JD 스크린샷 → 본문 텍스트. **폼 밖에 있어야 한다** (DEVLOG D48).

    `st.form` 안의 위젯은 제출 전까지 rerun을 일으키지 않는다. 업로더를 폼 안에 두면
    이미지가 서버에 닿는 시점이 곧 "제출 순간"이고, 그때는 빈 `text_area`가 이미 함께
    제출된 뒤라 **미리 채워 넣을 자리가 없다.** 카드 스니펫은 폼 안이었지만 그대로
    두면 동작하지 않는다 — `AppTest`는 폼 배칭을 모사하지 않아 이 차이를 못 잡는다(D41).

    추출은 대안이지 대체가 아니다. 붙여넣기 경로는 그대로 살아 있고, 여기서 무슨 일이
    나든 사용자는 아래 칸에 직접 입력해 진행할 수 있다.
    """
    uploaded = st.file_uploader(
        "JD가 이미지로만 있나요? 스크린샷을 올리면 아래 본문 칸을 채워 드립니다.",
        type=["png", "jpg", "jpeg"],
        help="추출된 텍스트는 그대로 실행되지 않습니다 — 확인하고 고친 뒤 실행하세요.",
    )
    if uploaded is not None:
        with st.spinner("이미지에서 JD 본문을 읽는 중…"):
            ingest_jd_image(uploaded, st.session_state)

    if error := st.session_state.get(IMAGE_ERROR_KEY):
        st.error(
            f"이미지에서 텍스트를 읽지 못했습니다 — {error}\n\n"
            "아래 **JD 본문**에 직접 붙여넣어도 그대로 진행됩니다."
        )
        st.button("이미지에서 다시 추출", on_click=retry_jd_image)
    elif st.session_state.get(IMAGE_DIGEST_KEY) and st.session_state.get(JD_TEXT_KEY):
        st.success(
            "이미지에서 본문을 읽어 아래 칸에 채웠습니다. "
            "**내용을 확인·수정한 뒤** 실행하세요 — 잘못 읽힌 글자가 있을 수 있습니다."
        )


def render_input_form() -> tuple[bool, dict]:
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        company = col1.text_input("회사", placeholder="예: 테크노베이션")
        role = col2.text_input("직무", placeholder="예: 백엔드 엔지니어")

        target_date = st.date_input(
            "지원 목표 시점",
            value=date.today() + timedelta(days=90),
            help="이 날짜까지 남은 기간으로 '기간 내 채울 것'과 '포기할 것'이 갈린다.",
        )
        # `key`를 주는 이유는 이미지 추출 결과를 여기에 미리 넣기 위해서다. 위젯이
        # 만들어지기 **전에** `st.session_state[JD_TEXT_KEY]`가 채워져 있으면 폼이
        # 그 값으로 그려지고, 사용자는 그대로 고칠 수 있다.
        # 라벨은 "JD 본문" 그대로 둔다. 비워도 된다는 사실은 **비었을 때 보이는**
        # placeholder가 말한다 — 그게 정확히 그 정보가 필요한 순간이고, 라벨에
        # "(선택)"을 붙이면 백본 경로(붙여넣기가 가장 정확하다)를 스스로 깎아내린다.
        raw_jd = st.text_area(
            "JD 본문",
            height=280,
            key=JD_TEXT_KEY,
            placeholder=(
                "채용 공고 본문을 그대로 붙여넣으세요. 공고 URL만 넣어도 됩니다.\n"
                "비워 두면 회사 채용페이지에서 공고를 찾아 목록으로 보여 드립니다."
            ),
            help=(
                "붙여넣기가 가장 정확합니다(전문 보장). 비우면 공고를 찾아 "
                "**어느 공고로 분석할지 직접 고르는** 단계가 추가됩니다."
            ),
        )

        st.caption("내 프로필 — 없으면 판정 근거가 없어 모든 항목이 공백 고지로 나갑니다.")
        pcol1, pcol2 = st.columns([3, 2])
        uploaded = pcol1.file_uploader("ProfileJSON 업로드", type="json")
        # UC-1이 남긴 프로필을 다운로드·업로드 없이 바로 집는 지름길(T24).
        # **기본이 꺼짐인 것이 중요하다** — 켜 두면 예전 프로필이 사용자도 모르게
        # 판정 근거가 된다. 위젯 자체는 늘 그린다(없으면 비활성) — 있을 때만
        # 나타나면 화면이 파일 시스템 상태에 따라 달라진다.
        has_saved = load_saved_profile() is not None
        use_saved = pcol2.checkbox(
            "직전에 만든 프로필 사용",
            value=False,
            disabled=not has_saved,
            help="[내 프로필 만들기]에서 만든 가장 최근 프로필을 씁니다."
            if has_saved
            else "아직 만들어 둔 프로필이 없습니다.",
        )
        use_sample = pcol2.checkbox("샘플 프로필로 시연", value=True)

        submitted = st.form_submit_button("분석 실행", type="primary")

    return submitted, {
        "company": company.strip(),
        "role": role.strip(),
        "target_date": target_date,
        "raw_jd": raw_jd.strip(),
        "uploaded": uploaded,
        "use_saved": use_saved,
        "use_sample": use_sample,
    }


def start(graph, thread_id: str, slot) -> None:
    # 폼보다 먼저 그린다 — 추출 결과가 `text_area`에 실리려면 위젯 생성 전에
    # `st.session_state`에 들어가 있어야 한다.
    render_jd_image_input()

    submitted, form = render_input_form()
    if not submitted:
        return

    # **JD 본문은 더 이상 필수가 아니다 (T19).** 비어 있으면 회사 채용페이지에서
    # 공고를 찾아 H1에서 고르게 한다. 회사·직무는 여전히 필수다 — 회사 없이는
    # 발견할 곳이 없고, 직무 없이는 역량 추출의 기준이 없다.
    missing = [
        label
        for label, value in (("회사", form["company"]), ("직무", form["role"]))
        if not value
    ]
    if missing:
        st.error(f"{' · '.join(missing)}을(를) 입력해야 실행할 수 있습니다.")
        return

    if not form["raw_jd"]:
        st.info(
            "JD 본문이 비어 있어 **회사 채용페이지에서 공고를 찾아** 목록으로 보여 드립니다. "
            "찾지 못하면 분석할 공고가 없으니, 그때는 본문을 붙여넣어 주세요."
        )

    try:
        profile = load_profile(form["uploaded"], form["use_sample"], form["use_saved"])
    except ValueError as exc:
        st.error(f"ProfileJSON을 읽지 못했습니다: {exc}")
        return

    if profile is None:
        st.warning("프로필이 없어 판정 근거 없이 진행합니다 — 결과가 대부분 공백이 됩니다.")

    # 진행은 `st.spinner`가 아니라 `advance()`가 여는 `st.status` 패널이 보여준다 —
    # 어느 단계에서 시간이 걸리는지가 보여야 1~2분을 기다릴 수 있다(T13).
    advance(
        graph,
        thread_id,
        slot=slot,
        initial_input=initial_state(
            company=form["company"],
            role=form["role"],
            target_date=form["target_date"],
            raw_jd=form["raw_jd"],
            profile=profile,
        ),
    )
    st.rerun()


# --- 화면: 중단됨 -------------------------------------------------------------


RESUME_NOTICES: dict[str, str] = {
    "job_selection": (
        "공고를 찾았습니다. **어느 공고로 분석할지 골라 주세요** — "
        "여러 건을 함께 고르면 묶어서 분석합니다. 고르지 않은 공고는 읽지 않습니다."
    ),
    "delta_interview": (
        "판정 근거가 부족한 항목이 있어 잠시 멈췄습니다. "
        "답하면 **처음부터 다시 분석하지 않고** 이 지점부터 이어서 진행합니다."
    ),
    "ocr_review": (
        "이력서를 **또렷하게 읽지 못했습니다.** 아래는 읽어 온 그대로이며, "
        "잘못된 글자를 고치거나 비어 있으면 직접 붙여넣어 주세요. "
        "고친 내용으로 역량을 뽑습니다 — 그냥 두고 제출해도 진행은 됩니다."
    ),
    "level_survey": (
        "이력서에서 읽은 역량은 **미리 골라 뒀습니다** — 확인하고 고쳐 주세요. "
        "잘하는 정도가 아니라 **무엇을 해봤는지**로 고르면 됩니다. "
        "해보지 않은 항목은 그냥 두거나 '잘 모르겠음'을 고르세요."
    ),
}
DEFAULT_RESUME_NOTICE = (
    "추가 입력이 필요해 잠시 멈췄습니다. 제출하면 이 지점부터 이어서 진행합니다."
)


def resume_notice(questions) -> str:
    """중단 국면의 안내 문구. **중단점마다 다르다** (T19에서 둘이 됐다).

    T12가 쓴 문구는 델타 인터뷰 전용이라("판정 근거가 부족한 항목이…") 공고 선택
    화면에 그대로 뜨면 사용자가 무슨 말인지 알 수 없다. `kind`로 고르되 모르는
    중단점은 기본 문구로 떨어진다 — 내부 이름이 화면에 새지 않는 것은 T13과 같다.
    """
    for pending in questions or []:
        kind = normalize_prompt(pending.payload).kind
        if kind in RESUME_NOTICES:
            return RESUME_NOTICES[kind]
    return DEFAULT_RESUME_NOTICE


def resume(graph, thread_id: str, status: RunStatus, slot) -> None:
    """질문 폼을 그리고, 제출되면 **중단 지점부터** 재개한다.

    여기가 T12의 완료 조건이다 — 제출해도 `ingest → extract → decompose`는 다시
    돌지 않는다. `resume_or_start()`가 국면을 먼저 보고 `Command(resume=)`만
    태우기 때문이다. **중단점이 둘로 늘어도(T19 H1) 이 함수는 그대로다** —
    폼은 `app/hitl.py`가 페이로드를 보고 그리고, 여기서는 안내 문구만 고른다.
    """
    st.info(resume_notice(status.questions))

    answers = render_pending(status.questions)
    if answers is None:
        return

    advance(graph, thread_id, slot=slot, resume=answers)
    st.rerun()


# --- 화면: 완료 ---------------------------------------------------------------


def render_result(final: GraphState) -> None:
    """결과 화면. **그리는 일은 `render/cards.py`가 한다.**

    T09 때 이 파일 안에 임시로 두었던 인라인 렌더링(`render_summary`·`render_tracks`·
    `render_gaps`)을 T15에서 걷어냈다. 화면과 `.md`가 같은 `StrategyBrief`를 읽는
    두 렌더러로 갈라져야 내용이 어긋날 수 없고, 그 일치를
    `tests/test_render_parity.py`가 지킨다. 여기에 표시 로직을 다시 심으면 그 그물
    밖에서 화면만 따로 자란다.

    `match_results`·`required`는 `BriefCard`에 없는 상세(기준별 판정·중요도)를
    채우는 보강 데이터다. 없으면 그 줄만 빠지고 카드는 그대로 나온다.
    """
    brief = final.get("brief")
    if brief is None:
        st.error("브리프가 생성되지 않았습니다. 아래 [새 분석 시작]으로 다시 시도해주세요.")
        return

    render_brief(
        brief,
        matches=final.get("match_results"),
        required=final.get("required"),
    )

    # 델타 인터뷰 라운드 수는 브리프가 아니라 스레드의 이력이라 여기 남는다(D24).
    if rounds := (final.get("interview_round") or 0):
        st.caption(f"델타 인터뷰 {rounds}라운드를 거쳐 판정을 보강했습니다.")

    st.download_button(
        "전략 브리프 .md 다운로드",
        data=render_markdown(brief).encode("utf-8"),
        file_name=filename_for(brief),
        mime="text/markdown",
        type="primary",
    )


def finish(status: RunStatus) -> None:
    render_result(status.values)
    st.divider()
    st.button("새 분석 시작", on_click=reset_thread)


# --- 화면: 회사 분석 (UC-2) ----------------------------------------------------


def analysis_screen() -> None:
    st.caption("JD를 붙여넣으면 3트랙 전략 브리프를 만들어 `.md`로 내려받습니다.")

    graph = get_graph()
    thread_id = get_or_create_thread(st.session_state)

    # **실행 전에 상태부터 조회한다.** 이 한 줄이 rerun마다 분석이 처음부터
    # 다시 도는 것을 막는다.
    status = inspect_thread(graph, thread_id)

    if error := st.session_state.get(ERROR_KEY):
        st.error(f"실행이 중단됐습니다 — {error}")

    # 진행 표시가 들어갈 자리를 **하나만** 잡는다. 아래에서 지난 진행을 그린 뒤
    # 실행이 시작되면 같은 자리를 덮어쓰므로 패널이 둘로 늘어나지 않는다.
    slot = panel_slot()

    # 실행이 끝난 뒤에도 진행 표시는 남는다. rerun마다 화면이 처음부터 다시 그려지므로
    # 여기서 다시 그리지 않으면 중단 국면의 ⏸가 사라진다.
    if status.phase is not ThreadPhase.NOT_STARTED and TRACKER_KEY in st.session_state:
        render_panel(get_tracker(), slot=slot)

    if status.phase is ThreadPhase.NOT_STARTED:
        start(graph, thread_id, slot)
    elif status.is_interrupted:
        resume(graph, thread_id, status, slot)
    else:
        finish(status)


# --- 화면: 내 프로필 만들기 (UC-1) ---------------------------------------------
#
# UC-2와 **같은 상태기계**다 (미시작 → 중단됨 → 완료). 그래프·스레드·트래커만
# 다른 것을 쓰고, 실행은 `advance()` 한 곳을 그대로 지난다. 화면을 두 벌 만들지
# 않는 것이 D27의 규칙("실행 전에 반드시 상태를 조회한다")을 그래프가 둘이 돼도
# 지키는 방법이다.


def profile_ns() -> dict:
    """UC-1 전용 세션 네임스페이스. UC-2와 키가 겹치지 않는 유일한 이유다."""
    if PROFILE_NS not in st.session_state:
        st.session_state[PROFILE_NS] = {}
    return st.session_state[PROFILE_NS]


def get_profile_graph(ns: dict):
    """UC-1 그래프도 세션당 하나. checkpointer 수명이 세션과 같아진다(D27)."""
    if GRAPH_KEY not in ns:
        ns[GRAPH_KEY] = build_profile_graph(build_checkpointer())
    return ns[GRAPH_KEY]


def get_profile_tracker(ns: dict) -> ProgressTracker:
    """진행 표시. **문구 시트가 UC-2와 다르다** — `presets/profile_labels.yaml`.

    같은 시트를 쓰면 프로필을 만드는 내내 "회사 분석 진행 중"이 떠 있다. 노드
    라벨만이 아니라 제목이 달라야 해서 시트를 통째로 가른 것이며, 코드에 문구를
    적지 않는다는 T13의 규칙은 그대로다.
    """
    if TRACKER_KEY not in ns:
        names = [name for name, _ in profile_node_sequence()]
        ns[TRACKER_KEY] = ProgressTracker(names, labels=load_labels(PROFILE_LABELS))
    return ns[TRACKER_KEY]


def reset_profile() -> None:
    """새 프로필 = 새 스레드. 올려 둔 이력서 표식도 함께 지운다."""
    profile_ns().clear()


def store_resume_upload(uploaded, ns: dict) -> str:
    """업로드된 이력서를 디스크에 놓고 그 경로를 돌려준다.

    **여기서 파싱하지 않는다.** 파싱은 그래프의 `parse_resume` 노드가 하고, 여기서
    한 번 더 하면 스캔본에서 **OCR이 두 번 돈다**(T22 — 돈이 드는 경로다).
    `parse_resume(file_path)`가 경로를 받으므로 파일로 떨어뜨리는 일만 한다.

    같은 파일이 rerun마다 다시 쓰이지 않게 내용 해시를 표식으로 남긴다 —
    JD 이미지의 `IMAGE_DIGEST_KEY`와 같은 장치다.
    """
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    suffix = Path(uploaded.name).suffix.lower()
    path = UPLOAD_DIR / f"{digest[:16]}{suffix}"

    if ns.get(RESUME_PATH_KEY) != str(path) or not path.exists():
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    ns[RESUME_PATH_KEY] = str(path)
    ns[RESUME_NAME_KEY] = uploaded.name
    return str(path)


def render_resume_input(ns: dict) -> str:
    """이력서 업로더. **폼 밖이다.**

    T15b가 같은 자리에서 깨졌다(D48) — `st.form` 안의 위젯은 제출 전까지 rerun을
    일으키지 않아서, 파일이 서버에 닿는 시점이 곧 제출 순간이다. 그러면 "무엇을
    올렸는지 확인하고 실행"이 성립하지 않고, H4 보정 화면(T22b)이 들어설 자리도
    사라진다. `AppTest`는 폼 배칭을 모사하지 않아 이 차이를 못 잡으므로(D41),
    테스트는 동작이 아니라 **배치**(`form_id`가 비었는지)를 고정한다.
    """
    uploaded = st.file_uploader(
        "이력서 · 경력기술서",
        type=["pdf", "docx", "png", "jpg", "jpeg"],
        help="텍스트가 살아 있는 PDF·DOCX가 가장 정확합니다. 스캔본·사진은 글자 인식을 거칩니다.",
    )
    if uploaded is None:
        return ""

    path = store_resume_upload(uploaded, ns)
    st.success(f"**{uploaded.name}**을(를) 받았습니다. 아래 [프로필 만들기 시작]을 누르면 읽습니다.")
    return path


def render_profile_form() -> tuple[bool, str]:
    with st.form("profile_form"):
        role = st.text_input(
            "희망 직무",
            placeholder="예: 백엔드 엔지니어",
            help="이력서에서 이 직무와 관련된 역량을 뽑습니다. 비워도 진행됩니다.",
        )
        submitted = st.form_submit_button("프로필 만들기 시작", type="primary")
    return submitted, role.strip()


def profile_start(graph, thread_id: str, ns: dict, slot) -> None:
    # 업로더를 폼보다 먼저 그린다 — 폼 밖이라는 사실이 순서로도 드러난다.
    resume_path = render_resume_input(ns)
    submitted, role = render_profile_form()
    if not submitted:
        return

    # **이력서 없이도 진행한다.** 설문만으로도 프로필은 만들어지며(질문세트의
    # "폭 보완"), 그때 `extract`는 문서가 없어 LLM을 아예 부르지 않는다.
    if not resume_path:
        st.info(
            "이력서 없이 **설문만으로** 프로필을 만듭니다. "
            "이력서를 올리면 해당 역량이 미리 골라진 채로 나와 훨씬 빠릅니다."
        )

    advance(
        graph,
        thread_id,
        slot=slot,
        tracker=get_profile_tracker(ns),
        store=ns,
        initial_input=initial_profile_state(resume_path, role=role),
    )
    st.rerun()


def profile_resume_screen(graph, thread_id: str, ns: dict, status: RunStatus, slot) -> None:
    """H2 폼을 그리고 답을 받아 재개한다. **UC-2의 `resume()`과 같은 모양이다.**"""
    st.info(resume_notice(status.questions))

    answers = render_pending(status.questions, form_key="profile_hitl_form")
    if answers is None:
        return

    advance(
        graph,
        thread_id,
        slot=slot,
        tracker=get_profile_tracker(ns),
        store=ns,
        resume=answers,
    )
    st.rerun()


def render_profile_result(profile: ProfileJSON) -> None:
    """완성된 프로필 요약 + 다운로드.

    **다운로드가 정본 export 경로다**(설계 §10-3 — 프로필은 영속·다운로드/업로드).
    디스크 저장은 `build_profile` 노드가 이미 해 뒀고, 그건 같은 화면에서 바로
    이어 쓰기 위한 편의판이다.
    """
    covered = list(profile.coverage.values())
    average = sum(covered) / len(covered) if covered else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("역량", f"{len(profile.competencies)}개")
    col2.metric("레벨 좌표", f"{len(profile.level_coordinates)}개")
    col3.metric("평균 커버리지", f"{average * 100:.0f}%")

    if profile.coverage:
        st.caption("대분류별 커버리지 — 답한 축의 비율입니다. '해당 없음'은 분자에서 빠집니다.")
        st.bar_chart({"커버리지": {c.value: v for c, v in profile.coverage.items()}})

    st.download_button(
        "ProfileJSON 다운로드",
        data=profile_bytes(profile),
        file_name=download_filename(profile),
        mime="application/json",
        type="primary",
    )
    st.caption(
        "이 파일을 **회사 분석** 화면에서 업로드하면 판정 근거가 생겨 되묻는 질문이 줄어듭니다. "
        "같은 브라우저에서 바로 쓰려면 [직전에 만든 프로필 사용]을 켜세요."
    )


def profile_finish(status: RunStatus) -> None:
    profile = status.values.get("profile")
    if profile is None:
        st.error("프로필이 만들어지지 않았습니다. 아래 [프로필 다시 만들기]로 다시 시도해주세요.")
    else:
        st.success("프로필이 완성됐습니다.")
        render_profile_result(profile)

    st.divider()
    st.button("프로필 다시 만들기", on_click=reset_profile)


def profile_screen() -> None:
    st.caption("이력서를 올리고 **해본 만큼만** 고르면 `ProfileJSON`이 나옵니다. 회사 무관 · 한 번만 만들면 됩니다.")

    ns = profile_ns()
    graph = get_profile_graph(ns)
    thread_id = get_or_create_thread(ns)

    # UC-2와 같은 규칙 — 실행 전에 상태부터 조회한다(D27).
    status = inspect_thread(graph, thread_id)

    if error := ns.get(ERROR_KEY):
        st.error(f"실행이 중단됐습니다 — {error}")

    slot = panel_slot()
    if status.phase is not ThreadPhase.NOT_STARTED and TRACKER_KEY in ns:
        render_panel(get_profile_tracker(ns), slot=slot)

    if status.phase is ThreadPhase.NOT_STARTED:
        profile_start(graph, thread_id, ns, slot)
    elif status.is_interrupted:
        profile_resume_screen(graph, thread_id, ns, status, slot)
    else:
        profile_finish(status)


# --- 진입점 -------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="취업준비 Helper Agent", page_icon="🎯", layout="wide")
    st.title("🎯 취업준비 Helper Agent")

    # 모드 전환은 **사이드바**다. 본문에 두면 위젯 순서가 두 화면 사이에서 흔들리고,
    # 무엇보다 중단 국면의 폼 바로 위에 "다른 걸 하러 가기"가 붙어 오답을 부른다.
    #
    # **`format_func`을 쓰지 않는다.** 라벨을 그대로 선택지로 두고 뒤에서 되짚는다 —
    # 내부 키를 선택지로 두고 `format_func`으로 꾸미면 위젯 값이 키라서 화면 문자열과
    # 어긋나고, `AppTest`가 그 조합을 못 다뤄 화면 테스트가 아예 불가능해진다(실측).
    label = st.sidebar.radio(
        "무엇을 할까요",
        options=[MODE_LABELS[MODE_ANALYSIS], MODE_LABELS[MODE_PROFILE]],
        key=MODE_KEY,
        help="프로필을 먼저 만들어 두면 회사 분석에서 되묻는 질문이 줄어듭니다.",
    )
    mode = MODE_BY_LABEL.get(label, MODE_ANALYSIS)

    # 모드를 오가도 각자의 스레드는 그대로 서 있다 — 네임스페이스가 갈려 있어서다.
    if mode == MODE_PROFILE:
        profile_screen()
    else:
        analysis_screen()


if __name__ == "__main__":
    main()

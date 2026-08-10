"""HITL 재개 루프의 UI 절반 — 중단 페이로드를 폼으로 그리고 답변을 돌려준다.

Streamlit과 LangGraph의 실행 모델 충돌은 `graphs/session.py`가 이미 흡수했다
(`resume_or_start()` — 실행 전에 반드시 상태를 조회한다). 이 모듈은 그 위에서
**"중단됨" 국면 하나만** 담당한다: `interrupt()`가 실어 보낸 페이로드를 위젯으로
펼치고, 사용자가 제출한 값을 `Command(resume=)`에 넘길 형태로 정규화한다.

**질문 유형에 종속되지 않게 만든다.** H3(델타 인터뷰, T11)이 첫 사용처지만
H1(공고 선택, T19)·H4(OCR 보정, T22)도 같은 루프를 재사용한다(카드 불변식,
AGENTS.md "HITL은 하나의 재개 루프를 재사용한다"). 그래서 이 모듈이 아는 것은
`{텍스트, 선택지 유무}` 뿐이고, `kind`는 제목을 고르는 데만 쓴다 — 모르는
`kind`가 와도 폼은 그려진다.

페이로드 계약은 T11의 `build_interview_payload()`가 정한 형태를 최소 공배수로
삼는다. 다만 그것만 받으면 일반화가 아니므로, 아래 형태를 모두 받아들인다:

    {"kind": …, "questions": [{"question_id", "criterion_id", "text", "options"}]}
    {"text": …, "options": […]}          # 질문 1건짜리 페이로드
    [{"text": …}, …] / [Question, …]     # 목록만 온 경우
    "무엇을 …?"                           # 문자열 하나
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st

# 답변을 안 한 것과 빈 문자열을 답한 것은 구분하지 않는다 — 노드가 공백 답변을
# 버리므로(D28 부수결정 2) 여기서도 보내지 않는다.
BLANK = ""

# `kind` → 화면 제목. 모르는 kind는 기본값으로 떨어진다.
KIND_TITLES: dict[str, str] = {
    "delta_interview": "몇 가지만 더 알려주세요",
    "job_selection": "어느 공고로 분석할까요",
    "ocr_review": "인식 결과를 확인해주세요",
    "level_survey": "해본 만큼만 골라주세요",
}
DEFAULT_TITLE = "추가 입력이 필요합니다"

# `kind` → 제출 버튼 문구. 제목과 같은 이유로 여기 둔다 — 모르는 kind는 기본값.
SUBMIT_LABELS: dict[str, str] = {
    "job_selection": "이 공고로 분석 시작",
    "level_survey": "프로필 만들기",
}
DEFAULT_SUBMIT_LABEL = "답변 제출하고 이어서 분석"

SKIP_LABEL = "잘 모르겠음 / 답하지 않음"

# 다중 선택 답변을 **문자열 하나로** 이어 붙일 때 쓰는 구분자.
#
# 반환 타입을 `dict[str, str]`로 유지하려고 이렇게 한다 — 목록을 그대로 돌려주면
# `render_pending()`·재개 값·노드의 정규화가 전부 `str | list`를 받게 되고, 그 분기가
# H3(텍스트 답변) 경로까지 번진다. 줄바꿈을 쓰는 것은 **선택지 라벨에 줄바꿈이 없기**
# 때문이며, 라벨을 만드는 쪽(T19 `label_map()`)이 그것을 보장한다.
MULTI_SEPARATOR = "\n"


@dataclass(frozen=True)
class FormQuestion:
    """위젯 하나로 그려질 질문. 그래프 쪽 모델(`Question`)과 일부러 분리했다.

    UI는 `criterion_id`가 무엇인지 알 필요가 없고, 거꾸로 그래프는 위젯 key를
    알 필요가 없다.

    **`multi`는 T19(H1 공고 선택)가 더했다.** 원래 이 파일은 "T19·T22가 전혀 다른
    페이로드를 보내도 렌더링 코드를 고치지 않는다"고 적혀 있었는데, 그건 틀린
    낙관이었다 — 공고 선택은 §9-3이 명시적으로 multiselect를 요구하고 라디오로는
    표현이 안 된다. 다만 **고친 것은 위젯 선택 한 곳뿐**이고 페이로드 정규화·폼
    구성·반환 타입은 그대로다(R2 예외, 사용자 승인, DEVLOG D62).
    """

    key: str                       # 답변 dict의 키 — question_id 우선
    text: str
    options: list[str] | None = None
    multi: bool = False            # 켜지면 radio 대신 multiselect (T19 H1)
    section: str | None = None     # 섹션 제목. 바뀔 때만 찍힌다 (T23 H2)
    default: str | None = None     # 미리 골라 둘 선택지 (T23 H2 — 사전 채움)


@dataclass(frozen=True)
class InterruptPrompt:
    kind: str
    questions: list[FormQuestion]
    round: int | None = None
    max_rounds: int | None = None

    @property
    def title(self) -> str:
        return KIND_TITLES.get(self.kind, DEFAULT_TITLE)

    @property
    def progress_caption(self) -> str | None:
        """"몇 번 더 물어보나"를 사용자가 알 수 있게 한다 (D28 후속)."""
        if self.round is None or self.max_rounds is None:
            return None
        return f"{self.round} / {self.max_rounds} 라운드 · 질문 {len(self.questions)}건"


# --- 페이로드 정규화 ----------------------------------------------------------


def _as_mapping(item: Any) -> Mapping[str, Any]:
    """dict든 Pydantic 모델(`Question`)이든 dict처럼 읽을 수 있게 만든다."""
    if isinstance(item, Mapping):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump()
    return {"text": str(item)}


def _to_question(item: Any, index: int) -> FormQuestion:
    data = _as_mapping(item)

    # 키 우선순위: question_id → criterion_id → 위치. 노드가 양쪽을 다 받아
    # question_id로 정규화하므로(D28) UI는 있는 것을 그대로 보내면 된다.
    key = str(
        data.get("question_id")
        or data.get("criterion_id")
        or data.get("id")
        or f"q{index}"
    )

    options = data.get("options")
    if options is not None:
        options = [str(o) for o in options]

    section = data.get("section")
    default = data.get("default")

    return FormQuestion(
        key=key,
        text=str(data.get("text") or data.get("label") or data.get("question") or key),
        options=options or None,
        multi=bool(data.get("multi")),
        section=str(section) if section else None,
        default=str(default) if default is not None else None,
    )


def normalize_prompt(payload: Any) -> InterruptPrompt:
    """`interrupt()` 페이로드가 어떤 모양이든 렌더 가능한 형태로 정규화한다."""
    kind = ""
    round_number: int | None = None
    max_rounds: int | None = None
    items: Iterable[Any]

    if isinstance(payload, Mapping):
        kind = str(payload.get("kind") or "")
        round_number = payload.get("round")
        max_rounds = payload.get("max_rounds")
        raw = payload.get("questions")
        # "questions"가 없으면 페이로드 자체를 질문 1건으로 본다.
        items = raw if isinstance(raw, Sequence) and not isinstance(raw, str) else [payload]
    elif isinstance(payload, Sequence) and not isinstance(payload, str):
        items = payload
    else:
        items = [payload]

    return InterruptPrompt(
        kind=kind,
        questions=[_to_question(item, i) for i, item in enumerate(items)],
        round=round_number if isinstance(round_number, int) else None,
        max_rounds=max_rounds if isinstance(max_rounds, int) else None,
    )


# --- 렌더링 -------------------------------------------------------------------


def _render_widget(question: FormQuestion, widget_key: str) -> str:
    if question.options and question.multi:
        # 기본 선택 없음 — radio와 같은 이유다. 손대지 않은 항목이 "고른 것"으로
        # 넘어가면 사용자가 의도하지 않은 공고를 분석하게 된다.
        picked = st.multiselect(
            question.text,
            options=question.options,
            default=[],
            key=widget_key,
            placeholder="하나 이상 고르세요",
        )
        return MULTI_SEPARATOR.join(str(p) for p in picked)

    if question.options:
        # 기본은 index=None — 아무것도 미리 고르지 않는다. 기본 선택을 두면 사용자가
        # 손대지 않은 항목이 "답한 것"으로 넘어가 근거 없는 판정의 빌미가 된다.
        #
        # **`default`가 올 때만 예외다 (T23 H2).** 레벨 측정의 사전 채움은 "없는 근거로
        # 미리 고르는 것"이 아니라 **이력서에서 추출된 근거로 고른 것**을 사용자가
        # 확인·수정하게 하는 것이라, 위 우려가 성립하지 않는다. 값을 보내지 않은 축은
        # 여전히 아무것도 안 고른 채로 뜬다.
        options = [*question.options, SKIP_LABEL]
        preselected = (
            options.index(question.default) if question.default in options else None
        )
        choice = st.radio(
            question.text,
            options=options,
            index=preselected,
            key=widget_key,
        )
        return BLANK if choice in (None, SKIP_LABEL) else str(choice)

    return st.text_area(question.text, key=widget_key, height=90, placeholder="자유롭게 적어주세요")


def render_prompt_form(
    prompt: InterruptPrompt, *, form_key: str, submit_label: str = DEFAULT_SUBMIT_LABEL
) -> dict[str, str] | None:
    """질문을 **한 폼에 모아** 그리고, 제출됐을 때만 답변 dict를 반환한다.

    질문을 하나씩 던지지 않는 이유는 라운드마다 재개·재실행 비용이 붙고 사용자도
    지치기 때문이다(카드 불변식 · §9-3 배치 원칙). 제출 전에는 `None`을 반환하며,
    그 사이 Streamlit이 몇 번을 rerun하든 그래프는 중단 지점에 그대로 서 있다.

    공백 답변은 **키째로 빼고** 보낸다 — 노드가 어차피 버리지만(D28), 여기서
    빼두면 "무엇에 답했나"가 재개 값만 봐도 드러난다.
    """
    with st.form(form_key):
        st.subheader(prompt.title)
        caption = prompt.progress_caption
        if caption:
            st.caption(caption)

        # 섹션은 **바뀔 때만** 찍는다. 레벨 측정(T23)은 문항이 26개라 구분이 없으면
        # 어디까지가 한 덩어리인지 읽히지 않는다. 섹션을 안 보낸 페이로드는 이 줄이
        # 한 번도 실행되지 않아 예전 화면 그대로다.
        answers: dict[str, str] = {}
        current_section: str | None = None
        for question in prompt.questions:
            if question.section and question.section != current_section:
                st.markdown(f"**{question.section}**")
                current_section = question.section
            answers[question.key] = _render_widget(
                question, f"{form_key}__{question.key}"
            )

        submitted = st.form_submit_button(submit_label, type="primary")

    if not submitted:
        return None

    return {key: value.strip() for key, value in answers.items() if value and value.strip()}


def render_pending(questions, *, form_key: str = "hitl_form") -> Any | None:
    """중단된 스레드의 질문 전부를 그리고 `Command(resume=)`에 넘길 값을 만든다.

    입력은 `graphs/session.py`의 `RunStatus.questions`(= `PendingQuestion` 목록).
    중단이 하나면 답변 dict를 그대로, 여럿이면 `{interrupt_id: 답변}`으로 감싼다 —
    langgraph는 중단이 여러 개일 때 어느 중단의 답인지 id로 구분한다.
    지금 그래프의 중단점은 하나뿐이라 실제로 타는 경로는 앞쪽이다.
    """
    if not questions:
        return None

    collected: dict[str, dict[str, str]] = {}
    for pending in questions:
        prompt = normalize_prompt(pending.payload)
        answers = render_prompt_form(
            prompt,
            form_key=f"{form_key}__{pending.interrupt_id}",
            submit_label=SUBMIT_LABELS.get(prompt.kind, DEFAULT_SUBMIT_LABEL),
        )
        if answers is None:
            return None  # 아직 제출 전 — 하나라도 안 됐으면 재개하지 않는다
        collected[pending.interrupt_id] = answers

    if len(collected) == 1:
        return next(iter(collected.values()))
    return collected

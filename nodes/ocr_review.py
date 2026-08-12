"""H4 — 인식 결과 사용자 보정 (T22b, 설계도 §9-3 H4 · §8-5).

스캔 이력서는 OCR(T22)까지 태워도 못 건지는 경우가 있다. 그때 흐름을 세우는
대신 **읽어 온 텍스트를 그대로 보여주고 사람이 고치게** 한다 — 고친 텍스트가
`extract`로 흘러가 프로필의 근거가 된다.

```
parse_resume → ocr_review(H4) → extract → level_survey(H2) → build_profile
                   ↑ interrupt()  이 카드(T22b)
```

**멈출지 말지는 판별자 하나만 본다.** `tools/parse_resume.py::needs_manual_correction`
이며, 이 노드는 층 이름(`텍스트레이어`/`OCR`/`실패`)을 알지 못한다 — 층 표는
T21의 것이고 여기서 다시 쓰면 두 벌이 된다(T16 `is_uncollected()`와 같은 규약).
잘 읽힌 이력서는 이 노드를 **그냥 지나간다**(중단 없음).

세 가지를 지킨다 (카드 불변식)
------------------------------
① **재개 루프는 T12 것을 그대로 쓴다.** 여기서 하는 일은 `interrupt()`에 페이로드를
   싣는 것뿐이고, 폼은 `app/hitl.py`가 그리며 재개는 `graphs/session.py`가 한다.
② **OCR 실패를 전체 실패로 만들지 않는다.** 빈 칸이 떠도 사용자가 직접 채워
   진행할 수 있고, 아무것도 안 채워도 흐름은 계속된다(원문 유지).
③ **고친 텍스트는 상태에 남는다.** 화면에서만 고쳐지고 `source_docs`에 안 실리면
   아무 일도 안 한 것이다.

`interrupt()`는 이 함수 중간에서 멈추고, 재개되면 **노드가 처음부터 다시 실행**되며
`interrupt()`만 답을 반환한다(D27). 그래서 위쪽(`docs_needing_review`·`build_payload`)은
전부 순수 계산이어야 하고 실제로 그렇다 — 파일도 네트워크도 건드리지 않는다.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from langgraph.types import interrupt

from contracts.enums import Confidence
from contracts.models import SourceDocument
from contracts.state import GraphState
from tools.parse_resume import needs_manual_correction

# 중단 페이로드의 `kind`. `app/hitl.py::KIND_TITLES`·`app/main.py::RESUME_NOTICES`가
# 이 문자열로 제목과 안내 문구를 고른다.
KIND = "ocr_review"

# 사람이 직접 읽고 고친 텍스트의 신뢰도.
#
# **`CONFIDENCE_BY_LAYER`(T21)에 넣지 않는다.** 그 표는 "어느 *추출 층*에서 건졌나"의
# 정의점이고, 사용자 보정은 층이 아니다 — 자동 추출이 실패한 뒤 사람이 눈으로 대조해
# 넣은 것이라 어떤 층보다 신뢰가 높다. 이 값을 안 올리면 보정을 마친 문서가 계속
# `needs_manual_correction()` 참으로 남아, 하류(T25 게이트)가 "아직 못 읽은 문서"로
# 센다. **빈 답변에는 적용하지 않는다** — 고친 것이 없으면 신뢰도도 그대로다.
CORRECTED_CONFIDENCE = Confidence.HIGH

PROMPT_TEMPLATE = (
    "**{title}**에서 읽어 온 내용입니다. 잘못 읽힌 글자를 고치거나, "
    "비어 있으면 직접 붙여넣어 주세요."
)


def docs_needing_review(docs: Iterable[SourceDocument]) -> list[SourceDocument]:
    """보정이 필요한 문서만. 판별자는 T21의 것 하나뿐이다."""
    return [doc for doc in docs if needs_manual_correction(doc.raw_text, doc.confidence)]


def build_payload(docs: Sequence[SourceDocument]) -> dict:
    """`interrupt()`에 실어 보낼 페이로드. `app/hitl.py`가 이대로 폼을 그린다.

    **문서 하나 = 위젯 하나**이고 키는 `doc_id`다. 지금 프로필 그래프에 실리는
    문서는 이력서 한 건뿐이지만, 키를 위치가 아니라 id로 두면 문서가 여럿인
    페이로드에서도 답이 엉키지 않는다.

    `value`에 읽어 온 텍스트를 실어 보내는 것이 이 카드의 핵심이다 — 빈 칸이 뜨면
    "인식 결과를 확인하고 고친다"는 화면의 목적이 사라진다.
    """
    return {
        "kind": KIND,
        "questions": [
            {
                "question_id": doc.doc_id,
                "text": PROMPT_TEMPLATE.format(title=doc.title),
                "value": doc.raw_text,
            }
            for doc in docs
        ],
    }


def apply_corrections(
    docs: Sequence[SourceDocument], answers: Mapping[str, str]
) -> list[SourceDocument]:
    """보정 텍스트를 문서에 얹는다. **목록 전체를 돌려준다.**

    `GraphState.source_docs`에는 reducer가 없어 노드가 돌려주는 목록이 통째로
    이전 것을 대체한다. 고친 문서만 돌려주면 나머지가 조용히 사라진다.

    답이 없거나 공백뿐인 문서는 **원문 그대로 둔다** — 사용자가 못 고쳤다고 해서
    건진 텍스트까지 버리면 불변식 ②(실패를 전체 실패로 만들지 않는다)가 깨진다.
    """
    corrected: list[SourceDocument] = []
    for doc in docs:
        text = (answers.get(doc.doc_id) or "").strip()
        if not text or text == doc.raw_text:
            corrected.append(doc)
            continue
        corrected.append(
            doc.model_copy(update={"raw_text": text, "confidence": CORRECTED_CONFIDENCE})
        )
    return corrected


def ocr_review(state: GraphState) -> dict:
    """H4 — 못 읽은 문서가 있으면 멈추고, 고친 텍스트를 상태에 되돌린다."""
    docs = list(state.get("source_docs") or [])
    pending = docs_needing_review(docs)
    if not pending:
        # 잘 읽힌 이력서, 또는 이력서 없이 설문만 도는 경로. 멈출 이유가 없다.
        return {}

    answers = interrupt(build_payload(pending))

    corrected = apply_corrections(docs, answers if isinstance(answers, Mapping) else {})
    if corrected == docs:
        return {}  # 고친 것이 없다 — 상태를 건드리지 않는다
    return {"source_docs": corrected}

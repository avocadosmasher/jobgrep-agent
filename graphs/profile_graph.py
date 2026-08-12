"""프로필 그래프(UC-1) 조립 — 이력서 한 장에서 `ProfileJSON`까지.

```
START → parse_resume → ocr_review(H4) → extract → level_survey(H2) → build_profile
                            ↑ interrupt() H4 (T22b)      ↑ interrupt() H2 (T23)
                              못 읽었을 때만 멈춘다                    → profile_gate → END
                                                                        ↑ 커버리지 게이트 (T25)
```

**왜 분석 그래프와 분리하나** (설계도 §10-3, 카드 맥락):
① 프로필은 회사와 무관하다 — 두 번째 회사를 분석할 때 UC-1을 다시 태울 이유가 없다.
② 그래프가 작아져 디버깅·데모가 쉽다.
③ "사용자 프로필 = 영속, 다운로드/업로드"라는 상태 모델과 정확히 일치한다.

**불변식 — 사용자 프로필과 수집 결과를 혼합하지 않는다.** 여기서 그 불변식은
"안 섞이게 조심한다"가 아니라 **배선의 부재**로 이행된다. 이 그래프에는 회사 수집
도구(T16~T19)가 한 줄도 배선돼 있지 않아서, `source_docs`에 들어갈 수 있는 것은
이력서뿐이다. 섞으려면 노드를 새로 꽂아야 하고, 그건 이 파일을 고치는 일이라
리뷰에 걸린다. 스레드도 분석 그래프와 별개다(`app/main.py`가 네임스페이스를 가른다).

**checkpointer는 선택이 아니다.** H2가 `interrupt()`로 멈추므로 checkpointer 없이는
재개가 성립하지 않는다. 분석 그래프처럼 `interactive=` 스위치를 두지 않는 이유는
D63과 같다 — 물어볼 사람이 없으면 레벨을 **시스템이 대신 골라야** 하는데 그것이
§12-2가 금지하는 임의 선택이다. 비대화형 UC-1은 만들 수 있는 물건이 아니다.

계약에 이력서 칸이 없다
----------------------
`GraphState`에는 이력서 파일 경로를 담을 자리가 없고 R1이라 못 늘린다. 그래서
`raw_jd_input`("사용자가 준 원본 입력")을 프로필 모드의 이력서 경로로 쓴다 —
이 모드에는 JD가 없어 칸이 비어 있고, T18이 `GateStatus`에 칸이 없어 `missing`
라벨을 재사용한 것(D59)·T23이 설문 산출을 `profile`로 돌려준 것(D71)과 같은 판단이다.
**문자열을 직접 쓰지 말고 `RESUME_PATH_KEY`·`initial_profile_state()`를 쓸 것.**

`extract`는 분석 그래프 것을 그대로 재사용한다
--------------------------------------------
T23이 이미 그 전제로 쓰였다 — `nodes/level_survey.py`가 "보유 역량은 `required`에서
읽는다. 프로필 모드에서는 `extract`가 그 칸을 채운다"고 적어 뒀다. 노드를 새로
만들면 같은 배선이 두 벌이 된다.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from contracts.enums import SourceType
from contracts.models import ProfileJSON, SourceDocument
from contracts.state import GraphState
from nodes.analysis_nodes import extract
from nodes.gates import profile_gate
from nodes.level_survey import level_survey
from nodes.ocr_review import ocr_review
from tools.parse_resume import parse_resume

# 이력서 파일 경로가 실리는 상태 칸. 위 "계약에 이력서 칸이 없다" 참조.
RESUME_PATH_KEY = "raw_jd_input"

# 이력서 문서의 고정 id. 붙여넣기 JD(`jd-pasted-001`)와 겹치지 않아야 한다 —
# `extract`가 `req-{doc_id}-NN`으로 역량 id를 발급하므로, 겹치면 UC-2에서
# 요구 역량 id와 보유 역량 id가 충돌한다.
RESUME_DOC_ID = "resume-001"
RESUME_TITLE = "이력서"

# 이력서는 회사 문서가 아니다. `SourceType`에 RESUME이 없고 R1이라 못 늘리므로
# `기타`로 둔다 — `source_coverage()`가 세는 세 유형(JD·기술블로그·인재상) 어디에도
# 안 걸리는 것이 오히려 맞다. 이력서는 회사 정보의 충족률에 들어갈 물건이 아니다.
RESUME_SOURCE_TYPE = SourceType.OTHER

# 저장 위치. 체크포인트와 같은 폴더 아래라 `.gitignore`가 이미 덮고 있다.
PROFILE_DIR = Path(".jobprep") / "profiles"
SAVED_GLOB = "profile_*.json"


# --- 상태 만들기 ---------------------------------------------------------------


def initial_profile_state(resume_path: str = "", *, role: str = "") -> GraphState:
    """UC-1 초기 상태. **경로 칸 이름을 호출부에 새지 않게** 하는 유일한 통로다."""
    return {
        "mode": "profile",
        "role": role,
        RESUME_PATH_KEY: resume_path,
    }


# --- 노드 ----------------------------------------------------------------------


def parse_resume_node(state: GraphState) -> dict:
    """이력서 파일 → `SourceDocument` 1건. 파싱 층은 `tools/parse_resume.py`(T21).

    **신뢰도는 문서의 `confidence`로 옮겨 실린다.** T21이 돌려주는 `Confidence`는
    "어느 층에서 건졌는지"(텍스트레이어 상 / OCR 중 / 실패 하)이고, 그 정보가
    갈 자리는 `SourceDocument.confidence`뿐이다 — 상태에 따로 담을 칸이 없다.

    **건진 텍스트가 비어도 문서를 만든다.** T21의 계약이 "실패해도 건진 텍스트를
    그대로 돌려준다"이고, 신뢰도를 버리면 H4(T22b)가 보정 화면을 띄울 근거를 잃는다.
    빈 문서 하나가 `extract` 호출 1회를 헛되이 쓰는 것은 알고 있는 값이며,
    **그 호출을 막는 것이 H4의 일**이다 — 사람이 고칠 기회를 먼저 줘야 한다.

    **그 자리는 이제 채워졌다** — 다음 노드가 `ocr_review`(H4, T22b)이고, 같은
    판별자(`needs_manual_correction`)로 멈출지를 정한 뒤 `raw_text`를 갈아끼운다.
    """
    path = (state.get(RESUME_PATH_KEY) or "").strip()
    if not path:
        # 이력서 없이 설문만으로 프로필을 만드는 경로다(폭 보완만). 빈 문서를
        # 만들면 위와 달리 **읽을 것 자체가 없었던 경우**와 구분이 안 된다.
        return {"source_docs": []}

    text, confidence = parse_resume(path)

    doc = SourceDocument(
        doc_id=RESUME_DOC_ID,
        source_type=RESUME_SOURCE_TYPE,
        company="",  # 이력서에는 대상 회사가 없다 — 프로필은 회사 무관이다
        title=RESUME_TITLE,
        collected_at=date.today(),
        raw_text=text,
        confidence=confidence,
    )
    return {"source_docs": [doc]}


def build_profile_node(state: GraphState) -> dict:
    """완성된 `ProfileJSON`을 파일로 떨군다 (설계도 §10-3).

    **조립은 이 노드가 하지 않는다.** `nodes/level_survey.py::build_profile()`이
    이미 순수 함수로 끝냈고(같은 이름이 둘이라 헷갈리기 쉽다), 여기서 다시 조립하면
    같은 규칙이 두 벌이 된다. 이 노드가 맡는 것은 **영속과 경계**다 —
    "UC-1의 산출은 파일이다"라는 §10-3의 모델을 코드로 만드는 자리다.

    **저장 실패는 흐름을 세우지 않는다.** 정본 export 경로는 화면의 다운로드
    버튼이고, 디스크 저장은 그 편의판(다운로드 없이 UC-2에서 바로 집어 쓰기)이다.
    쓰기가 막혔다고 방금 만든 프로필을 잃게 하는 것은 손해가 더 크다.
    """
    profile = state.get("profile")
    if profile is None:
        # H2가 답을 못 받았거나 상류가 비었다. 빈 프로필을 지어내지 않는다.
        return {}

    try:
        save_profile(profile)
    except OSError:
        pass

    return {"profile": profile}


# --- 영속 ----------------------------------------------------------------------


def profile_bytes(profile: ProfileJSON) -> bytes:
    """다운로드·저장이 **같은 바이트**를 쓰게 하는 유일한 직렬화 지점.

    두 경로가 다른 함수를 쓰면 한쪽만 바뀌었을 때 "내려받은 파일은 업로드가 되는데
    저장된 파일은 안 되는" 종류의 어긋남이 난다.
    """
    return profile.model_dump_json(indent=2).encode("utf-8")


def download_filename(profile: ProfileJSON) -> str:
    """사용자가 내려받을 이름. 사람이 폴더에서 알아볼 수 있어야 한다."""
    return f"내프로필_{profile.built_at:%Y%m%d}.json"


def _saved_name(payload: bytes, built_at: date) -> str:
    """디스크에 남길 이름. **내용 해시가 붙어 같은 날 여러 번 만들어도 안 덮인다.**

    파일명을 ASCII로 두는 것은 의도적이다 — 저장 경로는 사람이 읽을 자리가 아니고,
    OS·인코딩에 따라 한글 파일명이 깨지는 환경이 있다(Windows 실행이 기본이다).
    """
    return f"profile_{built_at:%Y%m%d}_{hashlib.sha256(payload).hexdigest()[:8]}.json"


def save_profile(profile: ProfileJSON, *, directory: str | Path | None = None) -> Path:
    """프로필을 디스크에 남기고 그 경로를 돌려준다."""
    target = Path(directory) if directory is not None else PROFILE_DIR
    target.mkdir(parents=True, exist_ok=True)

    payload = profile_bytes(profile)
    path = target / _saved_name(payload, profile.built_at)
    path.write_bytes(payload)
    return path


def load_saved_profile(*, directory: str | Path | None = None) -> ProfileJSON | None:
    """가장 최근에 저장된 프로필. 없거나 읽을 수 없으면 `None`.

    **읽기 실패를 예외로 올리지 않는다.** 이 함수를 부르는 곳은 UC-2 입력 폼이고,
    거기서 예외가 나면 프로필과 상관없는 회사 분석까지 통째로 막힌다. 깨진 파일은
    건너뛰고 그다음으로 최근인 것을 본다.
    """
    target = Path(directory) if directory is not None else PROFILE_DIR
    if not target.is_dir():
        return None

    for path in sorted(target.glob(SAVED_GLOB), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            return ProfileJSON.model_validate_json(path.read_bytes())
        except (OSError, ValueError):
            continue
    return None


# --- 배선 ----------------------------------------------------------------------

# 실행 순서 정본. 진행 표시(T13)도 이 목록을 봐야 화면과 배선이 어긋나지 않는다.
PROFILE_NODE_SEQUENCE: list[tuple[str, object]] = [
    ("parse_resume", parse_resume_node),
    # H4 (T22b). **못 읽은 이력서일 때만** 멈춘다 — 잘 읽힌 문서는 그냥 지나간다.
    # 자리가 여기인 이유: 고친 텍스트가 `extract`의 입력이어야 한다.
    ("ocr_review", ocr_review),
    ("extract", extract),  # 분석 그래프 것을 그대로 재사용 (위 모듈 설명 참조)
    ("level_survey", level_survey),
    ("build_profile", build_profile_node),
    # T25 — 완성된 프로필의 커버리지를 재 `gate_status`에 남긴다(§12-1 UC-1 하드 요건).
    # 화면이 "미완성"을 띄우는 근거이며, 여기서 멈추거나 되돌리지는 않는다.
    ("profile_gate", profile_gate),
]

PROFILE_NODE_NAMES: list[str] = [name for name, _ in PROFILE_NODE_SEQUENCE]


def profile_node_sequence() -> list[tuple[str, object]]:
    return list(PROFILE_NODE_SEQUENCE)


def build_profile_graph(checkpointer):
    """컴파일된 프로필 그래프를 반환한다. **checkpointer는 필수다.**

    분기가 없는 직선 배선이며, 그중 둘이 `interrupt()`로 멈춘다 — H2(레벨 측정)는
    항상, H4(OCR 보정, T22b)는 이력서를 못 읽었을 때만.
    checkpointer가 없으면 멈춘 자리를 저장할 곳이 없어 재개가 성립하지 않으므로,
    langgraph 안쪽에서 나는 알아보기 힘든 오류 대신 여기서 먼저 끊는다.

    checkpointer는 만들지 않고 **받는다** — 여기서 만들면 import 시점에 sqlite
    파일이 생기는 부작용이 되고, 테스트가 인메모리로 갈아끼울 수 없다(D27).
    """
    if checkpointer is None:
        raise ValueError(
            "프로필 그래프에는 checkpointer가 필요하다 — H2(레벨 측정)가 "
            "interrupt()로 멈추므로 저장할 곳 없이는 재개할 수 없다"
        )

    sequence = profile_node_sequence()
    names = [name for name, _ in sequence]

    builder = StateGraph(GraphState)
    for name, fn in sequence:
        builder.add_node(name, fn)

    builder.add_edge(START, names[0])
    for src, dst in zip(names, names[1:]):
        builder.add_edge(src, dst)
    builder.add_edge(names[-1], END)

    return builder.compile(checkpointer=checkpointer)

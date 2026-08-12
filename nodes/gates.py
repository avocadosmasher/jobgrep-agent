"""품질 게이트 + 실패 처리 (T25, 설계도 §12-1·§12-2·§12-3).

> **최악의 실패는 멈추는 것이 아니라 그럴듯하게 지어내는 것.**

게이트가 그 방어선이다. 에러 없이 돌아도 내용이 빈껍데기면 실패이며, 그 사실이
**결과물에 적혀 나가야** 한다.

이 카드가 실제로 닫는 구멍
--------------------------
게이트 판정 자체는 T18이 이미 만들어 뒀다(`nodes/collect.py::build_gate_status`) —
**그런데 아무도 안 읽었다.** `gate_status`는 상태에 쓰이기만 하고 화면도 브리프도
보지 않는 쓰기 전용 칸이었다(§2-1이 경계하는 "아무도 안 부르는 모듈"의 거울상).
그래서 이 모듈은 판정 규칙을 **다시 만들지 않는다.** T18 것을 그대로 부르고,
대신 세 가지를 더한다.

① **읽는 쪽을 만든다** — `gate_notices()`가 결과 화면 상단에 강등·예산 소진을 띄운다.
② **표기가 판정과 어긋나던 자리를 맞춘다** — `source_coverage()`(T04)는 문서의
   *유형*만 보고 세서 **본문이 빈 JD 문서도 "수집됨"으로 친다.** 그러면 신뢰등급은
   "추정 기반"인데 미수집 목록에는 JD가 없는, 서로 모순된 리포트가 나간다.
   이 노드가 `source_docs`에서 빈 문서를 걷어내 두 표기를 한 근거 위에 세운다.
③ **UC-1 커버리지 게이트를 신설한다** — §12-1의 세 게이트 중 유일하게 없던 것이다.

차단이 아니라 강등인 이유
------------------------
§12-1은 하드 요건 미달 시 "리포트 미생성 **또는** 추정 기반으로 강등 + 상단 명시"
둘 중 하나를 고르라고 한다. 이 제품은 **강등**을 고른다 — §12의 관통 규칙이
"부분 실패를 전체 실패로 만들지 않는다"이고, 사용자가 아무것도 못 받는 것보다
"어디까지 알아냈는지 밝힌 결과"가 낫기 때문이다. 차단하려면 조건부 간선으로
`END`로 빠져야 하는데 그건 §2-1이 막는 배선 알고리즘 변경이기도 하다.

임계값 (실측 근거는 DEVLOG D78)
-------------------------------
카드가 "임계값이 전부 미정"이라 적어 둔 셋 중 이 카드가 정하는 것은 **UC-1
커버리지**뿐이다. 나머지 둘은 이미 정본이 있다 — JD 최소 1건은 §12-1이 못 박았고,
수집 tool call 상한(12)은 T18이 `DEFAULT_TOOL_BUDGET`으로 갖고 있다. 여기서 다시
정의하면 두 벌이 된다.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from contracts.models import ProfileJSON, SourceDocument
from contracts.state import GateStatus, GraphState
from nodes.collect import (
    RELIABILITY_ESTIMATED,
    RELIABILITY_NORMAL,
    budget_exhausted,
    build_gate_status,
    collected_job_ids,
)
from tools.fetch_jd import is_uncollected

# --- 임계값 -------------------------------------------------------------------

# UC-2 하드 요건. §12-1이 정한 값이라 실측 대상이 아니다 — `build_gate_status`가
# 이미 이 기준으로 판정하므로 여기서는 **문서화용 상수**이고 판정은 T18 것을 쓴다.
MIN_JD_DOCS = 1

# UC-1 하드 요건 — 프리셋 대분류 커버리지 (실측으로 정했다, D78).
#
# 프리셋은 26축 × 대분류 5개(각 5~6축)다. 두 값을 함께 보는 이유는 **폭과 깊이가
# 따로 놀기 때문**이다. 한 대분류만 전부 답하면 평균 0.20으로 깊이는 있으나 폭이
# 1/5이고, JD가 다른 영역을 요구하면 그 프로필은 아무 근거도 못 준다. 반대로
# 대분류 5개에 1축씩만 답하면 폭은 5/5여도 26축 중 5축(평균 0.19)이라 얇다.
#
#   답 없음                  폭 0/5 · 평균 0.00  → 미달
#   이력서 사전 채움만(2축)   폭 1/5 · 평균 0.08  → 미달 (설문을 안 한 프로필이다)
#   한 대분류만 전부(5축)     폭 1/5 · 평균 0.20  → 미달 (폭이 없다)
#   대분류 3개 각 1축         폭 3/5 · 평균 0.12  → 미달 (얇다)
#   자기 영역 2개 + 1축(11축) 폭 3/5 · 평균 0.44  → 통과
#   축 절반(13축)            폭 3/5 · 평균 0.52  → 통과
MIN_PROFILE_BREADTH = 3
MIN_PROFILE_COVERAGE = 0.20


# --- UC-2 게이트 ---------------------------------------------------------------


def usable_documents(docs: Iterable[SourceDocument]) -> list[SourceDocument]:
    """본문을 실제로 건진 문서만. 판별자는 T16의 `is_uncollected()` 하나다.

    수집 실패가 예외가 아니라 **빈 본문 문서**로 표현되기 때문에(D52) 이 걸음이
    필요하다. 빈 문서를 1건으로 세면 게이트도 표기도 헛통과한다.
    """
    return [doc for doc in docs if not is_uncollected(doc)]


def quality_gate(state: GraphState) -> dict:
    """UC-2 품질 게이트 — 집계 직후·브리프 직전.

    **판정 규칙을 새로 쓰지 않는다.** `build_gate_status`(T18)가 정본이고 이 노드는
    그것을 **최종 문서 묶음**에 다시 적용한다. 수집 시점(collect) 이후로 H1 선택분이
    붙거나 본문이 빈 채 남은 문서가 있어서, 리포트 직전에 한 번 더 재는 것이 맞다.

    자리가 `build_brief` 바로 앞인 이유 — 여기서 정리한 `source_docs`·
    `selected_job_ids`가 곧 브리프 메타의 입력이다. 신뢰등급(`selected_jobs`)과
    미수집 목록(`source_coverage`)이 **같은 문서 묶음**에서 나와야 서로 모순되지 않는다.
    """
    docs = list(state.get("source_docs") or [])
    usable = usable_documents(docs)

    return {
        "source_docs": usable,
        "selected_job_ids": collected_job_ids(usable),
        # 예산 소진은 수집 시점의 사실이라 이전 상태에서 이어받는다 — 문서만 봐서는
        # "못 돈 것"과 "돌았는데 없던 것"이 구분되지 않는다.
        "gate_status": build_gate_status(
            usable, exhausted=budget_exhausted(state.get("gate_status"))
        ),
    }


# --- UC-1 게이트 ---------------------------------------------------------------


def coverage_breadth(profile: ProfileJSON) -> int:
    """답이 하나라도 있는 대분류 수 — **폭**."""
    return sum(1 for value in profile.coverage.values() if value > 0)


def average_coverage(profile: ProfileJSON) -> float:
    """대분류 커버리지의 평균 — **깊이**.

    화면이 이미 같은 숫자를 보여준다(`render_profile_result`의 "평균 커버리지").
    게이트가 다른 식으로 재면 "77%인데 왜 미완성이냐"는 화면이 된다.
    """
    values = list(profile.coverage.values())
    return sum(values) / len(values) if values else 0.0


def profile_is_complete(profile: ProfileJSON | None) -> bool:
    """UC-1 하드 요건 — 폭과 깊이를 **둘 다** 넘겨야 한다 (§12-1)."""
    if profile is None:
        return False
    return (
        coverage_breadth(profile) >= MIN_PROFILE_BREADTH
        and average_coverage(profile) >= MIN_PROFILE_COVERAGE
    )


def unanswered_categories(profile: ProfileJSON) -> list[str]:
    """답이 하나도 없는 대분류의 이름. 사용자가 어디를 채우면 되는지 알아야 한다."""
    return [category.value for category, value in profile.coverage.items() if value <= 0]


def profile_gate_status(profile: ProfileJSON | None) -> GateStatus:
    """프로필을 `GateStatus`로 요약한다 — 순수 함수, LLM 없음.

    **`jd_count`는 0이다.** UC-1에는 공고가 없고, 계약에 칸을 늘릴 수 없다(R1).
    `missing`에 담기는 것은 `SourceType`이 아니라 **대분류 이름**인데, 두 그래프는
    상태를 공유하지 않으므로 어휘가 섞이지 않는다(T18이 예산 라벨을 같은 칸에
    넣은 것과 같은 판단).

    `reliability`도 어휘를 새로 만들지 않고 `BriefMeta`의 둘을 그대로 쓴다 —
    미완성 프로필로 회사 분석을 돌리면 실제로 나오는 등급이 "추정 기반"이라,
    이 값이 곧 **UC-2에 들어갔을 때의 예고**가 된다.
    """
    if profile is None:
        return GateStatus(
            jd_count=0, hard_gate_passed=False, reliability=RELIABILITY_ESTIMATED, missing=[]
        )

    passed = profile_is_complete(profile)
    return GateStatus(
        jd_count=0,
        hard_gate_passed=passed,
        reliability=RELIABILITY_NORMAL if passed else RELIABILITY_ESTIMATED,
        missing=unanswered_categories(profile),
    )


def profile_gate(state: GraphState) -> dict:
    """UC-1 커버리지 게이트 — 프로필이 완성된 직후.

    **프로필이 없으면 아무것도 쓰지 않는다.** 만들다 만 스레드에 "미완성"을 찍으면
    화면이 실패를 두 번 말한다(그쪽은 이미 "프로필이 만들어지지 않았습니다"를 띄운다).
    """
    profile = state.get("profile")
    if profile is None:
        return {}
    return {"gate_status": profile_gate_status(profile)}


# --- 화면 문구 -----------------------------------------------------------------
#
# 문구를 게이트가 갖는 이유 — 판정과 설명이 갈리면 한쪽만 고쳐진다. 화면(app/main.py)은
# 부르기만 한다. 반대로 **브리프가 이미 표에 적어 주는 것은 여기서 다시 말하지 않는다**
# (신뢰등급·미수집 목록은 `render/cards.py`·`render/markdown.py`가 이미 찍는다).


HARD_GATE_NOTICE = (
    "**분석할 공고 본문이 한 건도 없습니다.** 회사 자료만으로 낸 **추정 기반** 결과라, "
    "요구 역량이 실제 공고와 다를 수 있습니다 — 공고 본문을 붙여넣으면 정확해집니다."
)
BUDGET_NOTICE = (
    "**수집 예산이 소진돼 일부 자료를 못 읽었습니다.** 지금까지 모은 것으로 낸 "
    "부분 리포트이며, 못 읽은 자료는 [미수집]에 적혀 있습니다."
)


def gate_notices(gate_status: GateStatus | None) -> list[str]:
    """결과 화면 **상단**에 띄울 경고 (§12-1 "강등 + 상단 명시").

    소프트 요건 미수집은 여기 넣지 않는다 — 브리프 헤더와 공백 고지에 이미 나가며,
    같은 사실을 두 곳에서 말하면 정작 하드 게이트 경고가 묻힌다.
    """
    if not gate_status:
        return []

    notices: list[str] = []
    if not gate_status.get("hard_gate_passed"):
        notices.append(HARD_GATE_NOTICE)
    if budget_exhausted(gate_status):
        notices.append(BUDGET_NOTICE)
    return notices


def profile_notice(profile: ProfileJSON | None) -> str | None:
    """UC-1 완료 화면 — 미완성이면 무엇이 모자란지 말한다. 완성이면 `None`."""
    if profile is None or profile_is_complete(profile):
        return None

    unanswered = unanswered_categories(profile)
    tail = (
        f" 아직 답이 없는 영역: {' · '.join(unanswered)}."
        if unanswered
        else ""
    )
    return (
        f"이 프로필은 **미완성**입니다 — 답한 대분류 "
        f"{coverage_breadth(profile)}/{len(profile.coverage)}개, "
        f"평균 커버리지 {average_coverage(profile):.0%}."
        f"{tail} 이대로도 회사 분석은 되지만 **되묻는 질문이 늘어납니다.**"
    )


def entry_warning(profile: ProfileJSON | None) -> str | None:
    """UC-2 진입 경고 (§12-1 "UC-2 진입 경고"). 완성 프로필이면 `None`.

    프로필이 아예 없는 경우는 여기서 말하지 않는다 — 화면이 이미 그 경고를 갖고
    있고(`프로필이 없어 판정 근거 없이 진행합니다`), 두 줄이 겹치면 둘 다 안 읽힌다.
    """
    if profile is None or profile_is_complete(profile):
        return None
    return (
        f"고른 프로필이 **미완성**입니다(평균 커버리지 {average_coverage(profile):.0%}) — "
        "판정 근거가 얇아 공백 고지와 되묻는 질문이 늘어납니다. "
        "[내 프로필 만들기]에서 못 고른 영역을 채우면 결과가 좋아집니다."
    )


def missing_labels(gate_status: GateStatus | None) -> Sequence[str]:
    """게이트가 적어 둔 미수집·미응답 라벨. 화면이 목록으로 쓸 때만 필요하다."""
    if not gate_status:
        return ()
    return tuple(gate_status.get("missing") or ())

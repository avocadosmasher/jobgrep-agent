"""H1 · 공고 발견 → 사용자 선택 (설계도 §6-2, §9-3 H1, §12-2).

노드 둘이 짝으로 움직인다.

    discover_jobs   회사 채용페이지에서 공고를 **열거**한다 (본문 없음, 메타만)
    select_job      그 목록을 사용자에게 보여주고 **고르게 한다** (interrupt)
                    → 고른 것만 `fetch_jd_body`로 본문을 가져온다

## 왜 시스템이 안 고르나

"백엔드 개발자" 공고가 3건인데 시스템이 하나를 골라 버리면 **사용자는 자기가 뭘
분석당했는지 모른다.** 카드가 "임의 선택·병합 금지"라고 쓴 것이 이것이고, §12-2가
"부서/공고 모호 → H1으로 처리"로 표에 박아 둔 것도 같은 규칙이다. 이 모듈에서
자동으로 좁히는 것은 하나도 없다 — `discover_jobs`의 직무명 정렬조차 순서만 바꾸지
목록에서 빼지 않는다.

## 언제 안 묻나

**사용자가 이미 JD 본문이나 URL을 준 경우 두 노드 다 통째로 건너뛴다**(카드 규칙,
§9-3 H1 생략 조건). 그때는 발견할 것도 고를 것도 없다 — 백본이 이미 답을 갖고 있다.
`raw_jd_input`이 그 판별자이며, 이 조건 덕에 기존 붙여넣기 경로는 네트워크를 한 번도
더 타지 않는다.

## 재개 루프는 새로 만들지 않는다

`interrupt()` → `app/hitl.py`가 폼을 그림 → `Command(resume=)`. T11(H3)이 쓰는
바로 그 루프이며 여기서 새로 만드는 메커니즘은 없다(AGENTS.md · 카드 불변식).
페이로드에 `multi: True`를 실으면 `app/hitl.py`가 `st.multiselect`로 그린다.

**`interrupt()` 앞에는 부작용을 두지 않는다.** 재개되면 노드가 처음부터 다시 실행되고
`interrupt()`만 답을 반환하기 때문이다(D27·D28). 그래서 본문 수집은 전부 그 뒤에 있다.
"""

from __future__ import annotations

from datetime import date

from langgraph.types import interrupt

from contracts.models import SourceDocument
from contracts.state import GraphState
from tools.discover import discover_jobs as discover_jobs_tool
from tools.discover import group_jobs
from tools.fetch_jd import fetch_jd_body, is_uncollected

# 답변 dict의 키. 질문이 하나뿐이라 고정값이면 충분하다.
SELECTION_KEY = "job_selection"

# `app/hitl.py`가 여러 선택을 하나의 문자열로 이어 붙일 때 쓰는 구분자.
# 라벨에 줄바꿈이 없다는 것이 이 표현의 전제이며 `job_label()`이 그것을 보장한다.
MULTI_SEPARATOR = "\n"

PROMPT_TEXT = "분석할 공고를 고르세요 (여러 건을 함께 고르면 묶어서 분석합니다)"


# --- 노드 ①: 발견 -------------------------------------------------------------


def discover(state: GraphState) -> dict:
    """회사 채용페이지에서 공고를 열거한다. 사용자 입력이 있으면 건너뛴다.

    발견 실패는 실패가 아니다 — 빈 목록이면 `select_job`이 아무것도 안 묻고,
    JD가 한 건도 없는 채로 진행되어 하드 게이트가 그 사실을 보고한다(§12-1).
    """
    if _has_user_input(state):
        return {}

    company = (state.get("company") or "").strip()
    if not company:
        return {"discovered_jobs": []}

    return {
        "discovered_jobs": discover_jobs_tool(company, state.get("role") or "")
    }


# --- 노드 ②: 선택 (H1) ---------------------------------------------------------


def select_job(state: GraphState) -> dict:
    """발견된 공고를 사용자에게 물어(중단) 고른 것의 본문을 가져온다.

    반환하는 부분 갱신:
        `selected_job_ids` — 사용자가 고른 공고 중 **본문을 실제로 건진** 것의 id
        `source_docs`      — 그 본문 문서들 (기존 문서 뒤에 붙는다)

    고른 공고의 본문 수집이 실패하면 그 건은 조용히 빠진다 — `fetch_jd_body`가
    빈 문서를 돌려주므로(D52) `is_uncollected()`로 걸러 낸다. 전부 실패하면
    `selected_job_ids`가 비고, 그건 "JD 없음"으로 하드 게이트에 잡힌다.
    """
    jobs = state.get("discovered_jobs") or []
    if _has_user_input(state) or not jobs:
        return {}  # 고를 것이 없으면 묻지 않는다

    # --- 여기까지 순수 계산. `interrupt()` 앞에 부작용을 두지 않는다(D28). ---
    by_label = label_map(jobs)
    replies = interrupt(build_selection_payload(jobs))

    chosen = resolve_selection(replies, by_label)
    if not chosen:
        # 아무것도 고르지 않은 것도 사용자의 선택이다. 대신 고르지 않는다.
        return {"selected_job_ids": []}

    collected = [
        doc
        for doc in (_fetch_body(job) for job in chosen)
        if not is_uncollected(doc)
    ]

    return {
        "selected_job_ids": [doc.doc_id for doc in collected],
        "source_docs": [*(state.get("source_docs") or []), *collected],
    }


# --- 페이로드 · 라벨 -----------------------------------------------------------


def build_selection_payload(jobs: list[SourceDocument]) -> dict:
    """`interrupt()`로 UI에 던질 페이로드.

    질문은 **하나**이고 선택지가 여럿이다 — 공고마다 질문을 만들면 "이 공고를
    분석할까요?"를 20번 묻는 꼴이 된다(§9-3 배치 원칙).

    선택지는 대분류로 묶어 순서만 정돈한다. 그룹을 선택 단위로 만들지 않는 것은
    그것이 곧 "임의 병합"이기 때문이다 — 묶음은 읽기 편하라고 있는 것이다.
    """
    return {
        "kind": "job_selection",
        "questions": [
            {
                "question_id": SELECTION_KEY,
                "text": PROMPT_TEXT,
                "options": job_labels(jobs),
                "multi": True,
            }
        ],
    }


def label_map(jobs: list[SourceDocument]) -> dict[str, SourceDocument]:
    """`{선택지 라벨: 공고}` — 대분류 순으로 정돈하고 **라벨 중복을 없앤다.**

    라벨이 곧 답변 값이다(위젯이 문자열을 돌려준다). 두 공고가 같은 라벨을 가지면
    어느 쪽을 고른 건지 알 수 없는데, 제목만 같고 부서가 다른 공고는 실제로 흔하다.
    그래서 겹치면 뒤에 일련번호를 붙인다 — **라벨의 유일성이 이 노드의 계약이다.**

    라벨에 줄바꿈이 없다는 것도 여기서 보장된다(`title`은 `read_links()`가 이미
    공백을 접어 뒀다). `MULTI_SEPARATOR`가 줄바꿈이라 이게 깨지면 선택이 쪼개진다.
    """
    mapping: dict[str, SourceDocument] = {}

    for group, members in group_jobs(jobs).items():
        for job in members:
            base = " ".join(f"[{group}] {job.title}".split())
            label = base
            serial = 2
            while label in mapping:
                label = f"{base} ({serial})"
                serial += 1
            mapping[label] = job
    return mapping


def job_labels(jobs: list[SourceDocument]) -> list[str]:
    """선택지 문자열 목록 — `label_map()`의 키 순서 그대로."""
    return list(label_map(jobs))


def resolve_selection(replies, by_label: dict[str, SourceDocument]) -> list[SourceDocument]:
    """UI가 돌려준 답을 공고 목록으로 되돌린다 — 순수 함수.

    받는 모양이 여럿이다. `app/hitl.py`는 `{question_id: "라벨\\n라벨"}` 형태로
    주지만, 재개 값을 직접 넣는 테스트나 다른 UI는 목록으로 줄 수 있다. 셋 다 받되
    **모르는 라벨은 조용히 버린다** — 없는 공고를 지어내지 않는다.

    결과 순서는 응답 순서가 아니라 **선택지 순서**를 따른다. 같은 선택이면 같은
    결과가 나와야 하고(멱등), 그래야 `selected_job_ids`가 화면 순서와 일치한다.
    """
    picked: set[str] = set()

    def take(value) -> None:
        if isinstance(value, dict):
            for item in value.values():
                take(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                take(item)
        elif value is not None:
            for part in str(value).split(MULTI_SEPARATOR):
                if part.strip():
                    picked.add(part.strip())

    take(replies)
    return [job for label, job in by_label.items() if label in picked]


def _has_user_input(state: GraphState) -> bool:
    """사용자가 이미 JD 본문·URL을 줬는가 — H1 생략 조건(§9-3)."""
    return bool((state.get("raw_jd_input") or "").strip())


def _fetch_body(job: SourceDocument) -> SourceDocument:
    """발견된 공고의 본문을 가져온다. 실패하면 빈 본문 문서다(예외 아님, D52)."""
    return fetch_jd_body(
        job.url or "",
        company=job.company,
        title=job.title,
    )

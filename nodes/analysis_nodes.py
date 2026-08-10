"""분석 그래프의 노드 — `(GraphState) -> dict` 부분 갱신만 반환한다.

노드는 `tools/`의 도구를 호출하는 배선일 뿐이며 비즈니스 로직을 갖지 않는다.
판정·집계·트랙 배정 규칙은 전부 도구 안에 있고, 여기서는 상태의 어느 칸을
도구에 넘기고 결과를 어느 칸에 담을지만 정한다.

범위 — 수집(T16~T19)은 아직 없다. 붙여넣은 JD 1건으로 끝까지 관통하는 것이 기본
경로이며, HITL(T11)과 후보쌍 검색(T14b)이 그 위에 얹혀 있다.
"""

from __future__ import annotations

from datetime import date

from contracts.enums import Confidence, Level, MatchState, SourceType
from contracts.models import CompetencyRecord, MatchResult, SourceDocument
from contracts.state import GraphState
from tools.aggregate import aggregate_states
from tools.brief import build_brief_meta, build_strategy_brief
from tools.decompose import decompose_criteria
from tools.extract import extract_competencies
from tools.fill_slots import fill_brief_slots
from tools.retrieve import retrieve_candidates
from tools.verify import verify_criteria

# 붙여넣기 경로는 공고가 하나뿐이라 문서 id가 고정이다. T16이 실제 수집을 붙이면
# 여기가 아니라 `fetch_jd_body`가 id를 발급한다.
PASTED_DOC_ID = "jd-pasted-001"

# 브리프 메타의 소스 충족률을 재는 분모. 설계도 §8-3의 수집 도구 3종에 대응한다.
EXPECTED_SOURCE_TYPES = (SourceType.JD, SourceType.TECH_BLOG, SourceType.VALUES)

# 요구 역량 1건당 후보로 둘 보유 역량 수. `my_level`이 실제로 쓰는 것은 top-1
# 하나뿐이지만, 상태에 남는 쌍은 감사와 후속 재사용(T23의 프리셋 축 매핑)을 위한
# 것이라 넉넉히 둔다. 이 숫자를 줄여도 `my_level` 결과는 바뀌지 않는다.
CANDIDATE_TOP_K = 3


def ingest_pasted_jd(state: GraphState) -> dict:
    """붙여넣은 JD 본문을 `SourceDocument` 1건으로 정규화한다.

    P0의 수집 대체물이다. `fetch_jd_body`(T16)가 들어오면 이 노드는 그 도구를
    부르는 형태로 바뀐다. 본문이 비면 빈 목록을 반환하고, 뒤 노드들이 각자
    빈 입력을 처리한다 — 여기서 예외를 던져 그래프를 세우지 않는다.
    """
    raw = (state.get("raw_jd_input") or "").strip()
    if not raw:
        return {"source_docs": [], "selected_job_ids": []}

    doc = SourceDocument(
        doc_id=PASTED_DOC_ID,
        source_type=SourceType.JD,
        company=state.get("company", ""),
        title=f"{state.get('role', '')} 채용공고 (붙여넣기)".strip(),
        collected_at=date.today(),
        raw_text=raw,
        # 사용자가 원문을 직접 준 경로라 수집 실패·부분 파싱 위험이 없다.
        confidence=Confidence.HIGH,
    )
    return {"source_docs": [doc], "selected_job_ids": [doc.doc_id]}


def extract(state: GraphState) -> dict:
    """수집 문서에서 요구 역량을 배치 1회 호출로 추출한다."""
    docs = state.get("source_docs") or []
    return {"required": extract_competencies(docs, state.get("role", ""))}


def retrieve(state: GraphState) -> dict:
    """요구 역량과 보유 역량을 임베딩 유사도로 이어 후보쌍을 만든다 (§7-2 1단계).

    `aggregate`가 이걸 받아 `my_level`을 채운다. `extract` 직후에 두는 이유는 입력
    (`required`·`profile`)이 여기서 갖춰지기 때문이며, 소비 지점 바로 앞이 아니라
    준비 지점에 두면 나중에 `verify` 프롬프트를 후보쌍으로 좁히는 선택지가 열린 채로
    남는다.

    프로필이 없으면 **호출하지 않는다.** 임베딩 API 왕복은 공짜가 아니고, 짝지을
    상대가 없으면 결과가 어차피 비기 때문이다.
    """
    required = state.get("required") or []
    profile = state.get("profile")
    owned = profile.competencies if profile else []
    if not required or not owned:
        return {"candidate_pairs": []}

    return {
        "candidate_pairs": retrieve_candidates(required, owned, top_k=CANDIDATE_TOP_K)
    }


def decompose(state: GraphState) -> dict:
    """역량을 체크 가능한 기준으로 배치 분해한다."""
    return {"criteria": decompose_criteria(state.get("required") or [])}


def verify(state: GraphState) -> dict:
    """기준을 프로필로 판정하고, 판정 불가분은 질문으로 받아 둔다.

    P0에서는 questions가 나와도 흐름을 멈추지 않는다 — 델타 인터뷰는 T11이다.
    다만 상태에는 남겨 둔다. 버리면 T11이 다시 만들어야 하고, 무엇을 판정하지
    못했는지가 브리프의 공백 고지(§11-2)로 이어져야 하기 때문이다.
    """
    criteria = [c for group in (state.get("criteria") or {}).values() for c in group]
    profile = state.get("profile")
    if not criteria or profile is None:
        return {"verdicts": [], "pending_questions": []}

    verdicts, questions = verify_criteria(
        criteria, profile, state.get("interview_answers")
    )
    return {"verdicts": verdicts, "pending_questions": questions}


def my_level_for(
    comp_id: str,
    match_state: MatchState,
    top_match: dict[str, str],
    owned_by_id: dict[str, CompetencyRecord],
) -> Level | None:
    """근거로 대응이 확인된 역량에만 보유 레벨을 붙인다 (§7-2 + §7-3).

    **유사도가 아니라 판정이 게이트다.** 임베딩 top-1은 "가장 가까운 것"이지
    "대응하는 것"이 아니다 — 보유가 `Docker 사용`뿐인데 요구가 `Kubernetes 운영`이면
    top-1은 Docker이고, 그 레벨을 그대로 찍으면 **없는 경력을 지어내는 것**이다.
    `retrieve_candidates`는 유사도 점수를 일부러 반환하지 않으므로(§7-2, D39)
    임계값으로 거를 수단 자체가 없다.

    그래서 `aggregate_states`의 판정을 필터로 쓴다. MET/ADJACENT가 나왔다는 것은
    기준별 근거 인용을 거쳐 대응이 실제로 확인됐다는 뜻이므로, 그때만 레벨을
    가져온다. UNMET이면 후보쌍이 있어도 `None`이다 — 이 조합이 §7-2와 §7-3을
    동시에 지키는 유일한 방법이다.
    """
    if match_state is MatchState.UNMET:
        return None

    owned_id = top_match.get(comp_id)
    if owned_id is None:
        return None

    owned = owned_by_id.get(owned_id)
    # 보유 역량에 레벨이 안 적혀 있을 수 있다 (CompetencyRecord.level은 optional).
    # 그때는 추정하지 않고 비워 둔다.
    return owned.level if owned else None


def aggregate(state: GraphState) -> dict:
    """역량별로 기준 판정을 3-state로 집계한다 — LLM 없음, 규칙만 (§8-2).

    후보쌍은 `retrieve` 노드가 이미 계산해 상태에 넣어 뒀고 여기서는 **읽기만
    한다.** 그래서 이 노드는 임베딩이 붙은 뒤에도 순수 규칙 함수로 남는다.

    `my_level` 결정 규칙은 `my_level_for()`에 있다. 요구 역량 하나에 후보가 여럿
    올 수 있으므로 **첫 번째만** 남기는데, `retrieve_candidates`가 같은 요구 역량
    안에서 유사도 내림차순으로 돌려주기 때문에 그게 top-1이다.
    """
    criteria_by_comp = state.get("criteria") or {}
    verdict_by_criterion = {v.criterion_id: v for v in state.get("verdicts") or []}

    profile = state.get("profile")
    owned_by_id = {c.comp_id: c for c in (profile.competencies if profile else [])}

    top_match: dict[str, str] = {}
    for required_id, owned_id in state.get("candidate_pairs") or []:
        top_match.setdefault(required_id, owned_id)

    results: list[MatchResult] = []
    for comp in state.get("required") or []:
        comp_criteria = criteria_by_comp.get(comp.comp_id) or []
        if not comp_criteria:
            continue  # 분해되지 않은 역량은 집계 대상이 아니다

        comp_verdicts = [
            verdict_by_criterion[c.criterion_id]
            for c in comp_criteria
            if c.criterion_id in verdict_by_criterion
        ]
        match_state = aggregate_states(comp_criteria, comp_verdicts)
        results.append(
            MatchResult(
                comp_id=comp.comp_id,
                name=comp.name,
                category=comp.category,
                required_level=comp.level,
                my_level=my_level_for(comp.comp_id, match_state, top_match, owned_by_id),
                state=match_state,
                verdicts=comp_verdicts,
            )
        )

    return {"match_results": results}


def source_coverage(docs: list[SourceDocument]) -> tuple[float, list[str]]:
    """수집된 출처 유형으로 충족률과 미수집 목록을 낸다 (■ 코드 결정론)."""
    present = {doc.source_type for doc in docs}
    missing = [t.value for t in EXPECTED_SOURCE_TYPES if t not in present]
    covered = len(EXPECTED_SOURCE_TYPES) - len(missing)
    return covered / len(EXPECTED_SOURCE_TYPES), missing


def build_brief(state: GraphState) -> dict:
    """집계 결과를 3트랙 브리프 골격(■)으로 세운다. ◇ 슬롯은 아직 빈 채다."""
    coverage, missing = source_coverage(state.get("source_docs") or [])
    meta = build_brief_meta(
        company=state.get("company", ""),
        role=state.get("role", ""),
        selected_jobs=state.get("selected_job_ids") or [],
        target_date=state["target_date"],
        source_coverage=coverage,
        missing_sources=missing,
    )
    return {"brief": build_strategy_brief(state.get("match_results") or [], meta)}


def fill_slots(state: GraphState) -> dict:
    """브리프의 서술형 슬롯(◇)을 채운다.

    `fill_brief_slots`는 LLM 실패 시 입력 브리프를 그대로 돌려주므로(DEVLOG D20)
    이 노드가 별도로 실패를 처리하지 않는다 — 슬롯이 비면 렌더러가 그 사실을
    표기한다(D18).
    """
    brief = state.get("brief")
    if brief is None:
        return {}
    return {"brief": fill_brief_slots(brief)}

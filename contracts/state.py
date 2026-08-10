from datetime import date
from typing import Literal, TypedDict

from .models import (
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    MatchResult,
    ProfileJSON,
    Question,
    SourceDocument,
    StrategyBrief,
)


class GateStatus(TypedDict):
    jd_count: int
    hard_gate_passed: bool
    reliability: str
    missing: list[str]


class GraphState(TypedDict, total=False):
    mode: Literal["profile", "analysis"]
    # 입력
    company: str
    role: str
    target_date: date
    profile: ProfileJSON | None
    raw_jd_input: str | None          # P0 백본: 붙여넣기
    # 수집
    discovered_jobs: list[SourceDocument]
    selected_job_ids: list[str]
    source_docs: list[SourceDocument]
    # 매칭
    required: list[CompetencyRecord]
    candidate_pairs: list[tuple[str, str]]
    criteria: dict[str, list[Criterion]]
    verdicts: list[CriterionVerdict]
    match_results: list[MatchResult]
    # HITL
    pending_questions: list[Question]
    interview_answers: dict[str, str]
    interview_round: int
    # 예산
    tool_budget: int
    iteration: int
    # 출력
    gate_status: GateStatus
    brief: StrategyBrief | None


# 노드 규약: 모든 노드는 `def node(state: GraphState) -> dict` — 부분 상태 갱신
# dict를 반환한다. 전체 상태를 반환하지 않는다.

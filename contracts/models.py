from datetime import date

from pydantic import BaseModel

from .enums import (
    Category,
    Confidence,
    Importance,
    Level,
    MatchState,
    SourceType,
    Track,
    VerdictState,
)


class Evidence(BaseModel):
    source_name: str
    url: str | None = None
    quote: str                      # 원문 인용 — 지어내기 금지
    collected_at: date


class SourceDocument(BaseModel):    # Layer-1
    doc_id: str
    source_type: SourceType         # 코드가 결정 (어느 도구가 가져왔나)
    company: str
    department: str | None = None
    title: str
    url: str | None = None
    published_at: date | None = None
    collected_at: date              # 코드가 채움
    raw_text: str                   # §12-5 인젝션 격리 대상
    keywords: list[str] = []        # 결정론적 추출
    confidence: Confidence          # 출처유형별 규칙


class CompetencyRecord(BaseModel):  # Layer-2
    comp_id: str
    category: Category
    name: str                       # 원문 표현 보존 — 일반화·병합 금지
    importance: Importance
    level: Level | None = None      # 요구 레벨(JD) 또는 보유 레벨(프로필)
    evidence: list[Evidence] = []


class Criterion(BaseModel):
    criterion_id: str
    comp_id: str
    text: str                       # 체크 가능한 단일 기준
    is_required: bool               # 집계 규칙 입력


class CriterionVerdict(BaseModel):
    criterion_id: str
    state: VerdictState
    rationale: str
    evidence: list[Evidence] = []


class Question(BaseModel):          # 델타 인터뷰 / 레벨 측정
    question_id: str
    criterion_id: str | None = None
    text: str
    options: list[str] | None = None


class MatchResult(BaseModel):
    comp_id: str
    name: str
    category: Category
    required_level: Level | None
    my_level: Level | None
    state: MatchState               # aggregate_states 산출
    verdicts: list[CriterionVerdict]
    is_strength: bool = False       # §7-3


class BriefCard(BaseModel):
    comp_id: str
    name: str
    track: Track
    state: MatchState
    required_level: Level | None
    my_level: Level | None
    priority: int | None = None     # 트랙2 순서
    body: str                       # ◇ LLM 슬롯
    evidence: list[Evidence] = []


class BriefMeta(BaseModel):         # ■ 전부 코드 결정론
    company: str
    role: str
    selected_jobs: list[str]
    target_date: date
    days_remaining: int
    source_coverage: float          # 0.0~1.0
    missing_sources: list[str]
    reliability: str                # "정상" | "추정 기반"


class ProfileJSON(BaseModel):       # UC-1 산출 · 영속
    competencies: list[CompetencyRecord]
    level_coordinates: dict[str, Level]
    coverage: dict[Category, float]
    built_at: date


class StrategyBrief(BaseModel):     # 최종 산출 — 렌더러 2종의 공통 입력
    meta: BriefMeta
    summary_counts: dict[MatchState, int]   # ■
    summary_line: str                       # ◇
    track1: list[BriefCard]
    track2: list[BriefCard]
    track3: list[BriefCard]
    culture_fit: str | None = None
    gaps: list[str] = []                    # 판단 못 한 항목 + 이유

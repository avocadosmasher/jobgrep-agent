from .enums import Confidence, MatchState
from .models import (
    BriefMeta,
    CompetencyRecord,
    Criterion,
    CriterionVerdict,
    MatchResult,
    ProfileJSON,
    Question,
    SourceDocument,
    StrategyBrief,
)


# 순수 규칙 (LLM 없음)
def aggregate_states(
    criteria: list[Criterion],
    verdicts: list[CriterionVerdict],
) -> MatchState:
    """한 역량에 속한 기준·판정을 결정론적 규칙으로 집계한다.

    입력: 같은 comp_id를 공유하는 Criterion 목록과 그에 대응하는 CriterionVerdict 목록.
    출력: MatchState — 필수 기준 전부 충족 시 MET, 필수 기준 절반 이상 충족 시
        ADJACENT, 그 외 UNMET (설계도 §7-2).
    불변식: LLM 호출 절대 금지 — 이 함수의 결정론성이 전체 파이프라인 멱등성의 근거다.
    """
    ...


# LLM 사용 — 전부 배치 호출 (설계도 §8-4)
def extract_competencies(
    docs: list[SourceDocument], role: str
) -> list[CompetencyRecord]:
    """수집 문서 묶음에서 요구/보유 역량을 배치 1회 호출로 추출한다.

    입력: SourceDocument 목록(회사 JD·기술블로그·인재상 등), 대상 직무명.
    출력: CompetencyRecord 목록 — 각 레코드는 evidence를 최소 1건 포함해야 한다.
    불변식: 문서별·역량별 개별 호출 금지(배치 호출만 허용). 역량명(name)은 원문
        표현 그대로 보존하며 일반화·병합하지 않는다. evidence.quote는 대응하는
        SourceDocument.raw_text에 실제로 존재하는 문자열이어야 한다.
    """
    ...


def decompose_criteria(
    comps: list[CompetencyRecord]
) -> dict[str, list[Criterion]]:
    """역량 하나를 체크 가능한 기준 3~5개로 배치 분해한다.

    입력: CompetencyRecord 목록.
    출력: comp_id → Criterion 목록 매핑. 각 Criterion.text는 예/아니오로
        판정 가능한 단일 문장이어야 한다.
    불변식: 역량별 개별 호출 금지 — 전체 목록을 배치로 처리한다.
    """
    ...


def verify_criteria(
    criteria: list[Criterion],
    profile: ProfileJSON,
    answers: dict[str, str] | None = None,
) -> tuple[list[CriterionVerdict], list[Question]]:
    """기준 목록을 프로필(및 인터뷰 답변)로 판정하고, 판정 불가 기준은 질문으로 승격한다.

    입력: 판정 대상 Criterion 목록, 사용자 ProfileJSON, 이전 델타 인터뷰 라운드의
        answers(question_id → 답변 텍스트, 없으면 None).
    출력: (판정된 CriterionVerdict 목록, 판정 불가 기준에 대한 Question 목록)의 튜플 —
        입력으로 받은 각 Criterion은 반드시 둘 중 한쪽에만 속한다.
    불변식: 배치 호출. CriterionVerdict.evidence의 quote는 profile 내 실제 원문
        인용이어야 하며, 근거 없이는 MET/PARTIAL/UNMET 판정을 내리지 않고
        UNKNOWN으로 두거나 Question으로 승격한다.
    """
    ...


def build_strategy_brief(
    results: list[MatchResult], meta: BriefMeta
) -> StrategyBrief:
    """집계된 매칭 결과를 3트랙(내세울것/채울것/포기할것) 전략 브리프로 구성한다.

    입력: MatchResult 목록(집계 완료 상태), BriefMeta(코드가 결정론적으로 채운
        메타데이터 — company/role/days_remaining/source_coverage 등).
    출력: StrategyBrief — summary_counts·meta 등 ■ 필드는 코드 결정론으로,
        BriefCard.body·summary_line 등 ◇ 필드만 LLM이 채운다.
    불변식: 트랙 분류·우선순위(priority) 산출은 규칙 기반이며, LLM은 서술형
        슬롯(body, summary_line, culture_fit)만 채운다.
    """
    ...


# 수집 (Phase 3)
def fetch_jd_body(url_or_text: str) -> SourceDocument:
    """JD 원문을 URL 또는 붙여넣은 텍스트로부터 SourceDocument로 정규화한다.

    입력: JD URL 또는 JD 본문 텍스트.
    출력: SourceDocument(source_type=SourceType.JD). Phase 3(T16)에서 3층
        폴백(정적 파싱 → 렌더링 → 수동 입력 유도)으로 구현된다.
    불변식: collected_at·confidence 등 코드 결정 필드는 이 함수가 채운다.
    """
    ...


def fetch_tech_blog(company: str) -> list[SourceDocument]:
    """회사의 기술 블로그 글을 수집해 SourceDocument 목록으로 반환한다.

    입력: 회사명.
    출력: SourceDocument 목록(source_type=SourceType.TECH_BLOG).
    """
    ...


def get_company_values(company: str) -> list[SourceDocument]:
    """회사의 인재상·핵심가치 페이지를 수집해 SourceDocument 목록으로 반환한다.

    입력: 회사명.
    출력: SourceDocument 목록(source_type=SourceType.VALUES).
    """
    ...


def discover_jobs(company: str, role: str) -> list[SourceDocument]:
    """회사·직무 조건에 맞는 채용 공고 후보를 열거한다.

    입력: 회사명, 직무명.
    출력: 후보 공고 SourceDocument 목록 — H1(공고 선택 HITL)의 입력이 된다.
    """
    ...


def parse_resume(file_path: str) -> tuple[str, Confidence]:
    """이력서 파일을 텍스트로 파싱하고 추출 신뢰도를 함께 반환한다.

    입력: 이력서 파일 경로.
    출력: (추출된 텍스트, Confidence) 튜플 — 텍스트 레이어 우선(T21), OCR
        폴백(T22) 경로를 타면 신뢰도가 낮아진다.
    """
    ...


# 검색 (Phase 2)
def retrieve_candidates(
    required: list[CompetencyRecord],
    owned: list[CompetencyRecord],
    top_k: int = 3,
) -> list[tuple[str, str]]:
    """요구 역량과 보유 역량을 임베딩 유사도로 매칭해 후보쌍만 선별한다.

    입력: required(요구 역량) 목록, owned(보유 역량) 목록, 요구 역량 1건당
        반환할 최대 후보 수 top_k.
    출력: (요구 CompetencyRecord.comp_id, 보유 CompetencyRecord.comp_id) 후보쌍 목록.
    불변식: 유사도는 후보를 추리는 데만 쓰이며, 최종 충족 여부 판정에는
        사용하지 않는다(가짜 정밀도 방지, 설계도 §7-2).
    """
    ...

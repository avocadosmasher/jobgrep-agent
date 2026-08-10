"""역량 단위 매칭 집계 — 순수 규칙 함수 (설계도 §7-2 3단계).

이 모듈은 LLM을 호출하지 않는다. 최종 등급이 모델이 뱉은 숫자가 아니라
감사 가능한 소판정들의 산술이라는 점이 멱등성 주장(§11-1)의 근거다.
"""

from contracts.enums import MatchState, VerdictState
from contracts.models import Criterion, CriterionVerdict

# 경계값 — 설계도 §16 S6 미확정 (P2~P4 실측 후 튜닝 대상)이므로 상수로 분리한다.
MET_THRESHOLD = 1.0        # 이 비율 이상이면 MET (= 필수 기준 전부 충족)
ADJACENT_THRESHOLD = 0.5   # 이 비율 이상이면 ADJACENT (= 필수 기준 절반 이상)

# 판정 상태별 가중치. UNKNOWN은 값이 없다 — 분모에서 제외되기 때문.
VERDICT_WEIGHTS: dict[VerdictState, float] = {
    VerdictState.MET: 1.0,
    VerdictState.PARTIAL: 0.5,
    VerdictState.UNMET: 0.0,
}


def aggregate_states(
    criteria: list[Criterion],
    verdicts: list[CriterionVerdict],
) -> MatchState:
    """한 역량에 속한 기준·판정을 결정론적 규칙으로 집계한다.

    입력: 같은 comp_id를 공유하는 Criterion 목록과 그에 대응하는 CriterionVerdict 목록.
    출력: MatchState — 필수 기준 전부 충족 시 MET, 필수 기준 절반 이상 충족 시
        ADJACENT, 그 외 UNMET (설계도 §7-2).
    불변식: LLM 호출 절대 금지 — 이 함수의 결정론성이 전체 파이프라인 멱등성의 근거다.

    계산 규칙:
        - `is_required=False`(우대) 기준은 집계에서 제외한다. 등급은 필수 기준만으로 정한다.
        - 충족=1.0, 부분=0.5, 미충족=0.0 으로 환산해 평균을 낸다.
        - 판단보류(UNKNOWN)와 대응 판정이 없는 기준은 **분모에서 제외**한다.
          verify_criteria가 판정 불가 기준을 Question으로 승격시키므로(계약 참조),
          판정 목록에 없는 기준은 UNKNOWN과 같게 취급한다.
        - 평균 >= MET_THRESHOLD → MET, >= ADJACENT_THRESHOLD → ADJACENT, 그 외 UNMET.
        - `criteria`에 없는 criterion_id의 판정은 무시한다.

    판정 불가(분모 0) 처리 — 필수 기준이 하나도 없거나 전부 UNKNOWN인 경우:
        MatchState에는 "판정 불가"에 해당하는 값이 없으므로 **UNMET을 반환한다.**
        단 이는 "미보유를 확인했다"가 아니라 "충족을 확인하지 못했다"는 뜻이며,
        근거 부재는 라벨링이 아니라 카드 미생성으로 처리해야 한다(설계도 §11-2 ②).
        따라서 호출부는 이 함수의 반환값만으로 카드를 만들지 말고, 판정된 기준이
        하나라도 있는지를 별도로 확인해야 한다.

    부작용 없음 — 입력을 변형하지 않으며 IO·전역 상태를 건드리지 않는다.
    """
    required_ids = [c.criterion_id for c in criteria if c.is_required]
    states = {v.criterion_id: v.state for v in verdicts}

    scored = [
        VERDICT_WEIGHTS[states[cid]]
        for cid in required_ids
        if cid in states and states[cid] is not VerdictState.UNKNOWN
    ]

    if not scored:
        return MatchState.UNMET

    ratio = sum(scored) / len(scored)

    if ratio >= MET_THRESHOLD:
        return MatchState.MET
    if ratio >= ADJACENT_THRESHOLD:
        return MatchState.ADJACENT
    return MatchState.UNMET

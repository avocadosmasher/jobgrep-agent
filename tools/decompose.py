"""역량 → 체크 가능한 기준 분해 — 역량 N개를 **1콜**로 (설계도 §7-2 2단계, §8-4).

큰 판단 하나("쿠버네티스 활용능력 상")를 감사 가능한 소판단 여럿으로 바꾸는 단계다.
모델은 기준 문장만 만들고, criterion_id·is_required는 코드가 채운다 — is_required는
이미 `CompetencyRecord.importance`에 있으므로 모델에게 다시 물을 이유가 없다(§7-1).
"""

from __future__ import annotations

from pydantic import BaseModel

from contracts.enums import Importance
from contracts.models import CompetencyRecord, Criterion
from llm.client import DEFAULT_INSTRUCTIONS, complete_structured

MIN_CRITERIA = 3
MAX_CRITERIA = 5


class DecomposedCompetency(BaseModel):
    comp_id: str
    criteria: list[str]


class DecompositionResult(BaseModel):
    competencies: list[DecomposedCompetency]


def build_decomposition_prompt(comps: list[CompetencyRecord]) -> str:
    """역량 전체를 한 프롬프트에 묶는다 — 역량별 개별 호출 금지 (§8-4)."""
    lines = []
    for comp in comps:
        level = comp.level.value if comp.level else "명시 없음"
        lines.append(
            f"- comp_id={comp.comp_id} | 분류={comp.category.value} | "
            f"중요도={comp.importance.value} | 요구레벨={level}\n"
            f"  역량명: {comp.name}"
        )
    listing = "\n".join(lines)

    return f"""아래 역량 목록을 각각 **예/아니오로 판정 가능한 기준** {MIN_CRITERIA}~{MAX_CRITERIA}개로 분해해라.

규칙
1. 기준은 한 문장이고, 하나의 사실만 묻는다. "그리고"로 두 가지를 묶지 않는다.
   좋은 예: "프로덕션 환경에서 Kubernetes 클러스터를 직접 운영한 경험이 있다"
   나쁜 예: "쿠버네티스를 잘 다루고 장애 대응도 할 수 있다" (판정 불가·복합)
2. 서로 다른 축을 덮는다. 예: 운영 경험 / 장애 대응 / 산출물 작성 / 운영 규모.
   같은 말을 바꿔 쓴 중복 기준을 만들지 않는다.
3. 역량이 요구하는 수준(요구레벨)을 기준 문장에 반영한다.
4. comp_id는 입력에 있는 값을 그대로 돌려준다. 목록에 없는 comp_id를 만들지 않는다.
5. 기준은 {MAX_CRITERIA}개를 넘기지 않는다.

{DEFAULT_INSTRUCTIONS}

<competencies>
{listing}
</competencies>"""


def to_criteria(
    comps: list[CompetencyRecord], decomposed: list[DecomposedCompetency]
) -> dict[str, list[Criterion]]:
    """모델 산출을 계약 모델로 변환한다.

    코드가 채우는 것:
        - criterion_id — `cr-{comp_id}-{순번}` 결정론적 생성
        - is_required — `importance == 필수` 에서 파생 (모델에게 묻지 않는다)
    정리 규칙:
        - 입력에 없는 comp_id는 버린다.
        - 같은 역량 안의 중복 문장은 제거하고, {MAX_CRITERIA}개를 넘으면 잘라낸다.
        - **입력 역량은 전부 키로 남는다.** 모델이 빠뜨린 역량은 빈 리스트가 되며,
          호출부는 이를 "분해 실패"로 감지할 수 있다.
    """
    by_id = {comp.comp_id: comp for comp in comps}
    texts: dict[str, list[str]] = {comp.comp_id: [] for comp in comps}

    for item in decomposed:
        if item.comp_id not in texts:
            continue
        bucket = texts[item.comp_id]
        for text in item.criteria:
            cleaned = " ".join(text.split())
            if cleaned and cleaned not in bucket and len(bucket) < MAX_CRITERIA:
                bucket.append(cleaned)

    return {
        comp_id: [
            Criterion(
                criterion_id=f"cr-{comp_id}-{i:02d}",
                comp_id=comp_id,
                text=text,
                is_required=by_id[comp_id].importance is Importance.REQUIRED,
            )
            for i, text in enumerate(bucket, start=1)
        ]
        for comp_id, bucket in texts.items()
    }


def decompose_criteria(comps: list[CompetencyRecord]) -> dict[str, list[Criterion]]:
    """역량 하나를 체크 가능한 기준 3~5개로 배치 분해한다.

    입력: CompetencyRecord 목록.
    출력: comp_id → Criterion 목록 매핑. 각 Criterion.text는 예/아니오로
        판정 가능한 단일 문장이어야 한다.
    불변식: 역량별 개별 호출 금지 — 역량이 몇 개든 `complete_structured` 호출은 1회다.
        입력의 모든 comp_id가 반환 딕셔너리의 키로 존재한다.
    """
    if not comps:
        return {}

    result = complete_structured(build_decomposition_prompt(comps), DecompositionResult)
    return to_criteria(comps, result.competencies)

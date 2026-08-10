"""요구/보유 역량 추출 — 문서 N건을 **1콜**로 처리한다 (설계도 §8-4).

코드와 모델의 분담 (설계도 §7-1):
    모델이 채우는 것 — category / name / importance / level / quote
    코드가 채우는 것 — comp_id / evidence.source_name / url / collected_at
정제 책임을 모델에 넘기지 않아야 모델이 바뀌어도 결과가 흔들리지 않는다.
"""

from __future__ import annotations

from pydantic import BaseModel

from contracts.enums import Category, Importance, Level
from contracts.models import CompetencyRecord, Evidence, SourceDocument
from llm.client import DEFAULT_INSTRUCTIONS, complete_structured

# 본문 구분자 — 인젝션 격리 (설계도 §12-5 규칙 2)
DOC_OPEN = "<document id={doc_id!r} source={source!r} company={company!r}>"
DOC_CLOSE = "</document>"


class ExtractedCompetency(BaseModel):
    """LLM이 채우는 슬롯만 담는 중간 모델. 그대로 계약 모델이 되지는 않는다."""

    doc_id: str
    category: Category
    name: str
    importance: Importance
    level: Level | None
    quote: str


class ExtractionResult(BaseModel):
    competencies: list[ExtractedCompetency]


def _normalized_with_map(text: str) -> tuple[str, list[int]]:
    """연속 공백을 하나로 접은 문자열과, 각 문자의 원문 인덱스를 함께 돌려준다."""
    chars: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            chars.append(" ")
        else:
            chars.append(ch)
        origin.append(i)
    return "".join(chars), origin


def locate_verbatim(raw_text: str, needle: str) -> str | None:
    """`needle`에 대응하는 **원문 그대로의 부분 문자열**을 찾아 반환한다.

    줄바꿈·공백 차이는 무시하고 대조하되, 반환값은 원문에서 잘라낸 조각이므로
    `반환값 in raw_text`가 항상 참이다. 찾지 못하면 None (→ 호출부에서 드롭).
    """
    target = " ".join(needle.split())
    if not target:
        return None

    haystack, origin = _normalized_with_map(raw_text)
    pos = haystack.find(target)
    if pos < 0:
        return None
    return raw_text[origin[pos] : origin[pos + len(target) - 1] + 1]


def build_extraction_prompt(docs: list[SourceDocument], role: str) -> str:
    """문서 전체를 하나의 프롬프트로 묶는다 — 문서별 개별 호출 금지 (§8-4)."""
    categories = "\n".join(f"  - {c.value}" for c in Category)
    levels = "\n".join(f"  - {lv.value}" for lv in Level)
    importances = " | ".join(i.value for i in Importance)

    blocks = []
    for doc in docs:
        head = DOC_OPEN.format(
            doc_id=doc.doc_id, source=doc.source_type.value, company=doc.company
        )
        blocks.append(f"{head}\n{doc.raw_text}\n{DOC_CLOSE}")
    documents = "\n\n".join(blocks)

    return f"""대상 직무: {role}

아래 <document> 블록들에서 이 직무가 **요구하는 역량**을 모두 추출해라.

규칙
1. 역량명(name)은 문서에 적힌 **원문 표현 그대로** 옮긴다. 일반화·요약·병합하지 않는다.
   ("Kubernetes 클러스터 운영 경험"을 "컨테이너 관리 능력"으로 바꾸지 말 것)
2. quote는 그 역량의 근거가 되는 **문서 원문 문장을 그대로** 인용한다. 지어내지 않는다.
   원문에 없는 인용은 코드 대조에서 걸러져 항목 전체가 버려진다.
3. doc_id는 그 역량이 나온 <document>의 id를 그대로 적는다.
4. category는 아래 목록에서만 고른다:
{categories}
5. importance는 {importances} 중 하나다. 문서가 "우대/선호"로 표기한 항목만 우대다.
6. level은 문서가 요구하는 숙련도이며 아래 중 하나, 판단할 수 없으면 null이다:
{levels}
7. 문서에 없는 역량을 추가하지 않는다. 누락보다 날조가 더 나쁘다.

{DEFAULT_INSTRUCTIONS}

{documents}"""


def to_records(
    docs: list[SourceDocument], extracted: list[ExtractedCompetency]
) -> list[CompetencyRecord]:
    """모델 산출을 계약 모델로 변환한다. 원문 대조 실패 항목은 **드롭**한다.

    드롭 조건 (설계도 §12-5 규칙 4, T04 불변식):
        - doc_id가 입력 문서에 없음
        - quote가 해당 문서 원문에 없음
        - name이 해당 문서 원문에 없음 (= 모델이 일반화·요약했다는 신호)
    같은 문서에서 같은 역량명이 중복되면 첫 항목만 남긴다.
    """
    by_id = {doc.doc_id: doc for doc in docs}
    records: list[CompetencyRecord] = []
    seen: set[tuple[str, str]] = set()
    seq: dict[str, int] = {}

    for item in extracted:
        doc = by_id.get(item.doc_id)
        if doc is None:
            continue

        quote = locate_verbatim(doc.raw_text, item.quote)
        name = locate_verbatim(doc.raw_text, item.name)
        if quote is None or name is None:
            continue

        key = (doc.doc_id, name)
        if key in seen:
            continue
        seen.add(key)

        seq[doc.doc_id] = seq.get(doc.doc_id, 0) + 1
        records.append(
            CompetencyRecord(
                comp_id=f"req-{doc.doc_id}-{seq[doc.doc_id]:02d}",
                category=item.category,
                name=name,
                importance=item.importance,
                level=item.level,
                evidence=[
                    Evidence(
                        source_name=doc.title,
                        url=doc.url,
                        quote=quote,
                        collected_at=doc.collected_at,
                    )
                ],
            )
        )

    return records


def extract_competencies(
    docs: list[SourceDocument], role: str
) -> list[CompetencyRecord]:
    """수집 문서 묶음에서 요구/보유 역량을 배치 1회 호출로 추출한다.

    입력: SourceDocument 목록(회사 JD·기술블로그·인재상 등), 대상 직무명.
    출력: CompetencyRecord 목록 — 각 레코드는 evidence를 최소 1건 포함한다.
    불변식: 문서별·역량별 개별 호출 금지 — 문서가 몇 건이든 `complete_structured`
        호출은 정확히 1회다. 역량명은 원문 표현 그대로 보존하며, evidence.quote가
        원문에 실제로 존재하는지 코드로 대조해 불일치 항목은 버린다.
    """
    if not docs:
        return []

    result = complete_structured(build_extraction_prompt(docs, role), ExtractionResult)
    return to_records(docs, result.competencies)

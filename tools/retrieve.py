"""후보쌍 검색 — 매칭 1단계 (설계도 §7-2 1단계, §14).

요구 역량 × 보유 역량 **전수 비교를 피하는** 부품이다. 역량명은 원문 표현을
보존하므로(T04 불변식) 문자열 매칭이 불가능하고, 그래서 임베딩이 필요하다.
"RAG를 위한 RAG"가 아니라 이 단계에 원래 필요했던 것이며 Faiss는 그 구현체다.

**가장 중요한 불변식 — 유사도는 후보를 추리는 데만 쓴다. 판정에는 절대 쓰지 않는다.**
유사도 점수를 판정 근거로 쓰면 가짜 정밀도가 생긴다. 판정은 T05의 기준 분해 +
근거 인용으로만 한다. 이 모듈이 **점수를 반환하지 않는 것**이 그 불변식의 구조적
보장이다 — 돌려주지 않으므로 하류가 쓸 수단이 없다.

인덱스는 **세션 내 메모리**이며 영속 캐시를 두지 않는다. 회사당 콘텐츠 규모가
작아 캐시 이득이 없고, 오래된 인덱스가 조용히 쓰이는 위험만 남는다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import faiss
import numpy as np

from contracts.models import CompetencyRecord
from llm.embeddings import embed_texts

# 임베딩은 `llm/` 어댑터를 통해서만 부른다(R6). 이 모듈은 OpenAI SDK를 모른다.
Embedder = Callable[[list[str]], list[list[float]]]


def _unit_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """행 단위로 L2 정규화한 float32 행렬을 만든다.

    정규화해 두면 내적이 곧 코사인 유사도가 되므로 `IndexFlatIP` 하나로 끝난다.
    영벡터는 0으로 나누지 않도록 그대로 둔다 — 유사도가 전부 0이 되어 후보에서
    자연히 밀릴 뿐, 예외로 파이프라인을 세우지는 않는다.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


def retrieve_candidates(
    required: list[CompetencyRecord],
    owned: list[CompetencyRecord],
    top_k: int = 3,
    *,
    embed: Embedder | None = None,
) -> list[tuple[str, str]]:
    """요구 역량과 보유 역량을 임베딩 유사도로 매칭해 후보쌍만 선별한다.

    입력: required(요구 역량) 목록, owned(보유 역량) 목록, 요구 역량 1건당
        반환할 최대 후보 수 top_k. embed는 임베딩 함수를 갈아끼우기 위한
        주입점이며 평소엔 None (기본값 `llm.embeddings.embed_texts`).
    출력: (요구 comp_id, 보유 comp_id) 후보쌍 목록. required 순서를 따르고,
        같은 요구 역량 안에서는 **유사도 내림차순**이다. 유사도 값 자체는
        반환하지 않는다 — 위 불변식의 구조적 보장이다.
    불변식: 유사도는 후보 추림에만 쓰이며 최종 충족 여부 판정에는 사용하지 않는다
        (가짜 정밀도 방지, 설계도 §7-2). 임베딩 API 왕복은 입력 규모와 무관하게
        **정확히 1회**다 — 요구·보유를 한 번에 묶어 보낸다.

    빈 결과 조건: required 또는 owned가 비었거나 top_k <= 0이면 호출 없이 `[]`.
    부작용 없음 — 입력을 변형하지 않으며 인덱스는 호출마다 새로 만들고 버린다.
    """
    if not required or not owned or top_k <= 0:
        return []

    embed_fn = embed or embed_texts

    # 요구·보유를 한 번에 보내고 잘라 쓴다 (왕복 1회).
    split = len(required)
    vectors = embed_fn([c.name for c in required] + [c.name for c in owned])
    if len(vectors) != split + len(owned):
        raise ValueError(
            f"임베딩 개수가 입력과 다르다: {len(vectors)} != {split + len(owned)}"
        )

    matrix = _unit_matrix(vectors)
    query, base = matrix[:split], matrix[split:]

    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)

    k = min(top_k, len(owned))
    _, neighbors = index.search(query, k)

    pairs: list[tuple[str, str]] = []
    for row, req in zip(neighbors, required):
        for pos in row:
            # faiss는 이웃이 모자라면 -1을 채운다. k를 len(owned)로 묶어 뒀으므로
            # 정상 경로에서는 나오지 않지만, 나오면 조용히 건너뛴다.
            if pos < 0:
                continue
            pairs.append((req.comp_id, owned[int(pos)].comp_id))

    return pairs

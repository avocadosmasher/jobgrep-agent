"""임베딩 어댑터 — 텍스트 묶음을 벡터로 바꾸는 **유일한** 지점 (R6).

`llm/client.py`가 생성 모델 호출을 감싸듯, 이 모듈은 임베딩 호출을 감싼다.
`tools/retrieve.py`는 `embed_texts`만 쓰고 OpenAI SDK를 직접 import 하지 않는다.

**왜 `client.py`에 넣지 않고 파일을 나눴나** — `llm/client.py`는 T04 소유라
수정하면 R2에 걸리고, `tools/retrieve.py`에 SDK를 직접 넣으면 R6에 걸린다(DEVLOG D37).
어느 카드의 소유 파일도 아닌 **새 파일**은 양쪽 다 어기지 않는다 — 기존 파일의
거동이 한 줄도 바뀌지 않으므로 다른 카드의 테스트에 영향이 없다(D21·D31·D35와 같은 판단).

환경 변수
    OPENAI_API_KEY      : 필수. 없으면 `LLMConfigError` (`client.py`와 공유).
    JOBPREP_EMBED_MODEL : 선택. 기본 모델 재정의 (기본값 `DEFAULT_EMBED_MODEL`).

모델 선정 근거는 DEVLOG D37에 있다 — 요약하면 픽스처 정답쌍으로는 후보가 변별되지
않았고, 한글↔영문·약어가 섞이는 **별칭 표기**에서 `3-large`만 Recall@1 100%였다.
"""

from __future__ import annotations

import os

# `_build_client`는 같은 `llm/` 패키지 안의 내부 함수다. 복제하지 않고 가져다 쓴다 —
# 두 벌로 두면 클라이언트 생성 방식(base_url·타임아웃 등)이 바뀔 때 한쪽만 고쳐져
# 조용히 어긋난다. 읽기만 하므로 `client.py`는 수정되지 않는다(R2 유지).
from llm.client import LLMError, _build_client, _ensure_env

# D37 실측 선정. 3-large는 Matryoshka 학습이라 3072 → 1536 절단으로도 품질이
# 떨어지지 않았고, 그러면 인덱스 메모리는 3-small과 완전히 같아진다.
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_DIMENSIONS = 1536


class EmbeddingError(LLMError):
    """임베딩 호출이 실패했거나 응답이 요청과 짝이 맞지 않음."""


def default_embed_model() -> str:
    _ensure_env()
    return os.environ.get("JOBPREP_EMBED_MODEL") or DEFAULT_EMBED_MODEL


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    dimensions: int = DEFAULT_DIMENSIONS,
    client=None,
) -> list[list[float]]:
    """텍스트 묶음을 **배치 1회 호출**로 임베딩한다.

    입력: 임베딩할 문자열 목록. 빈 목록이면 호출 없이 `[]`를 반환한다.
        dimensions는 응답 벡터의 차원 (3-large 계열은 절단을 지원한다).
        client는 테스트에서 SDK를 대체하기 위한 주입점이며 평소엔 None.
    출력: `texts`와 **같은 순서·같은 길이**의 벡터 목록.
    실패: API 오류, 또는 응답 개수·차원이 요청과 맞지 않으면 `EmbeddingError`.
        조용히 짧은 목록을 돌려주지 않는다 — 호출부가 인덱스로 짝을 맞추므로
        길이가 어긋나면 엉뚱한 역량끼리 연결된다.

    불변식: 텍스트별 개별 호출 금지. 몇 건이든 API 왕복은 정확히 1회다(§8-4와 같은 취지).
    """
    if not texts:
        return []

    if client is None:
        client = _build_client()

    try:
        response = client.embeddings.create(
            model=model or default_embed_model(),
            input=texts,
            dimensions=dimensions,
        )
    except Exception as exc:  # SDK·네트워크·쿼터 오류
        raise EmbeddingError(f"임베딩 호출이 실패했다: {exc!r}") from exc

    # 응답 순서를 신뢰하지 않고 index로 되맞춘다. SDK가 순서를 보장하더라도
    # 여기서 어긋나면 증상이 "엉뚱한 후보쌍"으로만 나타나 추적이 매우 어렵다.
    items = sorted(response.data, key=lambda d: d.index)
    if len(items) != len(texts):
        raise EmbeddingError(
            f"요청 {len(texts)}건에 응답 {len(items)}건이 왔다 — 짝을 맞출 수 없다."
        )

    vectors = [list(item.embedding) for item in items]
    if any(len(v) != dimensions for v in vectors):
        got = {len(v) for v in vectors}
        raise EmbeddingError(f"차원이 {dimensions}이 아니다 (실제 {sorted(got)}).")

    return vectors

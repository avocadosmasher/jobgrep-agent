"""T14 · 임베딩 어댑터 + retrieve_candidates 검증.

오프라인(기본): 후보 선별 로직·배치 불변식·점수 미반환·어댑터 오류 처리.
온라인(`-m llm`): 실제 임베딩으로 픽스처의 명백한 대응쌍이 top-3에 드는지 (카드 완료 조건).

R5에 따라 입력은 `fixtures/competencies_required.json`·`fixtures/profile_sample.json`
골든 데이터를 그대로 쓴다. 오프라인에서 임베딩을 대신하는 `lexical_embedder`는
**구현을 흉내 낸 것이 아니다** — 이 모듈이 검증하는 건 faiss 순위 산출이고,
스텁은 그것과 무관한 문자 바이그램 겹침으로 벡터를 만든다. 스텁이 순위를 정해주면
테스트가 무의미해지므로 그렇게 하지 않았다.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from contracts.models import CompetencyRecord, ProfileJSON
from llm.embeddings import DEFAULT_DIMENSIONS, EmbeddingError, embed_texts
from tools.retrieve import retrieve_candidates

FIXTURES = Path(__file__).parent.parent / "fixtures"

REQUIRED: list[CompetencyRecord] = TypeAdapter(list[CompetencyRecord]).validate_json(
    (FIXTURES / "competencies_required.json").read_bytes()
)
PROFILE = ProfileJSON.model_validate_json((FIXTURES / "profile_sample.json").read_bytes())
OWNED = PROFILE.competencies

# 사람이 픽스처를 눈으로 확인한 대응쌍 (DEVLOG D37 실험 1과 같은 목록).
GOLD_PAIRS = {
    "req-be-01": "pf-01",   # Python/Java 프로덕션 3년   ↔ Python 프로덕션 3년
    "req-be-02": "pf-03",   # RESTful/gRPC API 설계      ↔ FastAPI RESTful API 설계
    "req-be-03": "pf-04",   # RDB 스키마·쿼리 튜닝        ↔ PostgreSQL 쿼리 튜닝·인덱스
    "req-be-05": "pf-07",   # Docker 이미지 빌드·배포     ↔ Docker 이미지 빌드·배포
    "req-be-06": "pf-08",   # Kubernetes 클러스터 운영    ↔ Kubernetes 클러스터 운영
    "req-be-07": "pf-05",   # AWS 환경 운영               ↔ AWS EC2·RDS 운영
    "req-be-09": "pf-02",   # 단위 테스트·커버리지 관리    ↔ pytest 커버리지 80%
    "req-be-11": "pf-06",   # Terraform IaC 프로비저닝    ↔ Terraform 프로비저닝 자동화
    "req-ai-08": "pf-07",   # Docker 컨테이너 환경 구축   ↔ Docker 이미지 빌드·배포
}

# 표기가 사실상 같아 어떤 임베더로도 1위여야 하는 쌍 — 오프라인에서 쓴다.
NEAR_IDENTICAL = {"req-be-05": "pf-07", "req-be-06": "pf-08"}


# --- 오프라인용 임베더 (문자 바이그램 겹침) ---------------------------------


def _bigrams(text: str) -> list[str]:
    squeezed = "".join(text.split())
    return [squeezed[i : i + 2] for i in range(max(len(squeezed) - 1, 0))]


def lexical_embedder(texts: list[str]) -> list[list[float]]:
    """어휘 겹침만 보는 결정론적 임베더. API를 타지 않는다."""
    vocab = sorted({bg for t in texts for bg in _bigrams(t)})
    position = {bg: i for i, bg in enumerate(vocab)}
    vectors = []
    for t in texts:
        vec = [0.0] * len(vocab)
        for bg in _bigrams(t):
            vec[position[bg]] += 1.0
        vectors.append(vec)
    return vectors


class CountingEmbedder:
    """호출 횟수를 세는 래퍼 — 배치 불변식 검증용."""

    def __init__(self, inner=lexical_embedder):
        self.inner = inner
        self.calls = 0
        self.last_batch: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.last_batch = list(texts)
        return self.inner(texts)


def pairs_by_required(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for req_id, own_id in pairs:
        grouped.setdefault(req_id, []).append(own_id)
    return grouped


# --- 후보 선별 로직 ---------------------------------------------------------


def test_near_identical_names_rank_first():
    """표기가 같은 대응쌍은 1위여야 한다 — 순위가 실제로 산출된다는 최소 증거."""
    grouped = pairs_by_required(
        retrieve_candidates(REQUIRED, OWNED, top_k=3, embed=lexical_embedder)
    )
    for req_id, own_id in NEAR_IDENTICAL.items():
        assert grouped[req_id][0] == own_id, (
            f"{req_id}의 1순위가 {own_id}가 아니라 {grouped[req_id][0]}이다"
        )


def test_top_k_limits_candidates_per_required():
    for k in (1, 2, 3):
        pairs = retrieve_candidates(REQUIRED, OWNED, top_k=k, embed=lexical_embedder)
        grouped = pairs_by_required(pairs)
        assert len(pairs) == len(REQUIRED) * k
        assert all(len(v) == k for v in grouped.values())


def test_top_k_is_capped_by_owned_count():
    """top_k가 보유 역량 수보다 커도 있는 만큼만 돌려준다.

    **결과만 고정하고 수단은 고정하지 않는다.** `k = min(top_k, len(owned))` 캡과
    `pos < 0` 건너뛰기 둘 중 아무거나 하나만 있어도 통과한다 — 뮤테이션으로 확인했다
    (DEVLOG D39). 둘 다 두는 건 faiss의 패딩 거동에 기대지 않기 위한 이중 방어이며,
    이 테스트는 그 중 무엇이 일했는지가 아니라 밖으로 -1이 새지 않음을 본다.
    """
    pairs = retrieve_candidates(REQUIRED, OWNED, top_k=99, embed=lexical_embedder)
    grouped = pairs_by_required(pairs)
    assert all(len(v) == len(OWNED) for v in grouped.values())
    assert all(own_id for _, own_id in pairs)


def test_required_order_is_preserved():
    pairs = retrieve_candidates(REQUIRED, OWNED, top_k=2, embed=lexical_embedder)
    seen = list(dict.fromkeys(req_id for req_id, _ in pairs))
    assert seen == [c.comp_id for c in REQUIRED]


def test_candidates_are_unique_per_required():
    grouped = pairs_by_required(
        retrieve_candidates(REQUIRED, OWNED, top_k=3, embed=lexical_embedder)
    )
    for req_id, own_ids in grouped.items():
        assert len(own_ids) == len(set(own_ids)), f"{req_id}에 같은 후보가 중복됐다"


def test_similarity_scores_are_never_returned():
    """§7-2 불변식의 구조적 보장 — 점수를 안 돌려주므로 하류가 판정에 쓸 수단이 없다."""
    pairs = retrieve_candidates(REQUIRED, OWNED, top_k=3, embed=lexical_embedder)
    for pair in pairs:
        assert isinstance(pair, tuple) and len(pair) == 2
        assert all(isinstance(x, str) for x in pair)


def test_ranking_is_cosine_not_raw_dot_product():
    """벡터 크기가 순위를 흔들면 안 된다 — 방향이 같은 짧은 벡터가 이겨야 한다.

    `IndexFlatIP`는 내적이라 정규화를 빠뜨리면 **긴 텍스트가 그냥 유리해진다.**
    실제 OpenAI 임베딩은 이미 단위벡터로 와서 이 결함이 드러나지 않으므로,
    정규화되지 않은 벡터를 주는 임베더로만 잡을 수 있다.
    """
    req = CompetencyRecord(
        comp_id="q", category=REQUIRED[0].category, name="q",
        importance=REQUIRED[0].importance,
    )
    aligned = CompetencyRecord(   # 방향 일치(cos=1.0), 크기 작음(내적 0.5)
        comp_id="aligned", category=REQUIRED[0].category, name="aligned",
        importance=REQUIRED[0].importance,
    )
    bulky = CompetencyRecord(     # 방향 어긋남(cos≈0.71), 크기 큼(내적 10.0)
        comp_id="bulky", category=REQUIRED[0].category, name="bulky",
        importance=REQUIRED[0].importance,
    )
    vectors = {"q": [1.0, 0.0], "aligned": [0.5, 0.0], "bulky": [10.0, 10.0]}

    pairs = retrieve_candidates(
        [req], [aligned, bulky], top_k=1, embed=lambda texts: [vectors[t] for t in texts]
    )
    assert pairs == [("q", "aligned")], "크기가 큰 쪽이 이겼다면 정규화가 빠진 것이다"


def test_inputs_are_not_mutated():
    before = [(c.comp_id, c.name) for c in REQUIRED + OWNED]
    retrieve_candidates(REQUIRED, OWNED, top_k=3, embed=lexical_embedder)
    assert [(c.comp_id, c.name) for c in REQUIRED + OWNED] == before


# --- 배치 불변식 ------------------------------------------------------------


def test_embedding_is_called_exactly_once():
    """요구 26 × 보유 8이어도 API 왕복은 1회다 (§8-4와 같은 취지)."""
    counter = CountingEmbedder()
    retrieve_candidates(REQUIRED, OWNED, top_k=3, embed=counter)
    assert counter.calls == 1
    assert len(counter.last_batch) == len(REQUIRED) + len(OWNED)


@pytest.mark.parametrize(
    "required, owned, top_k",
    [([], OWNED, 3), (REQUIRED, [], 3), (REQUIRED, OWNED, 0), (REQUIRED, OWNED, -1)],
)
def test_empty_result_skips_the_api_entirely(required, owned, top_k):
    counter = CountingEmbedder()
    assert retrieve_candidates(required, owned, top_k=top_k, embed=counter) == []
    assert counter.calls == 0


def test_embedder_returning_wrong_count_raises():
    def short(texts):
        return lexical_embedder(texts)[:-1]

    with pytest.raises(ValueError):
        retrieve_candidates(REQUIRED, OWNED, top_k=3, embed=short)


# --- 임베딩 어댑터 (llm/embeddings.py) --------------------------------------


class FakeEmbeddingResponse:
    def __init__(self, items):
        self.data = items


class FakeItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class FakeEmbeddingClient:
    """SDK 대역 — 넘어온 kwargs를 그대로 보관해 검사할 수 있게 한다."""

    def __init__(self, items_for=None):
        self.kwargs = None
        self.calls = 0
        self.items_for = items_for or (
            lambda texts: [FakeItem(i, [0.0] * DEFAULT_DIMENSIONS) for i in range(len(texts))]
        )
        self.embeddings = self

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return FakeEmbeddingResponse(self.items_for(kwargs["input"]))


def test_adapter_sends_one_batch_with_selected_model_and_dimensions():
    client = FakeEmbeddingClient()
    embed_texts(["a", "b", "c"], client=client)

    assert client.calls == 1
    assert client.kwargs["input"] == ["a", "b", "c"]
    assert client.kwargs["dimensions"] == DEFAULT_DIMENSIONS
    assert client.kwargs["model"] == "text-embedding-3-large"  # D37 선정


def test_adapter_reorders_response_by_index():
    """SDK가 순서를 바꿔 주더라도 index로 되맞춘다 — 어긋나면 엉뚱한 쌍이 나온다."""

    def shuffled(texts):
        items = [FakeItem(i, [float(i)] * DEFAULT_DIMENSIONS) for i in range(len(texts))]
        return list(reversed(items))

    vectors = embed_texts(["a", "b", "c"], client=FakeEmbeddingClient(shuffled))
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]


def test_adapter_raises_when_response_count_mismatches():
    def dropped(texts):
        return [FakeItem(i, [0.0] * DEFAULT_DIMENSIONS) for i in range(len(texts) - 1)]

    with pytest.raises(EmbeddingError):
        embed_texts(["a", "b"], client=FakeEmbeddingClient(dropped))


def test_adapter_raises_on_unexpected_dimensions():
    def narrow(texts):
        return [FakeItem(i, [0.0] * 8) for i in range(len(texts))]

    with pytest.raises(EmbeddingError):
        embed_texts(["a"], client=FakeEmbeddingClient(narrow))


def test_adapter_wraps_sdk_errors():
    class Exploding(FakeEmbeddingClient):
        def create(self, **kwargs):
            raise RuntimeError("429 insufficient_quota")

    with pytest.raises(EmbeddingError):
        embed_texts(["a"], client=Exploding())


def test_adapter_skips_the_call_for_empty_input():
    client = FakeEmbeddingClient()
    assert embed_texts([], client=client) == []
    assert client.calls == 0


# --- 온라인: 실제 임베딩 (`-m llm`) -----------------------------------------


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY 없음")
def test_llm_obvious_pairs_land_in_top3():
    """카드 완료 조건 — 픽스처의 명백한 대응쌍이 top-3에 포함된다."""
    grouped = pairs_by_required(retrieve_candidates(REQUIRED, OWNED, top_k=3))

    misses = [
        (req_id, own_id, grouped.get(req_id, []))
        for req_id, own_id in GOLD_PAIRS.items()
        if own_id not in grouped.get(req_id, [])
    ]
    assert not misses, f"top-3에 못 든 대응쌍: {misses}"


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY 없음")
def test_llm_alias_notation_still_ranks_first():
    """한글↔영문·약어 표기에서도 1위여야 한다 — D37이 모델을 고른 근거다.

    역량명은 원문 보존이라(T04 불변식) 이 상황은 반드시 생긴다. 픽스처의
    대응쌍은 표기가 이미 일치해 이 성질을 검사하지 못하므로 별칭으로 바꿔 던진다.
    """
    aliases = {
        "쿠버네티스 클러스터 운영 경험": "pf-08",
        "도커 컨테이너 이미지 관리": "pf-07",
        "포스트그레스 성능 최적화": "pf-04",
        "테라폼 코드형 인프라": "pf-06",
    }
    probes = [
        CompetencyRecord(
            comp_id=f"alias-{i:02d}",
            category=REQUIRED[0].category,
            name=name,
            importance=REQUIRED[0].importance,
        )
        for i, name in enumerate(aliases)
    ]

    grouped = pairs_by_required(retrieve_candidates(probes, OWNED, top_k=1))
    actual = {p.name: grouped[p.comp_id][0] for p in probes}
    assert actual == aliases, f"별칭 표기 1순위가 어긋났다: {actual}"


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY 없음")
def test_llm_embedding_returns_selected_dimensions():
    vectors = embed_texts(["Kubernetes 클러스터 운영", "쿠버네티스 운영"])
    assert len(vectors) == 2
    assert all(len(v) == DEFAULT_DIMENSIONS for v in vectors)

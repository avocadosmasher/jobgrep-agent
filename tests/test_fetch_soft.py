"""T17 · 소프트 요건 수집(`fetch_tech_blog` / `get_company_values`) 검증.

전부 오프라인이다 — 네트워크를 주입점으로 갈아끼워 robots 차단·404·생예외를 재현한다.
실제 회사 사이트를 테스트에서 두드리면 결과가 남의 서버 상태에 매달리고, robots 준수를
어기면서 그걸 검증하는 꼴이 된다(T16과 같은 이유).

**이 카드의 완료 조건은 기능이 아니라 실패 처리다** — "네트워크 실패 시나리오에서 예외
없이 빈 결과 + missing 기록". 그래서 증명의 무게가 아래 세 개에 실려 있다:

- `test_total_collection_failure_is_empty_and_silent` — 전부 죽어도 예외가 안 올라온다
- `test_one_bad_candidate_does_not_kill_the_rest` — 부분 실패를 전체 실패로 만들지 않는다
- `test_missing_labels_mark_the_empty_side` — 못 가져온 사실이 라벨로 남는다

R5에 따라 본문 골든 데이터는 `fixtures/jd_sample_aiinfra.json`의 `raw_text`를 쓴다. 그
본문은 담당업무·인재상 절을 함께 갖고 있어 두 유형 모두의 표지어를 담는다 — 유형을
갈라 봐야 하는 곳(표지어 판별)에서만 좁은 본문을 따로 쓴다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from contracts.enums import Confidence, SourceType
from contracts.models import SourceDocument
from tools.fetch_jd import FetchError
from tools.fetch_soft import (
    KNOWN_SITES,
    MAX_ATTEMPTS,
    MAX_DOC_CHARS,
    MIN_SOFT_CHARS,
    CompanySite,
    candidate_urls,
    fetch_tech_blog,
    get_company_values,
    missing_soft_sources,
    resolve_site,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
JD = SourceDocument.model_validate_json((FIXTURES / "jd_sample_aiinfra.json").read_bytes())
BODY = JD.raw_text

SITE = CompanySite("example.com")
SITES = {"클라우드마인드": SITE}
COMPANY = "클라우드마인드"
ALLOW_ALL = "User-agent: *\nAllow: /\n"
# **오늘이 아닌 날짜여야 한다.** 주입한 수집일을 무시하고 `date.today()`를 쓰는 회귀는
# 하필 오늘 날짜로 고르면 그날 하루 동안 안 잡힌다(뮤테이션에서 실제로 살아남았다).
TODAY = date(2026, 6, 1)

TECH_URL = "https://tech.example.com/"
BLOG_PATH_URL = "https://example.com/blog"
CAREERS_URL = "https://careers.example.com/"
VALUES_PATH_URL = "https://example.com/culture"

# 유형 판별용 좁은 본문 — 한쪽 표지어만 담는다.
TECH_ONLY = (
    "쿠버네티스 클러스터에 오토스케일링을 적용하면서 겪은 일을 개발 팀이 정리했다.\n"
    "노드 풀을 분리하고 스케줄러 큐를 조정한 뒤 지연시간이 절반으로 줄었다.\n"
    "같은 방식을 다른 서비스에도 적용하는 중이며 수치는 매주 다시 잰다.\n"
) * 2
VALUES_ONLY = (
    "우리는 함께 성장하는 동료를 찾습니다.\n"
    "핵심가치는 신뢰와 자율입니다. 스스로 판단하고 그 결과를 설명합니다.\n"
    "인재상: 문제를 스스로 정의하고 끝까지 해내는 사람.\n"
) * 2
NEITHER = "죄송합니다. 요청하신 페이지를 찾을 수 없습니다. 홈으로 돌아가 주세요.\n" * 6


def page(body: str = BODY, *, head: str = "<title>클라우드마인드 이야기</title>") -> str:
    """본문 + 사람 눈에 안 보이는 잡음을 함께 담은 페이지."""
    paragraphs = "\n".join(f"<p>{line}</p>" for line in body.split("\n"))
    return (
        f"<!doctype html><html><head>{head}<style>.b {{ color: red; }}</style></head>"
        f"<body><nav>메뉴</nav><article>{paragraphs}</article>"
        "<script>var t = 'do-not-extract';</script></body></html>"
    )


class RecordingFetcher:
    """URL → 본문 맵. 없는 URL은 404, **모든 호출을 기록**한다.

    "안 불렀다"를 예외가 아니라 기록으로 증명한다 — 예외를 삼키는 계층이 사이에 있으면
    예외 기반 증명이 성립하지 않는다(DEVLOG D50·D54).
    """

    def __init__(self, pages: dict[str, str] | None = None, *, robots: object = ALLOW_ALL):
        self.pages = dict(pages or {})
        self.robots = robots            # str | Exception | None(=404)
        self.calls: list[str] = []

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            if isinstance(self.robots, BaseException):
                raise self.robots
            if self.robots is None:
                raise FetchError("HTTP 404", status=404)
            return str(self.robots)
        if url not in self.pages:
            raise FetchError("HTTP 404", status=404)
        return self.pages[url]

    @property
    def page_calls(self) -> list[str]:
        return [u for u in self.calls if not u.endswith("/robots.txt")]


def exploding_fetcher(url: str) -> str:
    raise RuntimeError("네트워크가 통째로 죽었다")


def collect(func, fetcher, **kwargs) -> list[SourceDocument]:
    return func(COMPANY, fetch=fetcher, today=TODAY, sites=SITES, **kwargs)


BOTH = pytest.mark.parametrize(
    "func, url",
    [(fetch_tech_blog, TECH_URL), (get_company_values, CAREERS_URL)],
    ids=["tech_blog", "values"],
)


# --- 완료 조건 · 실패는 조용히 ----------------------------------------------


@BOTH
def test_total_collection_failure_is_empty_and_silent(func, url):
    """카드 완료 조건 — 네트워크가 통째로 죽어도 예외 없이 빈 리스트다."""
    assert collect(func, exploding_fetcher) == []


@BOTH
@pytest.mark.parametrize(
    "fetcher_factory, label",
    [
        (lambda url: RecordingFetcher({}), "전부 404"),
        (lambda url: RecordingFetcher({url: page()}, robots="User-agent: *\nDisallow: /\n"), "robots 차단"),
        (lambda url: RecordingFetcher({url: "<p>짧다</p>"}), "본문 부족"),
        (lambda url: RecordingFetcher({url: page(NEITHER)}, ), "표지어 없음"),
        (lambda url: RecordingFetcher({url: page()}, robots=RuntimeError("이상한 예외")), "robots 생예외"),
    ],
)
def test_each_failure_mode_yields_an_empty_list(func, url, fetcher_factory, label):
    assert collect(func, fetcher_factory(url)) == [], label


@BOTH
def test_page_raising_after_robots_ok_is_still_silent(func, url):
    """robots는 통과했는데 **페이지에서** 생예외가 나는 경우.

    `exploding_fetcher`는 robots.txt부터 터져 `robots_allows()`의 방어에 걸리므로
    페이지 요청 경로까지 가지 않는다 — 그 구멍이 T16에서 뮤테이션 생존으로 드러났다(D54).
    """

    class RobotsOkThenBoom(RecordingFetcher):
        def __call__(self, u: str) -> str:
            if u.endswith("/robots.txt"):
                return super().__call__(u)
            self.calls.append(u)
            raise RuntimeError("HTML을 읽다가 폭발")

    fetcher = RobotsOkThenBoom()
    assert collect(func, fetcher) == []
    assert fetcher.page_calls, "페이지 요청까지는 갔어야 이 경로가 검증된다"


def test_one_bad_candidate_does_not_kill_the_rest():
    """관통 규칙 — 부분 실패를 전체 실패로 만들지 않는다.

    앞 후보 셋은 각각 다른 방식으로 죽고(404·본문 부족·표지어 없음) 뒤에 성한 게 하나
    있다. 어느 하나가 루프를 끊으면 이 테스트가 깨진다.
    """
    fetcher = RecordingFetcher(
        {
            "https://techblog.example.com/": "<p>짧다</p>",
            "https://engineering.example.com/": page(NEITHER),
            "https://blog.example.com/": page(),
        }
    )
    docs = collect(fetch_tech_blog, fetcher)

    assert [d.url for d in docs] == ["https://blog.example.com/"]
    assert "https://tech.example.com/" in fetcher.page_calls, "404 후보도 실제로 시도했다"


def test_disallowed_pages_are_never_requested():
    """차단이면 **요청 자체를 보내지 않는다** — 받아놓고 버리는 건 준수가 아니다."""
    fetcher = RecordingFetcher(
        {TECH_URL: page()}, robots="User-agent: *\nDisallow: /\n"
    )
    assert collect(fetch_tech_blog, fetcher) == []
    assert fetcher.page_calls == []


def test_an_unknown_company_never_touches_the_network():
    fetcher = RecordingFetcher({TECH_URL: page()})
    assert fetch_tech_blog("듣도보도못한회사", fetch=fetcher, today=TODAY, sites=SITES) == []
    assert fetcher.calls == []


@pytest.mark.parametrize("value", ["", "   ", None])
def test_blank_company_is_empty(value):
    fetcher = RecordingFetcher()
    assert fetch_tech_blog(value, fetch=fetcher, today=TODAY, sites=SITES) == []
    assert fetcher.calls == []


# --- missing 기록 -----------------------------------------------------------


def test_missing_labels_mark_the_empty_side():
    """카드 완료 조건의 나머지 절반 — 못 가져온 사실이 라벨로 남는다."""
    fetcher = RecordingFetcher({CAREERS_URL: page()})
    tech = collect(fetch_tech_blog, fetcher)
    values = collect(get_company_values, fetcher)

    assert tech == []
    assert values, "인재상 쪽은 성공해야 '부분'이 된다"
    assert missing_soft_sources(tech, values) == ["기술블로그"]


def test_missing_labels_list_both_sides_when_nothing_was_collected():
    assert missing_soft_sources([], []) == ["기술블로그", "인재상"]


def test_nothing_is_missing_when_both_sides_collected():
    fetcher = RecordingFetcher({TECH_URL: page(), CAREERS_URL: page()})
    tech = collect(fetch_tech_blog, fetcher)
    values = collect(get_company_values, fetcher)

    assert tech and values
    assert missing_soft_sources(tech, values) == []


def test_missing_labels_reuse_the_source_type_vocabulary():
    """라벨을 새로 지어내면 화면에 표기가 두 벌 생긴다 — `source_coverage`와 같은 어휘여야 한다."""
    labels = missing_soft_sources([], [])
    assert set(labels) <= {t.value for t in SourceType}


# --- 수집 성공 --------------------------------------------------------------


def test_tech_blog_document_is_typed_and_stamped():
    fetcher = RecordingFetcher({TECH_URL: page()})
    (doc,) = collect(fetch_tech_blog, fetcher)

    assert doc.source_type is SourceType.TECH_BLOG
    assert doc.url == TECH_URL
    assert doc.company == COMPANY
    assert doc.collected_at == TODAY
    assert doc.confidence is Confidence.MID
    assert "GPU 클러스터 스케줄링" in doc.raw_text
    assert doc.doc_id.startswith("blog-")


def test_values_document_is_typed_and_stamped():
    fetcher = RecordingFetcher({CAREERS_URL: page()})
    (doc,) = collect(get_company_values, fetcher)

    assert doc.source_type is SourceType.VALUES
    assert doc.url == CAREERS_URL
    assert doc.confidence is Confidence.MID
    assert doc.doc_id.startswith("values-")


@BOTH
def test_script_and_style_never_reach_the_body(func, url):
    fetcher = RecordingFetcher({url: page()})
    (doc,) = collect(func, fetcher)

    assert "do-not-extract" not in doc.raw_text
    assert "color: red" not in doc.raw_text


@BOTH
def test_soft_sources_never_claim_high_confidence(func, url):
    """상은 사용자가 직접 붙여넣은 원문의 자리다 — 긁어 온 페이지가 차지하면 신뢰등급이 헐거워진다."""
    fetcher = RecordingFetcher({url: page()})
    (doc,) = collect(func, fetcher)
    assert doc.confidence is not Confidence.HIGH


def test_external_hosting_is_low_confidence():
    """회사 도메인 밖(GitHub·Medium 등)은 ③층이다."""
    site = CompanySite("example.com", tech_blog=("https://medium.com/@example/1",), github_org="example")
    fetcher = RecordingFetcher({"https://medium.com/@example/1": page()})

    (doc,) = fetch_tech_blog(COMPANY, fetch=fetcher, today=TODAY, sites={"클라우드마인드": site})
    assert doc.confidence is Confidence.LOW
    assert doc.source_type is SourceType.TECH_BLOG, "GitHub·Medium도 계약상 유형은 TECH_BLOG다"


def test_a_curated_address_on_another_domain_is_still_the_company_channel():
    """기술블로그가 다른 등록 도메인에 사는 일이 흔하다(`toss.im` → `toss.tech`).

    관례로 찾은 게 아니라 **레지스트리에 손으로 확인해 박은 주소**라서 하로 떨어뜨리지 않는다.
    """
    site = CompanySite("example.com", tech_blog=("https://example.tech/",))
    fetcher = RecordingFetcher({"https://example.tech/": page(TECH_ONLY)})

    (doc,) = fetch_tech_blog(COMPANY, fetch=fetcher, today=TODAY, sites={"클라우드마인드": site})
    assert doc.confidence is Confidence.MID


def test_github_org_is_collected_as_a_tech_source():
    site = CompanySite("example.com", github_org="cloudmind")
    fetcher = RecordingFetcher({"https://github.com/cloudmind": page(TECH_ONLY)})

    (doc,) = fetch_tech_blog(COMPANY, fetch=fetcher, today=TODAY, sites={"클라우드마인드": site})
    assert doc.url == "https://github.com/cloudmind"
    assert doc.confidence is Confidence.LOW


def test_multiple_pages_are_collected_up_to_the_limit():
    fetcher = RecordingFetcher(
        {TECH_URL: page(), "https://techblog.example.com/": page(TECH_ONLY)}
    )
    docs = collect(fetch_tech_blog, fetcher)
    assert [d.url for d in docs] == [TECH_URL, "https://techblog.example.com/"]


def test_body_just_over_the_minimum_is_kept():
    """경계 — `MIN_SOFT_CHARS` 이상이면 산다. 상한을 슬쩍 올리면 이 테스트가 깨진다."""
    body = "인재상 " + "가" * (MIN_SOFT_CHARS - 4)
    fetcher = RecordingFetcher({CAREERS_URL: f"<html><body><p>{body}</p></body></html>"})

    (doc,) = collect(get_company_values, fetcher)
    assert doc.raw_text == body


# --- 유형 판별(표지어) ------------------------------------------------------


def test_a_page_with_no_markers_is_discarded():
    """회사 사이트의 404 안내가 200으로 돌아오는 일은 흔하다 — 그걸 근거로 삼지 않는다."""
    fetcher = RecordingFetcher({TECH_URL: page(NEITHER)})
    assert collect(fetch_tech_blog, fetcher) == []


def test_a_pure_tech_page_is_not_accepted_as_values():
    fetcher = RecordingFetcher({CAREERS_URL: page(TECH_ONLY)})
    assert collect(get_company_values, fetcher) == []


def test_a_pure_values_page_is_not_accepted_as_a_tech_blog():
    fetcher = RecordingFetcher({TECH_URL: page(VALUES_ONLY)})
    assert collect(fetch_tech_blog, fetcher) == []


def test_markers_in_the_title_alone_are_enough():
    """본문이 표지어를 안 써도 제목이 말해 주면 받는다."""
    fetcher = RecordingFetcher(
        {CAREERS_URL: page(NEITHER, head="<title>클라우드마인드 인재상</title>")}
    )
    assert len(collect(get_company_values, fetcher)) == 1


# --- 예산·중복 --------------------------------------------------------------


def test_page_requests_are_capped():
    """후보가 아무리 많아도 남의 서버를 정해진 횟수 이상 두드리지 않는다."""
    site = CompanySite("example.com", tech_blog=tuple(f"https://x{i}.example.com/" for i in range(6)))
    fetcher = RecordingFetcher({})

    assert fetch_tech_blog(COMPANY, fetch=fetcher, today=TODAY, sites={"클라우드마인드": site}) == []
    assert len(fetcher.page_calls) == MAX_ATTEMPTS


def test_collection_stops_once_the_limit_is_reached():
    fetcher = RecordingFetcher(
        {TECH_URL: page(), "https://techblog.example.com/": page(TECH_ONLY)}
    )
    docs = collect(fetch_tech_blog, fetcher, limit=1)

    assert len(docs) == 1
    assert "https://techblog.example.com/" not in fetcher.page_calls, "채웠으면 더 안 두드린다"


def test_a_zero_limit_collects_nothing_and_touches_nothing():
    fetcher = RecordingFetcher({TECH_URL: page()})
    assert collect(fetch_tech_blog, fetcher, limit=0) == []
    assert fetcher.calls == []


def test_the_same_body_on_two_paths_counts_once():
    """`/about`과 `/company/values`가 같은 페이지인 회사가 흔하다 — 문서 두 건이 되면 안 된다."""
    fetcher = RecordingFetcher(
        {CAREERS_URL: page(), "https://recruit.example.com/": page(), VALUES_PATH_URL: page(VALUES_ONLY)}
    )
    docs = collect(get_company_values, fetcher)

    assert len(docs) == 2
    assert len({d.doc_id for d in docs}) == 2


def test_robots_is_read_once_per_host():
    """준수하려고 읽는 파일로 서버를 더 두드리면 앞뒤가 안 맞는다."""
    fetcher = RecordingFetcher({BLOG_PATH_URL: page()})
    collect(fetch_tech_blog, fetcher)

    robots_calls = [u for u in fetcher.calls if u.endswith("/robots.txt")]
    assert len(robots_calls) == len(set(robots_calls))
    assert "https://example.com/robots.txt" in robots_calls


def test_robots_is_read_from_the_host_root_before_the_page():
    fetcher = RecordingFetcher({TECH_URL: page()})
    collect(fetch_tech_blog, fetcher)

    assert fetcher.calls.index("https://tech.example.com/robots.txt") < fetcher.calls.index(TECH_URL)
    assert fetcher.page_calls[0] == TECH_URL, "페이지 요청은 robots를 다 본 뒤에야 나간다"


# --- 큰 문서 ----------------------------------------------------------------


def test_an_oversized_body_is_truncated_at_a_line_boundary():
    """통짜 대형 문서(지속가능경영보고서 같은)에 컨텍스트를 통째로 내주지 않는다."""
    body = "\n".join(f"{i}행 인재상 문서 본문입니다." for i in range(3000))
    assert len(body) > MAX_DOC_CHARS
    fetcher = RecordingFetcher({CAREERS_URL: body})

    (doc,) = collect(get_company_values, fetcher)

    assert len(doc.raw_text) <= MAX_DOC_CHARS
    assert len(doc.raw_text) < len(body), "실제로 잘렸어야 이 테스트가 뭔가를 본다"
    assert body.startswith(doc.raw_text), "앞에서부터 잘랐고 없던 문장을 끼워 넣지 않았다"
    assert doc.raw_text.split("\n")[-1] in body.split("\n"), (
        "줄 가운데서 자르면 마지막 줄이 원문에 없는 문장이 된다 — evidence.quote 대조가 걸린다"
    )


# --- 메타데이터 -------------------------------------------------------------


def test_title_comes_from_the_page():
    fetcher = RecordingFetcher({TECH_URL: page(head="<title>클라우드마인드 기술블로그</title>")})
    (doc,) = collect(fetch_tech_blog, fetcher)
    assert doc.title == "클라우드마인드 기술블로그"


def test_published_at_is_parsed_from_meta():
    head = "<title>기술 이야기</title><meta property='article:published_time' content='2026-07-10T09:00:00+09:00'>"
    fetcher = RecordingFetcher({TECH_URL: page(head=head)})
    (doc,) = collect(fetch_tech_blog, fetcher)
    assert doc.published_at == date(2026, 7, 10)


def test_keywords_are_extracted():
    fetcher = RecordingFetcher({TECH_URL: page()})
    (doc,) = collect(fetch_tech_blog, fetcher)
    assert doc.keywords


def test_doc_id_is_content_addressed():
    """같은 본문이면 수집일이 달라도 같은 문서다."""
    fetcher = RecordingFetcher({TECH_URL: page()})
    first = collect(fetch_tech_blog, fetcher)[0]
    second = fetch_tech_blog(COMPANY, fetch=fetcher, today=date(2026, 9, 1), sites=SITES)[0]

    assert first.doc_id == second.doc_id


def test_doc_id_prefix_separates_the_two_types():
    fetcher = RecordingFetcher({TECH_URL: page(), CAREERS_URL: page()})
    tech = collect(fetch_tech_blog, fetcher)[0]
    values = collect(get_company_values, fetcher)[0]

    assert (tech.doc_id.split("-")[0], values.doc_id.split("-")[0]) == ("blog", "values")
    assert tech.doc_id != values.doc_id, "본문이 같아도 유형이 다르면 다른 문서다"


# --- 좌표 해석 --------------------------------------------------------------


def test_a_known_company_resolves_through_the_registry():
    assert resolve_site("현대오토에버") == KNOWN_SITES["현대오토에버"]


@pytest.mark.parametrize(
    "value",
    ["example.com", "EXAMPLE.com", "www.example.com", "https://example.com/careers", "https://www.example.com"],
)
def test_a_domain_or_url_is_used_directly(value):
    """호출자가 주소를 알고 넘겼다면 그게 사전보다 정확하다."""
    assert resolve_site(value, SITES) == CompanySite("example.com")


def test_a_blank_company_is_never_resolved_even_by_a_registry_with_a_blank_key():
    """빈 입력은 무조건 미수집이다 — 사전 조회로 흘러가면 빈 키가 회사로 둔갑한다."""
    assert resolve_site("", {"": CompanySite("example.com")}) is None


def test_an_address_beats_the_registry():
    """호출자가 주소를 알고 넘겼다면 그게 사전보다 정확하다 — 조회 순서를 못 박는다."""
    stale = {"example.com": CompanySite("이사가기전.example")}
    assert resolve_site("example.com", stale) == CompanySite("example.com")


@pytest.mark.parametrize("value", ["듣도보도못한회사", "", "   ", "회사 이름 with spaces", "ftp://example.com"])
def test_unresolvable_input_is_none(value):
    assert resolve_site(value, SITES) is None


@pytest.mark.parametrize("alias", ["현대오토에버", "(주)현대오토에버", "Hyundai Autoever", "hyundaiautoever"])
def test_aliases_and_corporate_suffixes_resolve_to_the_same_site(alias):
    assert resolve_site(alias) == KNOWN_SITES["현대오토에버"]


def test_a_url_company_falls_back_to_the_page_site_name():
    fetcher = RecordingFetcher(
        {TECH_URL: page(head="<meta property='og:site_name' content='클라우드마인드'>")}
    )
    (doc,) = fetch_tech_blog("https://example.com", fetch=fetcher, today=TODAY)
    assert doc.company == "클라우드마인드"


def test_a_url_company_without_site_name_uses_the_host():
    fetcher = RecordingFetcher({TECH_URL: page(head="<title>t</title>")})
    (doc,) = fetch_tech_blog("https://example.com", fetch=fetcher, today=TODAY)
    assert doc.company == "tech.example.com"


# --- 호스트 선택(apex vs www) ------------------------------------------------


class HostAwareFetcher(RecordingFetcher):
    """호스트별로 robots 응답이 다른 fetcher — apex가 통째로 죽은 사이트를 재현한다."""

    def __init__(self, pages=None, *, dead_hosts: tuple[str, ...] = ()):
        super().__init__(pages)
        self.dead_hosts = dead_hosts

    def __call__(self, url: str) -> str:
        if any(f"://{h}/" in url for h in self.dead_hosts):
            self.calls.append(url)
            raise FetchError("TLS 호스트명 불일치")      # 상태 코드 없는 실패 = 호스트가 없다
        return super().__call__(url)


def test_the_www_host_is_used_when_the_bare_domain_is_dead():
    """실 URL에서 나온 케이스 — apex는 인증서가 안 맞아 죽고 www만 사는 회사가 있다."""
    fetcher = HostAwareFetcher(
        {"https://www.example.com/culture": page()}, dead_hosts=("example.com",)
    )
    (doc,) = collect(get_company_values, fetcher)

    assert doc.url == "https://www.example.com/culture"
    assert doc.confidence is Confidence.MID, "www가 붙어도 회사 자체 도메인이다"


def test_the_bare_domain_is_kept_when_it_answers():
    """robots가 404여도 서버는 살아 있는 것이다 — www로 갈아탈 이유가 없다."""
    fetcher = RecordingFetcher({VALUES_PATH_URL: page()}, robots=None)
    (doc,) = collect(get_company_values, fetcher)

    assert doc.url == VALUES_PATH_URL
    assert not any("www.example.com" in u for u in fetcher.calls)


def test_the_host_probe_does_not_cost_an_extra_request():
    """robots.txt는 어차피 읽어야 하는 파일이다 — 탐색이 요청을 늘리면 안 된다."""
    fetcher = RecordingFetcher({VALUES_PATH_URL: page()})
    collect(get_company_values, fetcher)

    apex_robots = [u for u in fetcher.calls if u == "https://example.com/robots.txt"]
    assert len(apex_robots) == 1


def test_subdomain_candidates_ignore_the_host_probe():
    """`tech.example.com`은 그 자체로 다른 호스트다 — apex가 죽었다고 `www.tech.…`가 되지 않는다."""
    site = CompanySite("example.com")
    urls = candidate_urls(site, SourceType.TECH_BLOG, host="www.example.com")

    assert "https://tech.example.com/" in urls
    assert "https://www.example.com/blog" in urls
    assert not any("www.tech." in u for u in urls)


# --- 후보 목록 --------------------------------------------------------------


@pytest.mark.parametrize("kind", [SourceType.TECH_BLOG, SourceType.VALUES])
def test_candidate_urls_are_deterministic_and_unique(kind):
    urls = candidate_urls(SITE, kind)
    assert urls == candidate_urls(SITE, kind)
    assert len(urls) == len(set(urls))
    assert all(u.startswith("https://") for u in urls)


def test_known_urls_come_first_and_external_hosting_last():
    """상한에 잘릴 때 버려지는 쪽이 덜 믿을 만한 쪽이어야 한다."""
    site = CompanySite("example.com", tech_blog=("https://known.example/1",), github_org="cloudmind")
    urls = candidate_urls(site, SourceType.TECH_BLOG)

    assert urls[0] == "https://known.example/1"
    assert urls[-1] == "https://github.com/cloudmind"


def test_the_two_types_look_in_different_places():
    tech = set(candidate_urls(SITE, SourceType.TECH_BLOG))
    values = set(candidate_urls(SITE, SourceType.VALUES))
    assert not tech & values

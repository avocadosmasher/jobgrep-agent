"""T16 · `fetch_jd_body` 3층 폴백 검증.

전부 오프라인이다 — 이 카드에는 `-m llm`이 없다. 대신 **네트워크를 주입점으로 갈아끼워**
robots.txt 응답·HTTP 오류·괴상한 예외를 전부 재현한다. 실제 채용 사이트를 테스트에서
두드리면 결과가 남의 서버 상태에 매달리고, 무엇보다 카드의 불변식(robots 준수)을
어기면서 그걸 검증하는 꼴이 된다.

R5에 따라 본문 골든 데이터는 `fixtures/jd_sample_aiinfra.json`의 `raw_text`를 그대로
쓴다. 붙여넣기 경로는 그 문자열이 **한 글자도 안 바뀌고** 나오는지를 보고, ②③ 경로는
같은 본문을 HTML로 감싸서 태그를 벗겨낸 결과를 대조한다.

핵심 증명 두 개:
- `test_pasted_body_never_touches_the_network` — ①은 네트워크를 안 탄다 (호출 기록으로)
- `test_backbone_survives_total_collection_failure` — ②③이 전부 죽어도 ①은 정상 (카드 완료 조건)
"""

from __future__ import annotations

import gzip
import io
import zlib
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from contracts.enums import Confidence, SourceType
from contracts.models import SourceDocument
from tools import fetch_jd as mod
from tools.fetch_jd import (
    MIN_BODY_CHARS,
    UNCOLLECTED_TITLE,
    FetchError,
    fetch_jd_body,
    is_uncollected,
    keywords,
    read_html,
    robots_allows,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
JD = SourceDocument.model_validate_json((FIXTURES / "jd_sample_aiinfra.json").read_bytes())
BODY = JD.raw_text
# 붙여넣기 경로가 보장하는 것은 "앞뒤 공백만 뗀 원문 그대로"다. 안쪽 빈 줄·들여쓰기는
# 손대지 않는다 — 픽스처 본문이 섹션 사이에 빈 줄을 갖고 있어 그 성질을 같이 증명한다.
PASTED = BODY.strip()

COMPANY_URL = "https://careers.example.com/jobs/ai-infra"
PLATFORM_URL = "https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=1"
ALLOW_ALL = "User-agent: *\nAllow: /\n"
TODAY = date(2026, 8, 8)


def html_page(body: str = BODY, *, head: str = "<title>AI 인프라 엔지니어 채용</title>") -> str:
    """본문 + 사람 눈에 안 보이는 잡음(스크립트·스타일)을 함께 담은 페이지."""
    paragraphs = "\n".join(f"<p>{line}</p>" for line in body.split("\n"))
    return (
        "<!doctype html><html><head>"
        f"{head}"
        "<style>.jd { color: red; }</style>"
        "</head><body><nav>채용 홈</nav>"
        f"<div class='jd'>{paragraphs}</div>"
        "<script>var tracking = 'do-not-extract';</script>"
        "</body></html>"
    )


class RecordingFetcher:
    """URL → 본문 맵. 없는 URL은 404, **모든 호출을 기록**한다.

    D50의 교훈대로 "안 불렀다"는 예외가 아니라 이 기록으로 증명한다 — 중간에 예외를
    삼키는 계층(`fetch_jd_body`가 바로 그렇다)이 있으면 예외 기반 증명은 성립하지 않는다.
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


def exploding_fetcher(url: str) -> str:
    raise RuntimeError("네트워크가 통째로 죽었다")


# --- ① 백본 ----------------------------------------------------------------


def test_pasted_body_is_returned_verbatim_with_high_confidence():
    doc = fetch_jd_body(BODY, company="클라우드마인드", today=TODAY)

    assert doc.raw_text == PASTED, "붙여넣은 전문은 손대지 않는다 (법적 무결·전문 보장)"
    assert "\n\n" in doc.raw_text, "안쪽 빈 줄까지 그대로 — 정규화는 ②③에만 적용된다"
    assert doc.confidence is Confidence.HIGH
    assert doc.source_type is SourceType.JD
    assert doc.url is None
    assert doc.company == "클라우드마인드"
    assert doc.collected_at == TODAY
    assert not is_uncollected(doc)


def test_pasted_body_never_touches_the_network():
    """①이 백본인 이유 — 네트워크를 아예 안 탄다."""
    fetcher = RecordingFetcher()
    fetch_jd_body(BODY, fetch=fetcher, today=TODAY)
    assert fetcher.calls == []


def test_backbone_survives_total_collection_failure():
    """카드 완료 조건 — ②③이 전부 실패해도 ①만으로 정상 문서가 나온다.

    robots는 던지고 페이지는 폭발하는 fetcher를 준다. 그래도 붙여넣기는 멀쩡해야 한다.
    """
    doc = fetch_jd_body(BODY, fetch=exploding_fetcher, today=TODAY)

    assert doc.raw_text == PASTED
    assert doc.confidence is Confidence.HIGH
    assert not is_uncollected(doc)


@pytest.mark.parametrize(
    "pasted",
    [
        f"지원 링크: {COMPANY_URL}\n\n{PASTED}",
        f"{COMPANY_URL}\n\n{PASTED}",      # URL로 **시작**하는 붙여넣기가 진짜 함정이다
    ],
)
def test_text_containing_a_url_is_not_mistaken_for_a_url(pasted):
    """본문 안에 URL이 섞여 있다고 URL로 오인하면 백본이 통째로 날아간다."""
    fetcher = RecordingFetcher()

    doc = fetch_jd_body(pasted, fetch=fetcher, today=TODAY)

    assert doc.raw_text == pasted
    assert doc.confidence is Confidence.HIGH
    assert fetcher.calls == []


@pytest.mark.parametrize("value", ["file:///etc/passwd", "javascript:alert(1)", "ftp://x/y"])
def test_non_http_schemes_are_treated_as_text_not_fetched(value):
    fetcher = RecordingFetcher()
    doc = fetch_jd_body(value, fetch=fetcher, today=TODAY)
    assert fetcher.calls == []
    assert doc.raw_text == value


@pytest.mark.parametrize("value", ["", "   \n  ", None])
def test_empty_input_yields_an_uncollected_document(value):
    doc = fetch_jd_body(value, today=TODAY)
    assert is_uncollected(doc)
    assert doc.confidence is Confidence.LOW
    assert UNCOLLECTED_TITLE in doc.title


# --- ②③ 층 판별 -------------------------------------------------------------


def test_company_page_is_layer2_mid_confidence():
    fetcher = RecordingFetcher({COMPANY_URL: html_page()})
    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)

    assert doc.confidence is Confidence.MID
    assert doc.url == COMPANY_URL
    assert not is_uncollected(doc)


@pytest.mark.parametrize(
    "url",
    [
        PLATFORM_URL,
        "https://www.jobkorea.co.kr/Recruit/GI_Read/1",
        "https://www.wanted.co.kr/wd/1",
        "https://jobs.linkedin.com/view/1",          # 서브도메인도 플랫폼이다
        "http://indeed.com/viewjob?jk=1",            # http도 마찬가지
    ],
)
def test_platform_page_is_layer3_low_confidence(url):
    fetcher = RecordingFetcher({url: html_page()}, robots=ALLOW_ALL)
    doc = fetch_jd_body(url, fetch=fetcher, today=TODAY)

    assert doc.confidence is Confidence.LOW
    assert not is_uncollected(doc), "층이 ③이어도 수집 자체는 성공해야 한다"


def test_lookalike_host_is_not_a_platform():
    """`saramin.co.kr.evil.com`이 플랫폼으로 잡히면 접미사 비교가 헐거운 것이다."""
    url = "https://saramin.co.kr.evil.com/jobs/1"
    fetcher = RecordingFetcher({url: html_page()})
    assert fetch_jd_body(url, fetch=fetcher, today=TODAY).confidence is Confidence.MID


# --- ②③ 실패는 조용히 -------------------------------------------------------


@pytest.mark.parametrize(
    "fetcher, label",
    [
        (RecordingFetcher({}), "404"),
        (RecordingFetcher({COMPANY_URL: "<html><body>짧다</body></html>"}), "본문 부족"),
        (RecordingFetcher({COMPANY_URL: html_page()}, robots="User-agent: *\nDisallow: /\n"), "robots 차단"),
        (exploding_fetcher, "생예외"),
    ],
)
def test_collection_failure_returns_an_empty_document_without_raising(fetcher, label):
    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)

    assert is_uncollected(doc), label
    assert doc.confidence is Confidence.LOW
    assert doc.raw_text == ""
    assert doc.url == COMPANY_URL
    assert doc.collected_at == TODAY
    assert UNCOLLECTED_TITLE in doc.title


def test_page_fetch_raising_an_unexpected_exception_is_still_silent():
    """robots는 통과했는데 **페이지에서** 생예외가 나는 경우.

    위 파라미터 표의 `exploding_fetcher`는 robots.txt부터 터지므로 `robots_allows`의
    방어에 걸려 페이지 요청까지 가지도 않는다 — 그래서 `fetch_jd_body`의 catch를
    `FetchError`로 좁히는 뮤테이션이 거기선 살아남았다(DEVLOG D54). 이 테스트가 그 구멍이다.
    """

    class RobotsOkThenBoom(RecordingFetcher):
        def __call__(self, url: str) -> str:
            if url.endswith("/robots.txt"):
                return super().__call__(url)
            self.calls.append(url)
            raise RuntimeError("HTML을 읽다가 폭발")

    fetcher = RobotsOkThenBoom()
    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)

    assert COMPANY_URL in fetcher.calls, "페이지 요청까지는 갔어야 이 경로가 검증된다"
    assert is_uncollected(doc)
    assert doc.confidence is Confidence.LOW


def test_body_just_over_the_minimum_is_kept():
    """경계 — `MIN_BODY_CHARS` 이상이면 산다. 상한을 슬쩍 올리면 이 테스트가 깨진다."""
    body = "가" * MIN_BODY_CHARS
    fetcher = RecordingFetcher({COMPANY_URL: f"<html><body><p>{body}</p></body></html>"})

    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)
    assert doc.raw_text == body


# --- robots.txt -------------------------------------------------------------


def test_robots_is_read_from_the_host_root_before_the_page():
    fetcher = RecordingFetcher({COMPANY_URL: html_page()})
    fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)

    assert fetcher.calls[0] == "https://careers.example.com/robots.txt"
    assert fetcher.calls[1] == COMPANY_URL


def test_disallowed_path_is_never_requested():
    """차단이면 **요청 자체를 보내지 않는다** — 받아놓고 버리는 건 준수가 아니다."""
    fetcher = RecordingFetcher(
        {COMPANY_URL: html_page()}, robots="User-agent: *\nDisallow: /jobs/\n"
    )
    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)

    assert COMPANY_URL not in fetcher.calls
    assert is_uncollected(doc)


def test_other_paths_stay_allowed_when_only_one_is_disallowed():
    url = "https://careers.example.com/recruit/ai-infra"
    fetcher = RecordingFetcher({url: html_page()}, robots="User-agent: *\nDisallow: /jobs/\n")

    assert not is_uncollected(fetch_jd_body(url, fetch=fetcher, today=TODAY))


@pytest.mark.parametrize(
    "robots, allowed",
    [
        (ALLOW_ALL, True),
        (None, True),                                   # 404 — 파일이 없는 정상 상태
        (FetchError("HTTP 401", status=401), False),    # RFC 9309 — 인증 요구는 전면 금지
        (FetchError("HTTP 403", status=403), False),
        (FetchError("HTTP 410", status=410), True),     # 그 밖의 4xx는 허용
        (FetchError("HTTP 500", status=500), False),    # 알 수 없으면 금지
        (FetchError("timed out"), False),               # 상태 코드 없는 네트워크 오류
        (RuntimeError("이상한 예외"), False),            # 주입된 fetcher가 뭘 던지든
    ],
)
def test_robots_unavailable_rules(robots, allowed):
    fetcher = RecordingFetcher({COMPANY_URL: html_page()}, robots=robots)
    assert robots_allows(COMPANY_URL, fetcher) is allowed


def test_robots_rules_for_our_user_agent_are_honoured():
    """와일드카드 말고 우리 UA를 콕 집어 막은 경우."""
    fetcher = RecordingFetcher(
        {COMPANY_URL: html_page()},
        robots="User-agent: jobprep-agent\nDisallow: /\n\nUser-agent: *\nAllow: /\n",
    )
    assert robots_allows(COMPANY_URL, fetcher) is False


# --- HTML → 텍스트 ----------------------------------------------------------


def test_script_and_style_contents_never_reach_the_body():
    fetcher = RecordingFetcher({COMPANY_URL: html_page()})
    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)

    assert "do-not-extract" not in doc.raw_text
    assert "color: red" not in doc.raw_text


def test_extracted_body_keeps_the_fixture_lines():
    """태그를 벗긴 결과가 골든 본문의 줄들을 그대로 담고 있는가."""
    fetcher = RecordingFetcher({COMPANY_URL: html_page()})
    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)

    for line in (l.strip() for l in BODY.split("\n") if l.strip()):
        assert line in doc.raw_text


def test_block_tags_become_line_breaks_instead_of_glued_words():
    page = read_html("<html><body><li>Kubernetes 운영</li><li>Docker 빌드</li></body></html>")
    assert "Kubernetes 운영\nDocker 빌드" in page.text
    assert "운영Docker" not in page.text


def test_entities_are_unescaped():
    page = read_html("<html><body><p>R&amp;D 조직 &lt;AI&gt;</p></body></html>")
    assert "R&D 조직 <AI>" in page.text


def test_plain_text_response_passes_through():
    """서버가 text/plain을 주면 태그가 없다 — 그대로 본문이다."""
    fetcher = RecordingFetcher({COMPANY_URL: BODY})
    doc = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY)
    assert "GPU 클러스터 스케줄링" in doc.raw_text


def test_tag_structure_does_not_leak_as_blank_lines():
    """블록 하나 = 한 줄. 마크업이 몇 겹이든 출력 형태가 달라지지 않는다."""
    assert read_html("<p>가</p>\n\n\n\n<p>나</p>").text == "가\n나"
    assert read_html("<div><section><p>가</p></section></div><p>나</p>").text == "가\n나"


# --- 메타데이터 -------------------------------------------------------------


def test_title_comes_from_the_page_when_the_caller_has_none():
    fetcher = RecordingFetcher({COMPANY_URL: html_page()})
    assert fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY).title == "AI 인프라 엔지니어 채용"


def test_caller_metadata_wins_over_the_page():
    """호출자가 아는 회사·제목이 정본이다 — 페이지 메타는 보조."""
    fetcher = RecordingFetcher(
        {COMPANY_URL: html_page(head="<title>채용</title><meta property='og:site_name' content='엉뚱회사'>")}
    )
    doc = fetch_jd_body(
        COMPANY_URL, company="클라우드마인드", department="AI플랫폼실",
        title="AI 인프라 엔지니어", fetch=fetcher, today=TODAY,
    )
    assert (doc.company, doc.department, doc.title) == ("클라우드마인드", "AI플랫폼실", "AI 인프라 엔지니어")


def test_site_name_fills_company_only_when_the_caller_left_it_blank():
    fetcher = RecordingFetcher(
        {COMPANY_URL: html_page(head="<meta property='og:site_name' content='클라우드마인드 채용'>")}
    )
    assert fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY).company == "클라우드마인드 채용"


def test_og_title_is_used_when_there_is_no_title_tag():
    fetcher = RecordingFetcher(
        {COMPANY_URL: html_page(head="<meta property='og:title' content='AI 인프라 채용'>")}
    )
    assert fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY).title == "AI 인프라 채용"


def test_pasted_text_titles_itself_with_its_first_line():
    doc = fetch_jd_body(BODY, today=TODAY)
    assert doc.title == BODY.splitlines()[0].strip()[:80]


@pytest.mark.parametrize(
    "content, expected",
    [
        ("2026-07-10T09:00:00+09:00", date(2026, 7, 10)),
        ("2026-07-10", date(2026, 7, 10)),
        ("어제", None),          # 파싱 실패는 지어내지 않고 비운다
        ("", None),
    ],
)
def test_published_at_is_parsed_from_meta_when_possible(content, expected):
    head = f"<title>t</title><meta property='article:published_time' content='{content}'>"
    fetcher = RecordingFetcher({COMPANY_URL: html_page(head=head)})
    assert fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY).published_at == expected


def test_published_at_is_none_without_meta():
    fetcher = RecordingFetcher({COMPANY_URL: html_page()})
    assert fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY).published_at is None


# --- doc_id -----------------------------------------------------------------


def test_doc_id_is_deterministic_for_the_same_input():
    first = fetch_jd_body(BODY, today=TODAY)
    second = fetch_jd_body(BODY, today=date(2026, 9, 1))
    assert first.doc_id == second.doc_id, "수집일이 달라도 같은 원문은 같은 문서다"


def test_doc_id_differs_across_inputs_and_paths():
    fetcher = RecordingFetcher({COMPANY_URL: html_page()})
    pasted = fetch_jd_body(BODY, today=TODAY).doc_id
    other = fetch_jd_body(BODY + " 추가", today=TODAY).doc_id
    fetched = fetch_jd_body(COMPANY_URL, fetch=fetcher, today=TODAY).doc_id

    assert len({pasted, other, fetched}) == 3


def test_doc_id_prefix_marks_the_source_type():
    assert fetch_jd_body(BODY, today=TODAY).doc_id.startswith("jd-")


# --- 키워드 -----------------------------------------------------------------


def test_fixture_keywords_are_all_recovered():
    """골든 대조 — 픽스처가 손으로 적어 둔 키워드가 원문에서 전부 나온다."""
    extracted = keywords(BODY, limit=30)
    missing = [k for k in JD.keywords if k not in extracted]
    assert not missing, f"못 뽑은 키워드: {missing}"


def test_keywords_are_ranked_by_frequency_then_first_appearance():
    text = "Docker Kubernetes Docker Terraform Kubernetes Docker Terraform"
    assert keywords(text) == ["Docker", "Kubernetes", "Terraform"]


def test_keyword_ties_break_by_appearance_not_alphabet():
    """동률 처리를 사전순으로 바꿔도 위 테스트는 통과한다 — 그래서 이 케이스가 따로 있다."""
    assert keywords("Zeta Alpha Zeta Alpha Beta") == ["Zeta", "Alpha", "Beta"]


def test_keywords_drop_stopwords_and_single_letters():
    assert keywords("the GPU and a Kubernetes of Docker") == ["GPU", "Kubernetes", "Docker"]


def test_keywords_are_deduplicated_case_insensitively_keeping_first_form():
    assert keywords("Kubernetes kubernetes KUBERNETES") == ["Kubernetes"]


def test_keywords_respect_the_limit():
    text = " ".join(f"Tool{i}" for i in range(50))
    assert len(keywords(text, limit=5)) == 5
    assert keywords(text, limit=0) == []


def test_keywords_are_deterministic():
    assert keywords(BODY, limit=30) == keywords(BODY, limit=30)


def test_document_keywords_are_capped_by_default():
    doc = fetch_jd_body(BODY, today=TODAY)
    assert 0 < len(doc.keywords) <= mod.DEFAULT_KEYWORD_LIMIT


def test_korean_only_text_yields_no_garbage_keywords():
    """한국어는 형태소 분석기 없이 안 자른다 — 조사 붙은 쓰레기를 만드느니 비운다."""
    assert keywords("클러스터 운영 경험과 서빙 파이프라인 구축 경험") == []


# --- 기본 fetcher (`http_get`) ----------------------------------------------


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, charset: str | None = "utf-8", **headers: str):
        super().__init__(payload)
        self.headers = _FakeHeaders(charset, headers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _FakeHeaders:
    def __init__(self, charset: str | None, values: dict[str, str] | None = None):
        self._charset = charset
        self._values = {k.replace("_", "-").lower(): v for k, v in (values or {}).items()}

    def get_content_charset(self):
        return self._charset

    def get(self, key, default=None):
        return self._values.get(key.lower(), default)


def test_http_get_decodes_with_the_declared_charset(monkeypatch):
    payload = "채용 공고".encode("euc-kr")
    monkeypatch.setattr(mod, "urlopen", lambda *a, **k: FakeResponse(payload, "euc-kr"))
    assert mod.http_get("https://x.example/jd") == "채용 공고"


def test_http_get_caps_the_download(monkeypatch):
    """통짜 대형 문서에 메모리를 통째로 내주지 않는다."""
    seen: dict[str, int] = {}

    class Capped(FakeResponse):
        def read(self, size=-1):
            seen["size"] = size
            return super().read(size)

    monkeypatch.setattr(mod, "urlopen", lambda *a, **k: Capped(b"x" * 10))
    mod.http_get("https://x.example/jd")
    assert seen["size"] == mod.MAX_BYTES


def test_http_get_wraps_http_errors_with_their_status(monkeypatch):
    def boom(*a, **k):
        raise HTTPError("https://x.example/jd", 403, "Forbidden", {}, None)

    monkeypatch.setattr(mod, "urlopen", boom)
    with pytest.raises(FetchError) as caught:
        mod.http_get("https://x.example/jd")
    assert caught.value.status == 403


def test_http_get_wraps_network_errors_without_a_status(monkeypatch):
    def boom(*a, **k):
        raise URLError("timed out")

    monkeypatch.setattr(mod, "urlopen", boom)
    with pytest.raises(FetchError) as caught:
        mod.http_get("https://x.example/jd")
    assert caught.value.status is None


def test_http_get_unzips_gzip_responses(monkeypatch):
    """실 HTTP 왕복 한 번이 잡아낸 결함 — 압축을 안 풀면 본문이 통째로 깨진다(D52)."""
    payload = gzip.compress(html_page().encode("utf-8"))
    monkeypatch.setattr(
        mod, "urlopen", lambda *a, **k: FakeResponse(payload, "utf-8", Content_Encoding="gzip")
    )
    assert "GPU 클러스터 스케줄링" in mod.http_get("https://x.example/jd")


def test_http_get_unzips_even_when_the_header_is_missing(monkeypatch):
    """CDN이 `Content-Encoding`을 흘리는 경우가 실제로 있다 — 매직 바이트로도 본다."""
    payload = gzip.compress("압축된 본문".encode("utf-8"))
    monkeypatch.setattr(mod, "urlopen", lambda *a, **k: FakeResponse(payload))
    assert mod.http_get("https://x.example/jd") == "압축된 본문"


def test_http_get_inflates_deflate_responses(monkeypatch):
    payload = zlib.compress("디플레이트 본문".encode("utf-8"))
    monkeypatch.setattr(
        mod, "urlopen", lambda *a, **k: FakeResponse(payload, "utf-8", Content_Encoding="deflate")
    )
    assert mod.http_get("https://x.example/jd") == "디플레이트 본문"


def test_http_get_falls_back_to_the_meta_charset(monkeypatch):
    """한국어 채용 페이지는 헤더 없이 `<meta charset=euc-kr>`만 있는 경우가 흔하다."""
    payload = "<meta charset='euc-kr'><p>채용 공고</p>".encode("euc-kr")
    monkeypatch.setattr(mod, "urlopen", lambda *a, **k: FakeResponse(payload, None))
    assert "채용 공고" in mod.http_get("https://x.example/jd")


def test_http_get_survives_an_unknown_charset(monkeypatch):
    monkeypatch.setattr(
        mod, "urlopen", lambda *a, **k: FakeResponse("공고".encode("utf-8"), "x-made-up")
    )
    assert mod.http_get("https://x.example/jd") == "공고"


def test_http_get_does_not_advertise_brotli(monkeypatch):
    """표준 라이브러리로 못 푸는 인코딩을 광고하면 서버가 그걸로 준다."""
    captured: dict[str, object] = {}

    def capture(request, **kwargs):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return FakeResponse(b"ok")

    monkeypatch.setattr(mod, "urlopen", capture)
    mod.http_get("https://x.example/jd")
    assert "br" not in captured["headers"]["Accept-encoding".lower()].split(", ")


def test_http_get_sends_a_user_agent(monkeypatch):
    captured: dict[str, object] = {}

    def capture(request, **kwargs):
        captured["headers"] = request.headers
        return FakeResponse(b"ok")

    monkeypatch.setattr(mod, "urlopen", capture)
    mod.http_get("https://x.example/jd")
    assert "jobprep-agent" in str(captured["headers"])

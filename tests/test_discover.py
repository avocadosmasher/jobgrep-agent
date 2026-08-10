"""T19 · `discover_jobs` 검증 — 공고 열거.

**네트워크를 한 번도 타지 않는다.** HTTP는 `fetch=`로 주입한다. 다만 D53·D56이
남긴 것을 기억할 것 — 주입점 뒤에 갇힌 테스트는 바이트·DNS 계층을 아예 안 본다.
이 모듈이 그 계층을 새로 만들지 않고 T16·T17 것을 import 하는 이유가 그것이며,
실 URL 확인은 DEVLOG D65에 따로 남겼다.
"""

from __future__ import annotations

from datetime import date

import pytest

from contracts.enums import Confidence, SourceType
from tools.discover import (
    MAX_ATTEMPTS,
    OTHER_CATEGORY,
    careers_urls,
    categorize,
    discover_jobs,
    group_jobs,
    is_job_link,
    job_doc_id,
    read_links,
)
from tools.fetch_jd import FetchError, is_uncollected
from tools.fetch_soft import CompanySite

TODAY = date(2026, 8, 9)  # 오늘이 아닌 날짜 — 수집일 무시 회귀가 하루만 통과하는 함정(D57)

COMPANY = "테크노베이션"
DOMAIN = "technovation.example"
SITES = {COMPANY: CompanySite(DOMAIN)}

ROBOTS_OK = "User-agent: *\nAllow: /\n"

CAREERS_HTML = """
<html><body>
  <a href="/careers/jobs/101">백엔드 엔지니어 (신입)</a>
  <a href="/careers/jobs/102">프론트엔드 개발자</a>
  <a href="https://boards.greenhouse.io/technovation/jobs/103">AI 플랫폼 엔지니어</a>
  <a href="/login">로그인</a>
  <a href="/notice/17">채용 공지사항</a>
  <a href="https://twitter.com/technovation">트위터</a>
  <a href="/about">회사 소개</a>
  <a href="/team">팀 이야기</a>
  <a href="#">채용 공고 목록</a>
  <a href="https://alba.technovation.example/job-posts/9">주5일 주방 직원 구합니다</a>
</body></html>
"""


class Server:
    """URL → 응답 본문. 없는 주소는 404로 던진다 — 실제 서버와 같은 모양."""

    def __init__(self, pages: dict[str, str], *, robots: str | None = ROBOTS_OK) -> None:
        self.pages = pages
        self.robots = robots
        self.calls: list[str] = []

    def __call__(self, url: str) -> str:
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            if self.robots is None:
                raise FetchError("robots 없음", status=404)
            return self.robots
        if url in self.pages:
            return self.pages[url]
        raise FetchError("없음", status=404)

    def page_calls(self) -> list[str]:
        return [u for u in self.calls if not u.endswith("/robots.txt")]


def careers_server(**extra: str) -> Server:
    return Server({f"https://careers.{DOMAIN}/": CAREERS_HTML, **extra})


def find(company: str = COMPANY, role: str = "", *, fetch=None, **kw):
    return discover_jobs(
        company, role, fetch=fetch or careers_server(), today=TODAY, sites=SITES, **kw
    )


# --- ① 후보 URL ----------------------------------------------------------------


def test_careers_urls_are_ordered_and_deduplicated():
    urls = careers_urls(CompanySite(DOMAIN))
    assert urls[0] == f"https://careers.{DOMAIN}/", "서브도메인이 경로보다 먼저"
    assert len(urls) == len(set(urls))
    assert f"https://{DOMAIN}/careers" in urls


def test_path_candidates_hang_off_the_live_host():
    """`live_host()`가 고른 호스트에 경로가 매달린다 — apex/`www.` 문제(D56)."""
    urls = careers_urls(CompanySite(DOMAIN), host=f"www.{DOMAIN}")
    assert f"https://www.{DOMAIN}/careers" in urls
    # 서브도메인 후보는 그 자체로 다른 호스트라 영향받지 않는다.
    assert f"https://careers.{DOMAIN}/" in urls


# --- ② 링크 파싱 ---------------------------------------------------------------


def test_read_links_makes_urls_absolute_and_collapses_text():
    links = read_links(
        '<a href="/jobs/1">  백엔드\n  엔지니어 </a>', "https://x.example/careers"
    )
    assert links == [("https://x.example/jobs/1", "백엔드 엔지니어")]


def test_read_links_drops_fragments_and_duplicates():
    html = (
        '<a href="/jobs/1">백엔드</a>'
        '<a href="/jobs/1#apply">백엔드 지원</a>'
        '<a href="/jobs/2">프론트</a>'
    )
    urls = [u for u, _ in read_links(html, "https://x.example/")]
    assert urls == ["https://x.example/jobs/1", "https://x.example/jobs/2"]


def test_read_links_ignores_anchors_without_text():
    """글자가 없는 링크(아이콘 등)는 공고 제목이 될 수 없다."""
    html = '<a href="/jobs/1"><img src="i.png"></a><a href="/jobs/2">채용</a>'
    assert [t for _, t in read_links(html, "https://x.example/")] == ["채용"]


@pytest.mark.parametrize(
    "url,text,expected",
    [
        ("https://x.example/careers/jobs/1", "무제", True),      # 경로 관례
        ("https://x.example/kor/main.do?menuNo=9", "채용 안내", True),  # 글자만
        ("https://x.example/login", "채용 로그인", False),        # 제외가 먼저
        ("https://x.example/notice/1", "채용 공지", False),
        ("javascript:void(0)", "지원하기", False),
        ("mailto:hr@x.example", "채용 문의", False),
        ("https://x.example/about", "회사 소개", False),          # 어느 표지도 없음
        # **뮤테이션 M08** — 표지를 URL 전체에 대면 호스트명이 경로 표지를 만족시킨다.
        # 채용 서브도메인은 관례라 이 오탐이 기본값이 된다.
        ("https://careers.x.example/team", "팀 이야기", False),
        ("https://recruit.x.example/press/3", "보도자료", False),
    ],
)
def test_is_job_link(url, text, expected):
    """**뮤테이션 M09/M10 — 표지 목록이나 제외 목록을 지우면 여기가 죽는다.**"""
    assert is_job_link(url, text) is expected


# --- ③ 그룹핑 ------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,expected",
    [
        ("백엔드 엔지니어", "백엔드"),
        ("AI 플랫폼 엔지니어", "AI·ML"),
        ("MLOps Engineer", "AI·ML"),
        ("데이터 분석가", "데이터"),
        ("Cloud Infrastructure Engineer", "인프라·클라우드"),
        ("총무 담당자", OTHER_CATEGORY),
    ],
)
def test_categorize_is_deterministic(title, expected):
    assert categorize(title) == expected


def test_group_order_follows_the_table_and_other_is_last():
    """목록 순서가 실행마다 흔들리면 사용자가 자기가 뭘 봤는지 기억하지 못한다."""
    jobs = find(fetch=careers_server())
    jobs.append(jobs[0].model_copy(update={"title": "총무", "doc_id": "job-etc"}))

    groups = list(group_jobs(jobs))
    assert groups[-1] == OTHER_CATEGORY
    assert groups.index("AI·ML") < groups.index("백엔드") < groups.index("프론트엔드")


# --- ④ 발견 --------------------------------------------------------------------


def test_discovers_job_links_from_the_careers_page():
    jobs = find()
    titles = {j.title for j in jobs}

    assert titles == {"백엔드 엔지니어 (신입)", "프론트엔드 개발자", "AI 플랫폼 엔지니어"}
    assert all(j.source_type is SourceType.JD for j in jobs)


def test_discovered_jobs_carry_no_body():
    """**뮤테이션 M05 — 여기서 본문을 받아 오면 이 테스트가 죽는다.**

    계약도 설계도도 "메타"라고 적었고, 20건을 열거해 19건을 버리는 낭비를 막는
    장치이기도 하다. `is_uncollected()`가 참이라 하드 게이트도 이걸 안 센다(D52).
    """
    jobs = find()
    assert all(j.raw_text == "" for j in jobs)
    assert all(is_uncollected(j) for j in jobs)


def test_same_url_always_gets_the_same_id():
    """멱등 — 같은 공고를 몇 번 발견해도 id가 같아야 선택이 재현된다."""
    first = {j.doc_id for j in find()}
    second = {j.doc_id for j in find()}
    assert first == second
    assert job_doc_id("https://a.example/1") != job_doc_id("https://a.example/2")


def test_role_ranks_but_never_filters():
    """**뮤테이션 M06 — 직무명으로 거르면 여기가 죽는다.** 임의 선택 금지(§12-2)."""
    jobs = find(role="백엔드 엔지니어")

    assert jobs[0].title == "백엔드 엔지니어 (신입)", "관련된 것이 앞으로 와야 한다"
    assert len(jobs) == 3, "관련 없는 공고도 목록에 남아야 한다"


def test_company_pages_are_mid_confidence():
    own = [j for j in find() if DOMAIN in (j.url or "")]
    assert own and {j.confidence for j in own} == {Confidence.MID}


def test_the_query_string_is_part_of_a_job_identity():
    """쿼리를 떼면 안 된다 — ATS·대기업 CMS는 `?jobId=`·`?rec_idx=`가 곧 공고 번호다.

    같은 경로에 쿼리만 다른 두 링크는 **다른 공고**로 남는다. 합치는 쪽이 위험하다
    (§12-2 임의 병합 금지).
    """
    server = careers_server(
        **{
            f"https://careers.{DOMAIN}/": (
                '<a href="/careers/jobs?id=101">백엔드 엔지니어</a>'
                '<a href="/careers/jobs?id=102">프론트엔드 개발자</a>'
            )
        }
    )
    jobs = find(fetch=server)
    assert len({j.doc_id for j in jobs}) == len(jobs) == 2


def test_search_stops_at_the_first_page_that_yields_jobs():
    """**뮤테이션 M28 — 여러 페이지 결과를 합치면 여기가 죽는다.**

    당근이 그 실물이다(D64) — `careers.daangn.com`은 회사 채용 사이트인데
    `jobs.daangn.com`은 당근알바(동네 구인)라, 합치면 "김밥집 주방 직원"이 회사
    채용공고 목록에 올라온다. 후보 순서가 곧 신뢰 순서이므로 **처음 건진 곳에서 멈춘다.**
    """
    server = Server(
        {
            f"https://careers.{DOMAIN}/": '<a href="/careers/jobs/1">백엔드 엔지니어</a>',
            f"https://jobs.{DOMAIN}/": '<a href="/job-posts/9">김밥집 주방 직원 구합니다</a>',
        }
    )
    jobs = find(fetch=server)

    assert [j.title for j in jobs] == ["백엔드 엔지니어"]
    assert f"https://jobs.{DOMAIN}/" not in server.page_calls(), "이미 건졌는데 더 두드렸다"


# --- ④-2 범위 — 채용 페이지가 거느리는 공고인가 (D64) ---------------------------


def test_links_leaving_the_careers_host_are_out_of_scope():
    """**뮤테이션 M27 — `in_scope`를 지우면 여기가 죽는다.**

    실물에서 나왔다(D64) — `careers.daangn.com`에는 `jobs.daangn.com`(당근알바)로
    나가는 링크가 잔뜩 있고, 그건 "김밥집 주방 직원" 같은 동네 구인이다. 진짜
    채용공고이긴 해서 표지어로는 못 거르고, 같은 등록 도메인이라 도메인 비교로도
    못 거른다.
    """
    titles = {j.title for j in find()}
    assert "주5일 주방 직원 구합니다" not in titles, "다른 호스트의 구인글이 섞였다"


def test_a_link_back_to_the_careers_page_is_not_a_job():
    """**뮤테이션 M23 — 자기 링크 검사를 지우면 여기가 죽는다.**

    `<a href="#">`은 조각을 떼면 페이지 주소 그대로가 되고, 채용 페이지 경로는
    공고 표지를 당연히 만족한다 — 목록 자신이 공고 한 건으로 둔갑한다.
    """
    assert "채용 공고 목록" not in {j.title for j in find()}


def test_known_ats_hosts_stay_in_scope():
    """회사 공고가 ATS에 사는 것은 정상이다 — 범위 규칙이 그걸 죽이면 안 된다."""
    assert "AI 플랫폼 엔지니어" in {j.title for j in find()}


def test_job_platforms_stay_in_scope_with_low_confidence():
    """회사 채용페이지가 자기 사람인 공고로 링크하는 것도 정상(③층)."""
    server = careers_server(
        **{
            f"https://careers.{DOMAIN}/": (
                '<a href="https://www.saramin.co.kr/zf_user/jobs/relay/view?rec_idx=1">'
                "백엔드 채용</a>"
            )
        }
    )
    jobs = find(fetch=server)
    assert [j.confidence for j in jobs] == [Confidence.LOW]


# --- ⑤ 실패는 전부 빈 리스트 ----------------------------------------------------


def test_unknown_company_touches_nothing():
    """사전에 없는 회사는 **주소를 하나도 두드리지 않는다** — 조용한 미수집."""
    server = careers_server()
    assert discover_jobs("듣보사", fetch=server, today=TODAY, sites=SITES) == []
    assert server.calls == []


def test_blank_company_is_not_a_lookup():
    assert discover_jobs("", fetch=careers_server(), today=TODAY, sites=SITES) == []


def test_robots_blocked_yields_nothing():
    server = Server(
        {f"https://careers.{DOMAIN}/": CAREERS_HTML},
        robots="User-agent: *\nDisallow: /\n",
    )
    assert find(fetch=server) == []
    assert server.page_calls() == [], "차단된 페이지를 가져오면 안 된다"


def test_an_exploding_fetcher_is_silent():
    """**어디서 실패하는지를 지정한다** — 앞 계층이 먹으면 뒤가 검증 안 된다(D54).

    robots는 정상 응답하고 **페이지에서만** 생예외를 던진다.
    """

    class Exploding(Server):
        def __call__(self, url: str) -> str:
            if url.endswith("/robots.txt"):
                return super().__call__(url)
            self.calls.append(url)
            raise RuntimeError("네트워크 붕괴")

    server = Exploding({})
    assert find(fetch=server) == []
    assert server.page_calls(), "페이지 요청이 실제로 나갔나"


def test_a_page_without_job_links_yields_nothing():
    server = careers_server(**{f"https://careers.{DOMAIN}/": "<a href='/about'>소개</a>"})
    assert find(fetch=server) == []


# --- ⑥ 상한 --------------------------------------------------------------------


def test_page_requests_stop_at_the_attempt_cap():
    """**뮤테이션 M07 — `MAX_ATTEMPTS` 검사를 지우면 여기가 죽는다.**

    후보가 상한보다 많아도 남의 서버를 그만큼만 두드린다.
    """
    server = Server({})  # 전부 404 — 끝까지 후보를 소진하려 든다
    assert find(fetch=server) == []
    assert len(server.page_calls()) == MAX_ATTEMPTS


def test_limit_caps_the_result():
    jobs = find(limit=2)
    assert len(jobs) == 2


def test_zero_limit_makes_no_request():
    """**뮤테이션 M08 — `limit <= 0` 가드를 지우면 여기가 죽는다**(D57과 같은 자리)."""
    server = careers_server()
    assert find(fetch=server, limit=0) == []
    assert server.calls == []

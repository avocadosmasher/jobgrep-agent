"""공고 발견 — 회사 채용페이지에서 열린 공고를 **열거**한다 (설계도 §6-2, §12-2).

**이 모듈은 크리티컬 패스가 아니다.** §6-2가 v1에서 바꾼 것이 정확히 그 점이다 —
백본은 사용자 입력(붙여넣기/URL)이고 발견은 "있으면 UX가 좋아지는 것"이다. 그래서
실패는 전부 **빈 리스트**이고 예외를 한 건도 위로 올리지 않는다(T17과 같은 규약,
D55). 발견이 0건이면 사용자는 그냥 JD를 붙여넣으면 된다.

## 본문을 안 가져온다 — 메타만 가져온다

계약(`contracts/tools.py`)이 "후보 공고 SourceDocument 목록 — H1의 입력이 된다"라
적었고 설계도 §5 표도 "SourceDocument[] **메타**"다. 그래서 여기서 돌려주는 문서는
`raw_text=""` — 즉 `is_uncollected()`가 **참**이다. 본문은 사용자가 고른 것만
`fetch_jd_body`로 가져온다(T19의 `select_job` 노드).

이건 게으름이 아니라 예산이다. 공고가 20건 열거됐는데 전부 본문을 받아 오면
남의 서버에 20번 요청하고 그중 19건은 버린다. **고른 것만 가져온다.**

`is_uncollected()`가 참이라는 사실이 하류에서 그대로 값을 한다 — T18의
`collected_job_ids()`가 이 문서들을 세지 않으므로, 발견만 되고 선택되지 않은 공고가
`BriefMeta.selected_jobs`에 실려 하드 게이트를 헛통과시키는 일이 구조적으로 없다(D52).

## 어떻게 찾나

검색 엔진 연동이 이 프로젝트에 없으므로 T17과 같은 길을 간다 — 회사명에서 도메인을
얻고(`resolve_site()`), 관례적인 채용 페이지를 두드리고, 그 페이지의 **링크 중
공고처럼 생긴 것**을 골라낸다. 못 찾으면 조용히 빈 리스트다.

HTTP·robots·HTML·도메인 해석은 **T16·T17 것을 그대로 import 한다.** 같은 걸 다시
짜면 gzip 결함(D53)이나 apex/`www.` 문제(D56) 같은 게 한쪽에만 고쳐진다.

## 임의 선택·병합 금지

이 모듈은 **고르지 않는다.** 직무명으로 순서를 매기긴 하지만 관련 없어 보이는
공고를 버리지 않는다 — 무엇이 관련 있는지는 사용자가 안다(§12-2 "부department/공고
모호 → H1으로 처리, 임의 선택·병합 금지"). 상한(`MAX_JOBS`)에 걸려 잘릴 때 뒤쪽이
버려지므로, 정렬은 **덜 관련된 것이 뒤로 가도록** 하는 것이 목적이다.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import date
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from contracts.enums import Confidence, SourceType
from contracts.models import SourceDocument
from tools.fetch_jd import (
    PLATFORM_HOSTS,
    Fetcher,
    http_get,
    robots_allows,
)
from tools.fetch_soft import CompanySite, live_host, resolve_site

MAX_ATTEMPTS = 6      # 한 호출이 남의 서버에 보내는 채용페이지 요청 상한
MAX_JOBS = 20         # 열거 상한 — multiselect가 읽을 수 있는 길이를 넘지 않는다
MIN_TITLE_CHARS = 2   # 링크 글자가 이보다 짧으면 공고 제목으로 안 본다
MAX_TITLE_CHARS = 120

# 채용 페이지 관례. 순서가 곧 우선순위다 (T17 `_VALUES_PATHS`와 같은 사고).
JOB_SUBDOMAINS = ("careers", "recruit", "job", "jobs")
JOB_PATHS = ("/careers", "/recruit", "/jobs", "/career", "/careers/jobs", "/recruit/jobs")

# 링크가 **공고 상세**처럼 생겼는가. 채용 페이지에는 공고 말고도 온갖 링크가 있다.
_JOB_HREF_MARKERS = (
    "/job", "/jobs/", "/position", "/recruit", "/career", "/vacanc", "/opening",
    "/apply", "/posting", "jobid=", "job_id=", "rec_idx=", "recruitno", "annoid",
)

# href만으로는 안 걸리지만 글자가 공고를 말하는 경우 (한국 대기업 CMS는 경로가 무의미하다).
_JOB_TEXT_MARKERS = (
    "채용", "모집", "공채", "경력", "신입", "인턴", "지원",
    "engineer", "developer", "manager", "designer", "intern", "hiring",
)

# 공고가 아닌 것이 확실한 경로. 위 표지를 우연히 만족해도 버린다.
# **경로만 본다** — 아래 `is_job_link()` 주석 참조.
_EXCLUDE_PATHS = (
    "/login", "/logout", "/signup", "/privacy", "/terms", "/faq",
    "/notice", "/news", "/blog", "/search", "/about", "/contact",
)

# 공고가 회사 자체 호스트 밖에 사는 정상적인 경우 — 채용 ATS.
# **허용 목록이다.** 여기 없는 남의 호스트로 나가는 링크는 공고로 안 친다(D64).
ATS_HOSTS = frozenset(
    {
        "greenhouse.io",
        "boards.greenhouse.io",
        "lever.co",
        "jobs.lever.co",
        "ashbyhq.com",
        "jobs.ashbyhq.com",
        "workable.com",
        "recruitee.com",
        "smartrecruiters.com",
        "myworkdayjobs.com",
        "taleo.net",
    }
)

# 직무 대분류 — 제목에서 결정론적으로 유도한다. **LLM 없음.**
# 순서가 우선순위다(앞에서 걸리면 거기서 멈춘다). 어디에도 안 걸리면 "기타".
JOB_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AI·ML", ("ai", "ml", "머신러닝", "딥러닝", "인공지능", "llm", "mlops", "비전", "nlp")),
    ("데이터", ("데이터", "data", "analytics", "분석", "dba", "etl", "빅데이터")),
    ("인프라·클라우드", ("인프라", "클라우드", "cloud", "infra", "devops", "sre", "kubernetes", "네트워크", "시스템엔지니어")),
    ("백엔드", ("백엔드", "backend", "server", "서버", "api")),
    ("프론트엔드", ("프론트", "frontend", "front-end", "web", "웹개발", "ui개발")),
    ("모바일", ("android", "ios", "안드로이드", "모바일", "flutter", "react native")),
    ("보안", ("보안", "security", "정보보호", "침해")),
    ("QA·테스트", ("qa", "테스트", "품질", "test engineer")),
    ("기획·PM", ("기획", "pm", "product manager", "po", "프로덕트")),
    ("디자인", ("디자인", "design", "ux", "ui디자")),
)
OTHER_CATEGORY = "기타"

_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")


# --- 공개 API ---------------------------------------------------------------


def discover_jobs(
    company: str,
    role: str = "",
    *,
    fetch: Fetcher | None = None,
    today: date | None = None,
    sites: Mapping[str, CompanySite] | None = None,
    limit: int = MAX_JOBS,
) -> list[SourceDocument]:
    """회사의 채용 페이지에서 공고 후보를 열거한다 — **메타만, 본문 없음.**

    입력: 회사명 — 도메인("example.com")이나 채용 페이지 URL을 그대로 넘겨도 된다.
        role은 **정렬에만** 쓰인다(관련 있어 보이는 것이 앞으로 온다). 거르지 않는다.
    출력: `SourceDocument(source_type=JD, raw_text="")` 목록. `url`이 정본이고
        `title`은 링크 글자다. **못 찾으면 빈 리스트다.**
    불변식: 어떤 실패도 예외로 올리지 않는다. robots.txt를 준수한다. 페이지 요청은
        `MAX_ATTEMPTS`를 넘지 않는다. 같은 입력은 같은 `doc_id`를 낳는다(URL 해시).

    **후보를 우선순위 순으로 두드리다가 처음 건진 곳에서 멈춘다.** 여러 페이지의
    결과를 합치지 않는 이유는 실물에서 나왔다(D64) — 당근은 `careers.daangn.com`이
    회사 채용 사이트인데 `jobs.daangn.com`은 **당근알바**(동네 구인 서비스)라, 둘을
    합치면 "김밥집 주방 직원"·"배달 라이더" 공고가 회사 채용공고 목록에 섞인다.
    후보 순서가 이미 "이게 진짜 채용 사이트일 확률" 순이므로, 합치는 것은 **덜
    믿을 만한 출처를 더 믿을 만한 출처와 같은 급으로 올리는 일**이다.

    대가도 있다 — 회사가 랜딩 페이지(`careers.x.com`)와 실제 목록(`x.com/careers`)을
    따로 두면 앞엣것에서 멈춘다. 덜 가져오는 쪽이며, 발견은 크리티컬 패스가 아니라
    그 방향이 안전하다(§6-2).
    """
    site = resolve_site(company, sites)
    if site is None or limit <= 0:
        return []

    fetch_fn = fetch or http_get
    stamp = today or date.today()

    found: dict[str, SourceDocument] = {}
    attempts = 0

    for page_url in careers_urls(site, host=live_host(site.domain, fetch_fn)):
        if attempts >= MAX_ATTEMPTS:
            break
        attempts += 1
        for job in _scrape(page_url, company, stamp, fetch_fn):
            found.setdefault(job.doc_id, job)      # 한 페이지가 같은 공고를 두 번 걸기도 한다
        if found:
            break   # **처음 건진 곳에서 멈춘다** — 아래 참조

    ranked = sorted(found.values(), key=lambda d: (-_role_overlap(d.title, role), d.title))
    return ranked[:limit]


def careers_urls(site: CompanySite, *, host: str | None = None) -> list[str]:
    """두드려 볼 채용 페이지를 우선순위 순으로 만든다 (결정론).

    `host`는 경로 후보를 매달 실제 호스트다(`live_host()`가 고른다). 서브도메인
    후보는 그 자체로 다른 호스트라 영향받지 않는다 — T17 `candidate_urls()`와
    같은 구조이며, 이유도 같다(D56).
    """
    domain = site.domain
    base = host or domain
    urls = [
        *(f"https://{sub}.{domain}/" for sub in JOB_SUBDOMAINS),
        *(f"https://{base}{path}" for path in JOB_PATHS),
    ]

    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def group_jobs(jobs: list[SourceDocument]) -> dict[str, list[SourceDocument]]:
    """공고를 대분류로 묶는다 — 순수 규칙, LLM 없음 (카드 "대분류로 그룹핑").

    묶는 것은 **보여주기 위해서**다. 그룹이 곧 선택 단위가 되면 "임의 병합"이 되므로
    선택은 끝까지 공고 하나 단위다. 그룹 순서는 `JOB_CATEGORIES` 순서를 따르고
    "기타"는 항상 마지막이다 — 목록이 실행마다 흔들리면 사용자가 자기가 뭘 봤는지
    기억하지 못한다.
    """
    grouped: dict[str, list[SourceDocument]] = {}
    for job in jobs:
        grouped.setdefault(categorize(job.title), []).append(job)

    order = [name for name, _ in JOB_CATEGORIES if name in grouped]
    if OTHER_CATEGORY in grouped:
        order.append(OTHER_CATEGORY)
    return {name: grouped[name] for name in order}


def categorize(title: str) -> str:
    """제목에서 직무 대분류를 유도한다. 어디에도 안 걸리면 "기타"."""
    haystack = title.lower()
    for name, markers in JOB_CATEGORIES:
        if any(marker in haystack for marker in markers):
            return name
    return OTHER_CATEGORY


# --- 스크레이핑 --------------------------------------------------------------


def _scrape(
    page_url: str, company: str, stamp: date, fetch: Fetcher
) -> list[SourceDocument]:
    """채용 페이지 하나에서 공고 링크를 건진다. 실패하면 빈 리스트 — 예외 없음."""
    try:
        if not robots_allows(page_url, fetch):
            return []
        links = read_links(fetch(page_url), page_url)
    except Exception:  # noqa: BLE001 — 발견 실패는 조용히 스킵 (§6-2)
        return []

    # 자기 자신으로 돌아오는 링크는 공고가 아니다 — `<a href="#">`이 조각을 떼면
    # 페이지 주소 그대로가 되고, 채용 페이지 경로는 표지를 당연히 만족한다.
    return [
        _document(url, text, company, stamp)
        for url, text in links
        if url != page_url and in_scope(url, page_url) and is_job_link(url, text)
    ]


def in_scope(url: str, page_url: str) -> bool:
    """이 링크가 **이 채용 페이지가 거느리는 공고**로 볼 범위 안인가.

    규칙은 하나다 — 채용 페이지와 **같은 호스트**이거나, 알려진 채용 ATS다.

    실물에서 나왔다(D64). `careers.daangn.com`을 긁으면 `jobs.daangn.com`으로 나가는
    링크가 잔뜩 걸리는데, 그건 **당근알바**(동네 구인 서비스)라 "김밥집 주방 직원"·
    "배달 라이더" 공고가 회사 채용공고 행세를 하며 목록에 올라온다. 같은 등록
    도메인이라 도메인 비교로는 못 거르고, 표지어로도 못 거른다(진짜 채용공고이긴 하다).

    **허용 목록으로 간 이유** — 회사의 구인 *서비스*와 회사의 *채용*을 일반 규칙으로
    가르는 방법이 없다. 차단 목록은 다음 회사에서 또 뚫린다. 반대로 "공고는 이
    채용 사이트 안이나 알려진 ATS·채용 플랫폼에 있다"는 참인 경우가 압도적이고,
    빗나가면 **덜 가져올 뿐**이다 — 발견은 크리티컬 패스가 아니라 그 방향이
    안전하다(§6-2). D55가 소프트 요건에 `_MARKERS`를 둔 것과 같은 판단이다.

    **채용 플랫폼(`PLATFORM_HOSTS`)은 T16 것을 그대로 쓴다.** 회사 채용페이지가 자기
    사람인·원티드 공고로 링크하는 것은 정상이고, 그 층은 이미 ③(신뢰도 하)으로
    정의돼 있다. 여기서 어휘를 새로 만들면 층 정의가 두 벌이 된다(D55와 같은 이유).
    """
    host = (urlparse(url).hostname or "").lower()
    page_host = (urlparse(page_url).hostname or "").lower()
    if host == page_host:
        return True

    bare = host.removeprefix("www.").removeprefix("m.")
    known = (*ATS_HOSTS, *PLATFORM_HOSTS)
    return any(bare == allowed or bare.endswith("." + allowed) for allowed in known)


def read_links(payload: str, base_url: str) -> list[tuple[str, str]]:
    """HTML에서 `(절대 URL, 링크 글자)` 목록을 뽑는다 — 순서 보존, 중복 제거.

    `tools.fetch_jd.read_html()`을 안 쓰는 이유는 그것이 **본문 텍스트**를 만드는
    파서라 `<a href>`를 버리기 때문이다. 여기서 필요한 건 정반대(링크)라서 별도
    파서를 둔다. 바이트·인코딩 계층은 여전히 T16 것이다(`http_get`).
    """
    reader = _LinkReader()
    reader.feed(payload)
    reader.close()

    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for href, text in reader.links:
        url = urljoin(base_url, href).split("#")[0]
        text = " ".join(text.split())[:MAX_TITLE_CHARS]
        if url in seen or len(text) < MIN_TITLE_CHARS:
            continue
        seen.add(url)
        links.append((url, text))
    return links


class _LinkReader(HTMLParser):
    """`<a href>`와 그 안의 글자만 모은다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        self._flush()
        href = dict(attrs).get("href")
        self._href = href.strip() if href else None

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._href:
            self.links.append((self._href, "".join(self._parts)))
        self._href = None
        self._parts = []


def is_job_link(url: str, text: str) -> bool:
    """이 링크가 **공고 상세**로 보이는가.

    경로 관례 또는 링크 글자, **둘 중 하나만** 걸려도 받는다 — 한국 대기업 CMS는
    경로가 `/kor/main.do?menuNo=...` 식이라 경로만 보면 하나도 안 걸리고, 반대로
    영문 ATS(`/jobs/4012`)는 글자가 직무명뿐이라 텍스트 표지에 안 걸린다.
    **제외 경로가 먼저다** — 로그인·공지 링크가 우연히 "채용"을 달고 있는 일이 흔하다.

    **표지는 URL 전체가 아니라 경로에만 댄다.** 전체에 대면 호스트명이 경로 표지를
    만족시켜 버린다 — `https://careers.example.com/about`은 `//careers.`가 `/career`를
    포함해서 "공고"로 둔갑한다. 채용 서브도메인이 관례라(`JOB_SUBDOMAINS`) 이 오탐은
    드문 경우가 아니라 **기본값**이었고, 첫 테스트 실행에서 `/about`이 공고로 잡혔다.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False   # javascript: · mailto: · tel: 이 여기서 걸린다

    path = parsed.path.lower()
    if parsed.query:
        path = f"{path}?{parsed.query.lower()}"

    if any(marker in path for marker in _EXCLUDE_PATHS):
        return False

    haystack = text.lower()
    return any(marker in path for marker in _JOB_HREF_MARKERS) or any(
        marker in haystack for marker in _JOB_TEXT_MARKERS
    )


def _document(url: str, title: str, company: str, stamp: date) -> SourceDocument:
    """발견된 공고 하나 — **본문은 비어 있다**(모듈 주석 참조)."""
    return SourceDocument(
        doc_id=job_doc_id(url),
        source_type=SourceType.JD,
        company=company,
        title=title,
        url=url,
        collected_at=stamp,
        raw_text="",
        confidence=_confidence(url),
    )


def job_doc_id(url: str) -> str:
    """URL이 곧 정체성이다 — 같은 공고는 몇 번을 발견해도 같은 id."""
    return f"job-{hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]}"


def _confidence(url: str) -> Confidence:
    """회사 자체 채용페이지는 중(②층), 플랫폼은 하(③층) — `CONFIDENCE_BY_LAYER`와 같은 잣대."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.").removeprefix("m.")
    is_platform = any(host == p or host.endswith("." + p) for p in PLATFORM_HOSTS)
    return Confidence.LOW if is_platform else Confidence.MID


def _role_overlap(title: str, role: str) -> int:
    """제목과 직무명이 공유하는 토큰 수 — **정렬용이며 필터가 아니다.**

    0이어도 목록에 남는다. 무엇이 관련 있는지는 사용자가 안다(§12-2 임의 선택 금지).
    """
    if not role:
        return 0
    wanted = {t.lower() for t in _TOKEN.findall(role) if len(t) > 1}
    if not wanted:
        return 0
    have = {t.lower() for t in _TOKEN.findall(title)}
    return len(wanted & have)

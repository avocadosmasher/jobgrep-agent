"""소프트 요건 수집 — 기술블로그·GitHub org / 인재상·핵심가치 (설계도 §6, §12-1).

**소프트 요건은 없어도 제품이 죽지 않는다.** JD(하드 게이트 백본)와 결정적으로 다른
점이고, 이 모듈의 설계 결정이 전부 거기서 나온다 — 실패는 **빈 리스트**로 표현하고,
예외는 한 건도 위로 올리지 않으며, 못 가져온 사실은 `gate_status["missing"]`에 **표기만**
된다. 관통 규칙은 하나다: *부분 실패를 전체 실패로 만들지 않는다.*

T16과 갈라지는 지점 — `fetch_jd_body`는 실패해도 문서 **한 건**을 돌려줘야 했다(계약
반환형이 `SourceDocument`라 그 표현뿐이었고, 그래서 `is_uncollected()`라는 판별자가
따로 필요했다). 여기는 반환형이 `list`라 **빈 리스트가 곧 미수집**이다. 판별자를 새로
만들지 않는다.

**수집 경로는 "후보 URL 열거 → 확인"이다.** 검색 엔진 연동이 이 프로젝트에 없으므로
회사명에서 도메인을 얻고(`KNOWN_SITES` 또는 호출자가 넘긴 URL·도메인), 관례적인
호스트·경로를 정해진 순서로 두드린다. 못 찾으면 조용히 빈 리스트다 — 추측한 URL이
빗나가는 것과 회사가 기술블로그를 안 하는 것은 결과가 같아야 한다.

**주워 온 게 엉뚱한 페이지면 안 가져온 것만 못하다.** 회사 사이트의 404 안내나 홈이
200으로 돌아오는 일이 흔해서, 본문에 그 유형의 표지(`_MARKERS`)가 하나도 없으면 버린다.
"근거 없이 지어내지 않는다"를 수집 단계에 적용한 것이다.

HTTP·robots·HTML 파싱·키워드는 **T16이 만든 것을 그대로 import 한다**(`tools.fetch_jd`).
같은 걸 다시 짜면 gzip 결함(D53) 같은 것이 한쪽에만 고쳐진다.

`raw_text`는 §12-5 인젝션 격리 대상이다 — 이 모듈은 **격리하지 않는다**(T26 소관).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import date
from typing import NamedTuple
from urllib.parse import urlparse

from contracts.enums import Confidence, SourceType
from contracts.models import SourceDocument
from tools.fetch_jd import (
    CONFIDENCE_BY_LAYER,
    FetchError,
    Fetcher,
    http_get,
    keywords,
    read_html,
    robots_allows,
)

MIN_SOFT_CHARS = 150      # 인재상 페이지는 JD보다 짧다 — MIN_BODY_CHARS(200)를 그대로 쓰면 다 버려진다
MAX_DOC_CHARS = 20_000    # 통짜 대형 문서 방어 (아래 `_truncate` 주석 참조)
MAX_ATTEMPTS = 8          # 한 호출이 남의 서버에 보내는 페이지 요청 상한
MAX_DOCS = 3              # 유형당 수집 상한 — 컨텍스트는 유한하고 소프트 요건은 보조다

# 회사 자체 도메인이 아닌 곳에 얹힌 블로그·발표. 층(=신뢰도)이 갈리는 유일한 기준이다.
EXTERNAL_HOSTS = frozenset(
    {
        "github.com",
        "medium.com",
        "velog.io",
        "brunch.co.kr",
        "tistory.com",
        "substack.com",
        "dev.to",
        "speakerdeck.com",
        "slideshare.net",
        "youtube.com",
    }
)

# 유형별 표지어. 하나도 없으면 그 페이지는 버린다(위 모듈 주석 참조).
# 소문자로 비교하므로 영문은 소문자로 적는다.
_MARKERS: dict[SourceType, tuple[str, ...]] = {
    SourceType.TECH_BLOG: (
        "개발", "엔지니어", "기술", "아키텍처", "오픈소스", "배포", "장애",
        "engineering", "developer", "tech", "blog", "github",
    ),
    SourceType.VALUES: (
        "인재상", "핵심가치", "핵심 가치", "가치체계", "일하는 방식", "비전", "미션",
        "문화", "인재", "values", "culture", "mission", "vision",
    ),
}

_DOC_PREFIX: dict[SourceType, str] = {
    SourceType.TECH_BLOG: "blog",
    SourceType.VALUES: "values",
}

# 도메인에서 후보를 만드는 관례. 순서가 곧 우선순위다 — 앞쪽이 더 회사 자체 페이지일 확률이 높다.
_TECH_SUBDOMAINS = ("tech", "techblog", "engineering", "blog", "developers")
_TECH_PATHS = ("/blog", "/tech")
_VALUES_SUBDOMAINS = ("careers", "recruit")
_VALUES_PATHS = ("/careers", "/recruit", "/culture", "/about", "/company/values")


class CompanySite(NamedTuple):
    """회사 하나의 수집 좌표.

    **깊은 URL은 추측해서 적지 않는다.** 도메인과 GitHub org까지가 확인 가능한 사실이고,
    그 아래 경로는 `candidate_urls()`가 관례로 만들어 두드려 본다. 확실히 아는 주소가
    있으면 `tech_blog`·`values`에 절대 URL로 넣는다 — 그 쪽이 먼저 시도된다.
    """

    domain: str
    tech_blog: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    github_org: str | None = None


def _normalize(name: str) -> str:
    """이름 대조용 정규화 — 대소문자·공백·법인 표기를 없앤다."""
    cleaned = name.strip().lower()
    for noise in ("주식회사", "(주)", "㈜", "inc.", "inc", "corp.", "corp", "ltd."):
        cleaned = cleaned.replace(noise, "")
    return "".join(cleaned.split())


def _aliases(site: CompanySite, *names: str) -> dict[str, CompanySite]:
    """별칭 여러 개를 같은 좌표에 매단다."""
    return {_normalize(name): site for name in names}


# 이름 → 좌표 **씨앗**이다. 회사 사전이 아니다 — 없는 회사는 호출자가 URL·도메인을
# 그대로 넘기거나 `sites=`로 주입하면 된다. 빗나가도 결과는 조용한 빈 리스트다.
#
# **절대 URL은 추측해서 적지 않았다.** 여기 박힌 주소는 전부 실제로 한 번 가져와
# 본문이 나오는 것만 남긴 것이다(D56) — 관례(`tech.{도메인}` 등)로 찾아지는 회사는
# 굳이 적지 않는다. 반대로 관례가 안 통하는 곳(다른 등록 도메인을 쓰는 기술블로그)은
# 여기 없으면 영영 못 찾는다.
KNOWN_SITES: dict[str, CompanySite] = {
    **_aliases(CompanySite("hyundai-autoever.com"), "현대오토에버", "hyundai autoever", "hyundaiautoever"),
    **_aliases(CompanySite("naver.com"), "네이버", "naver"),
    **_aliases(
        CompanySite("kakaocorp.com", tech_blog=("https://tech.kakao.com/",)),
        "카카오", "kakao",
    ),
    **_aliases(
        CompanySite("linecorp.com", tech_blog=("https://techblog.lycorp.co.jp/ko/",)),
        "라인", "line", "linecorp",
    ),
    **_aliases(CompanySite("coupang.com"), "쿠팡", "coupang"),
    **_aliases(CompanySite("woowahan.com"), "배달의민족", "우아한형제들", "woowahan"),
    **_aliases(
        CompanySite("toss.im", tech_blog=("https://toss.tech/",)),
        "토스", "비바리퍼블리카", "toss",
    ),
    **_aliases(CompanySite("daangn.com"), "당근", "당근마켓", "daangn"),
    **_aliases(CompanySite("samsungsds.com"), "삼성에스디에스", "삼성SDS", "samsung sds", "samsungsds"),
    **_aliases(CompanySite("lgcns.com"), "엘지씨엔에스", "LG CNS", "lgcns"),
    **_aliases(CompanySite("sktelecom.com"), "에스케이텔레콤", "SK텔레콤", "sk telecom", "sktelecom"),
}


# --- 공개 API ---------------------------------------------------------------


def fetch_tech_blog(
    company: str,
    *,
    fetch: Fetcher | None = None,
    today: date | None = None,
    sites: Mapping[str, CompanySite] | None = None,
    limit: int = MAX_DOCS,
) -> list[SourceDocument]:
    """회사의 기술 블로그·GitHub org를 수집해 `SourceDocument` 목록으로 돌려준다.

    입력: 회사명 — 도메인("example.com")이나 URL을 그대로 넘겨도 된다.
        fetch·today·sites는 주입점이며 평소엔 None.
    출력: `SourceDocument(source_type=TECH_BLOG)` 목록. **못 가져오면 빈 리스트다.**
    불변식: 어떤 실패도 예외로 올리지 않는다. robots.txt를 준수한다. 페이지 요청은
        `MAX_ATTEMPTS`를 넘지 않는다.
    """
    return _collect(company, SourceType.TECH_BLOG, fetch, today, sites, limit)


def get_company_values(
    company: str,
    *,
    fetch: Fetcher | None = None,
    today: date | None = None,
    sites: Mapping[str, CompanySite] | None = None,
    limit: int = MAX_DOCS,
) -> list[SourceDocument]:
    """회사의 인재상·핵심가치 페이지를 수집해 `SourceDocument` 목록으로 돌려준다.

    `fetch_tech_blog`와 규약이 같다 — 실패는 빈 리스트, 예외 전파 없음, robots 준수.
    """
    return _collect(company, SourceType.VALUES, fetch, today, sites, limit)


def missing_soft_sources(
    tech_blog: list[SourceDocument],
    values: list[SourceDocument],
) -> list[str]:
    """빈 쪽의 라벨을 돌려준다 — 상류가 `gate_status["missing"]`에 그대로 넣는다.

    **라벨 어휘를 새로 만들지 않고 `SourceType`의 값을 쓴다.** `nodes.analysis_nodes`의
    `source_coverage()`가 이미 같은 어휘로 `BriefMeta.missing_sources`를 채우고 있어서,
    여기서 "기술블로그 미수집" 같은 문구를 지어내면 화면에 두 가지 표기가 섞인다.

    수집 결과에서 라벨을 유도하므로 호출자가 따로 기억할 상태가 없다 — 카드 불변식
    ("실패 시 빈 리스트 + missing 항목 추가")이 이 한 줄로 닫힌다.
    """
    labels = []
    if not tech_blog:
        labels.append(SourceType.TECH_BLOG.value)
    if not values:
        labels.append(SourceType.VALUES.value)
    return labels


def resolve_site(
    company: str,
    sites: Mapping[str, CompanySite] | None = None,
) -> CompanySite | None:
    """회사명·도메인·URL을 수집 좌표로 바꾼다. 못 알아보면 None(= 조용한 미수집).

    URL·도메인을 **먼저** 본다. 호출자가 주소를 알고 넘겼다면 그게 사전보다 정확하다.
    """
    raw = (company or "").strip()
    if not raw:
        return None

    domain = _domain_of(raw)
    if domain:
        return CompanySite(domain)

    table = KNOWN_SITES if sites is None else sites
    return table.get(_normalize(raw))


def candidate_urls(
    site: CompanySite,
    kind: SourceType,
    *,
    host: str | None = None,
) -> list[str]:
    """두드려 볼 URL을 우선순위 순서로 만든다 (결정론 — 같은 입력이면 같은 목록).

    확실히 아는 절대 URL → 회사 자체 서브도메인 → 회사 도메인 경로 → 외부 호스팅 순이다.
    **신뢰도가 높은 순서와 같다.** 상한(`MAX_ATTEMPTS`)에 잘리면 뒤쪽이 버려지므로,
    잘릴 때 버려지는 게 덜 믿을 만한 쪽이 되도록 이 순서로 둔다.

    `host`는 경로 후보를 매달 실제 호스트다(`live_host()`가 고른다). 서브도메인 후보는
    영향을 받지 않는다 — `tech.example.com`은 그 자체로 다른 호스트다.
    """
    domain = site.domain
    base = host or domain
    if kind is SourceType.TECH_BLOG:
        urls = [
            *site.tech_blog,
            *(f"https://{sub}.{domain}/" for sub in _TECH_SUBDOMAINS),
            *(f"https://{base}{path}" for path in _TECH_PATHS),
        ]
        if site.github_org:
            urls.append(f"https://github.com/{site.github_org}")
        return _dedupe(urls)

    return _dedupe(
        [
            *site.values,
            *(f"https://{sub}.{domain}/" for sub in _VALUES_SUBDOMAINS),
            *(f"https://{base}{path}" for path in _VALUES_PATHS),
        ]
    )


def live_host(domain: str, fetch: Fetcher) -> str:
    """`example.com`과 `www.example.com` 중 **응답하는 쪽**을 고른다.

    실 URL 확인에서 나왔다 — `hyundai-autoever.com`은 인증서 호스트명이 안 맞아 TLS에서
    죽고 `www.hyundai-autoever.com`만 산다. 한국 대기업 사이트에 흔한 모양이라, apex만
    두드리면 그 회사는 통째로 미수집이 된다. 반대로 apex만 서비스하는 곳도 있어서 어느
    한쪽으로 못 박을 수 없다.

    **판별은 robots.txt 응답으로 한다.** 어차피 후보마다 읽어야 하는 파일이고
    `_RobotsCache`가 그 결과를 재사용하므로 **요청이 늘지 않는다.** 404 같은 HTTP 상태가
    돌아왔다면 서버는 살아 있는 것이니 그 호스트를 쓰고, 상태 없는 실패(DNS·TLS·타임아웃)만
    "이 호스트는 없다"로 읽는다. 둘 다 실패하면 원래 도메인을 돌려준다 — 뒤 후보들이
    조용히 실패할 뿐이라 결과는 같다.
    """
    for host in (domain, f"www.{domain}"):
        try:
            fetch(f"https://{host}/robots.txt")
            return host
        except FetchError as exc:
            if exc.status is not None:
                return host          # 서버는 응답했다 (robots가 없을 뿐)
        except Exception:  # noqa: BLE001 — 주입된 fetcher가 뭘 던지든 다음 후보로
            continue
    return domain


# --- 수집 루프 --------------------------------------------------------------


def _collect(
    company: str,
    kind: SourceType,
    fetch: Fetcher | None,
    today: date | None,
    sites: Mapping[str, CompanySite] | None,
    limit: int,
) -> list[SourceDocument]:
    site = resolve_site(company, sites)
    if site is None or limit <= 0:
        return []      # 아무것도 안 받을 거면 `live_host` 탐색부터 하지 않는다

    fetch_fn = _RobotsCache(fetch or http_get)
    stamp = today or date.today()

    docs: list[SourceDocument] = []
    seen: set[str] = set()
    attempts = 0

    for url in candidate_urls(site, kind, host=live_host(site.domain, fetch_fn)):
        if len(docs) >= limit or attempts >= MAX_ATTEMPTS:
            break
        attempts += 1
        doc = _try_one(url, kind, company, site, stamp, fetch_fn)
        if doc is None or doc.doc_id in seen:
            continue          # 같은 본문이 두 경로에서 나오는 건 흔하다 (/about == /company/values)
        seen.add(doc.doc_id)
        docs.append(doc)

    return docs


def _try_one(
    url: str,
    kind: SourceType,
    company: str,
    site: CompanySite,
    stamp: date,
    fetch: Fetcher,
) -> SourceDocument | None:
    """후보 하나를 확인한다. 못 건지면 None — **어떤 이유로든 예외를 내지 않는다.**

    robots 차단·404·본문 부족·표지어 없음·주입된 fetcher의 생예외가 전부 같은 결과로
    수렴한다. 소프트 요건에서 실패 사유를 구분해 봐야 상류가 할 일이 달라지지 않고,
    구분하려면 실패를 예외로 표현해야 하는데 그러면 카드 불변식과 정면으로 부딪친다.
    """
    try:
        if not robots_allows(url, fetch):
            return None
        page = read_html(fetch(url))
        text = _truncate(page.text)
        if len(text) < MIN_SOFT_CHARS or not _is_relevant(kind, page.title, text):
            return None
        return SourceDocument(
            doc_id=f"{_DOC_PREFIX[kind]}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}",
            source_type=kind,
            company=_company_name(company, page.site_name, url),
            title=page.title or _first_line(text),
            url=url,
            published_at=page.published_at,
            collected_at=stamp,
            raw_text=text,
            keywords=keywords(text),
            confidence=_confidence(url, site),
        )
    except Exception:  # noqa: BLE001 — 소프트 요건 실패는 조용히 스킵 (카드 불변식)
        return None


class _RobotsCache:
    """호출 한 번 동안 robots.txt 응답을 호스트별로 재사용하는 얇은 껍질.

    후보를 8개까지 두드리는데 그 대부분이 같은 호스트라서, 없으면 robots.txt를 같은
    서버에 여러 번 물어보게 된다. 준수하려고 읽는 파일로 서버를 더 두드리는 건 앞뒤가
    안 맞는다. **실패도 캐시한다** — 재시도하면 그것대로 요청이 는다.
    """

    def __init__(self, fetch: Fetcher) -> None:
        self._fetch = fetch
        self._cache: dict[str, str | BaseException] = {}

    def __call__(self, url: str) -> str:
        if not url.endswith("/robots.txt"):
            return self._fetch(url)
        if url not in self._cache:
            try:
                self._cache[url] = self._fetch(url)
            except BaseException as exc:  # noqa: BLE001 — 그대로 다시 던져 판정은 T16에 맡긴다
                self._cache[url] = exc
        cached = self._cache[url]
        if isinstance(cached, BaseException):
            raise cached
        return cached


# --- 판별 ------------------------------------------------------------------


def _is_relevant(kind: SourceType, title: str, text: str) -> bool:
    """이 페이지가 그 유형이 맞다고 볼 표지가 하나라도 있는가."""
    haystack = f"{title}\n{text}".lower()
    return any(marker in haystack for marker in _MARKERS[kind])


def _confidence(url: str, site: CompanySite) -> Confidence:
    """회사 자체 채널이면 중(②층), 외부 호스팅이면 하(③층).

    "자체 채널"은 회사 도메인 **또는 레지스트리에 손으로 확인해 박아 둔 주소**다. 후자를
    같이 쳐 주는 이유는 기술블로그가 다른 등록 도메인에 사는 일이 흔해서다(`toss.im` →
    `toss.tech`). 관례로 찾은 것과 달리 그건 사람이 확인한 주소다.

    층 → 신뢰도 매핑은 `CONFIDENCE_BY_LAYER` 한 곳에만 있다(DEVLOG D52) — 여기서 다시
    정의하지 않는다. 소프트 요건에 **상은 없다**. 상은 사용자가 직접 붙여넣은 원문의
    자리이고, 긁어 온 페이지가 그 자리를 차지하면 신뢰등급이 헐거워진다.
    """
    host = _hostname(url)
    own = host == site.domain or host.endswith("." + site.domain)
    curated = url in site.tech_blog or url in site.values
    external = any(host == e or host.endswith("." + e) for e in EXTERNAL_HOSTS)
    return CONFIDENCE_BY_LAYER[2 if (own or curated) and not external else 3]


def _truncate(text: str) -> str:
    """상한을 넘으면 줄 경계에서 자른다.

    카드는 통짜 대형 문서(지속가능경영보고서 같은)에 **그때만** RAG 청킹을 붙이라고
    했다. 지금 필요한 건 청킹이 아니라 컨텍스트를 통째로 날리지 않는 것이고, 소프트
    요건은 보조 근거라 앞부분으로 충분하다. **잘린 자리에 표식을 넣지 않는다** — 원문에
    없는 문장이 `raw_text`에 섞이면 evidence.quote 대조(§7)가 그걸 인용할 수 있다.
    """
    if len(text) <= MAX_DOC_CHARS:
        return text
    head = text[:MAX_DOC_CHARS]
    cut = head.rfind("\n")
    return head[:cut] if cut > 0 else head


# --- 잡동사니 --------------------------------------------------------------


def _domain_of(value: str) -> str | None:
    """URL이나 도메인처럼 생겼으면 호스트를, 아니면 None."""
    if any(ch.isspace() for ch in value):
        return None
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https"):
            return None
        return _strip_www(parsed.hostname or "") or None
    if "." in value and "/" not in value and value.rpartition(".")[2].isalpha():
        return _strip_www(value.lower())
    return None


def _hostname(url: str) -> str:
    return _strip_www(urlparse(url).hostname or "")


def _strip_www(host: str) -> str:
    return host.lower().removeprefix("www.")


def _company_name(company: str, site_name: str, url: str) -> str:
    """호출자가 준 이름이 정본. 주소를 넘긴 경우엔 페이지 메타·호스트로 메운다."""
    if _domain_of(company) is None and company.strip():
        return company.strip()
    return site_name or _hostname(url)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:80]
    return ""


def _dedupe(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]

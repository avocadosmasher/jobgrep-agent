"""JD 본문 3층 폴백 (설계도 §6-1, §6-2).

세 층은 **난이도가 아니라 성격이 다르다.**

| 층 | 경로 | 성격 | confidence |
| --- | --- | --- | --- |
| ① | 사용자 붙여넣기 | 하드 게이트 백본 — 항상 동작해야 함 | 상 |
| ② | 회사 자체 채용페이지 | 기본 증강 | 중 |
| ③ | 플랫폼 상세페이지 | best-effort — 제품이 의존하지 않음 | 하 |

**①은 네트워크를 아예 타지 않는다.** 그래서 ②③이 무슨 이유로 죽든 백본은 멀쩡하다 —
카드의 완료 조건이 이걸 요구하고, `test_pasted_body_never_touches_the_network`가 호출
기록으로 증명한다. 반대로 ②③은 **어떤 예외도 위로 올리지 않는다.** robots.txt가 막든,
404가 뜨든, 주입된 fetcher가 엉뚱한 예외를 던지든 결과는 하나 — 빈 본문 + 신뢰도 하의
`SourceDocument`다(`is_uncollected()`가 참). 수집 실패를 예외로 표현하면 best-effort라는
말과 모순되고, 상류(T18 수집 루프)가 매번 try/except로 감싸야 한다.

**층 판별은 URL의 호스트로 한다** — 카드 표의 ②와 ③을 한 인자(`url_or_text`)로 가르는
유일한 결정론적 수단이다. 알려진 채용 플랫폼이면 ③, 그 밖이면 회사 자체 페이지로 보고 ②.
설계도가 ①에 "지정 URL"을 포함시킨 것은 **발견 과정을 안 거쳤다**는 뜻이지 스크래핑한
본문이 전문임을 보장한다는 뜻이 아니므로, 가져온 본문은 상이 아니라 중/하를 준다
(DEVLOG D52).

`raw_text`는 §12-5 인젝션 격리 대상이다 — 이 모듈은 **격리하지 않는다**(T26 소관).
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections.abc import Callable
from datetime import date
from html.parser import HTMLParser
from urllib import robotparser
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from contracts.enums import Confidence, SourceType
from contracts.models import SourceDocument

# 층 → 신뢰도. 카드 불변식 "어느 층에서 왔는지 confidence에 반영"의 유일한 정의 지점이다.
CONFIDENCE_BY_LAYER: dict[int, Confidence] = {
    1: Confidence.HIGH,
    2: Confidence.MID,
    3: Confidence.LOW,
}

USER_AGENT = "jobprep-agent/0.1 (job application helper; contact: user)"
TIMEOUT_SECONDS = 10.0
MAX_BYTES = 2 * 1024 * 1024        # 통짜 대형 문서 방어 (§6-3)
MIN_BODY_CHARS = 200               # 이보다 짧으면 본문을 못 건진 것으로 본다
DEFAULT_KEYWORD_LIMIT = 12

UNCOLLECTED_TITLE = "본문 미수집 — 직접 붙여넣어 주세요"

# 알려진 채용 플랫폼 = ③층. 서브도메인까지 걸리도록 접미사로 비교한다.
PLATFORM_HOSTS = frozenset(
    {
        "saramin.co.kr",
        "jobkorea.co.kr",
        "wanted.co.kr",
        "incruit.com",
        "jumpit.co.kr",
        "programmers.co.kr",
        "career.programmers.co.kr",
        "rocketpunch.com",
        "jasoseol.com",
        "catch.co.kr",
        "worknet.go.kr",
        "linkedin.com",
        "indeed.com",
        "glassdoor.com",
    }
)

# URL을 넘겨 본문 문자열을 돌려주는 무언가. 실패는 예외로 알린다(가급적 `FetchError`).
Fetcher = Callable[[str], str]


class FetchError(RuntimeError):
    """HTTP 실패. `status`가 있으면 robots.txt 처리 규칙(RFC 9309)이 그걸 본다."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# --- 공개 API ---------------------------------------------------------------


def fetch_jd_body(
    url_or_text: str,
    *,
    company: str = "",
    department: str | None = None,
    title: str | None = None,
    fetch: Fetcher | None = None,
    today: date | None = None,
) -> SourceDocument:
    """JD 원문을 URL 또는 붙여넣은 텍스트로부터 `SourceDocument`로 정규화한다.

    입력: JD URL 또는 JD 본문 텍스트. company/department/title은 **호출자가 아는 값이
        정본**이라 넘어오면 문서에서 파싱한 값보다 우선한다(빈 값이면 파싱값 사용).
        fetch·today는 주입점이며 평소엔 None.
    출력: `SourceDocument(source_type=JD)`. 층에 따라 confidence가 상/중/하로 갈리고,
        수집에 실패하면 `raw_text=""` + 하 (예외 아님). `is_uncollected()`로 판별한다.
    불변식: ① 경로는 네트워크를 타지 않는다. ②③ 실패는 조용히 스킵하며 예외를 위로
        전파하지 않는다. robots.txt를 준수한다. collected_at·confidence·doc_id·keywords는
        코드가 채운다.

    같은 입력은 같은 `doc_id`를 낳는다(수집일만 다를 뿐 멱등).
    """
    source = (url_or_text or "").strip()
    stamp = today or date.today()

    if not source:
        return _uncollected(None, "입력 없음", stamp, company, department)

    if not _is_http_url(source):
        # ① 백본 — 붙여넣은 전문은 한 글자도 손대지 않는다(법적 무결·전문 보장).
        return _document(
            layer=1,
            key=source,
            raw_text=source,
            url=None,
            company=company,
            department=department,
            title=title or _first_line(source),
            published_at=None,
            stamp=stamp,
        )

    layer = 3 if _is_platform(source) else 2
    fetch_fn = fetch or http_get

    if not robots_allows(source, fetch_fn):
        return _uncollected(source, "robots.txt 차단", stamp, company, department)

    try:
        payload = fetch_fn(source)
    except Exception as exc:  # noqa: BLE001 — ②③ 실패는 조용히 스킵(카드 불변식)
        return _uncollected(source, f"수집 실패: {type(exc).__name__}", stamp, company, department)

    page = read_html(payload)
    if len(page.text) < MIN_BODY_CHARS:
        return _uncollected(source, "본문이 너무 짧음", stamp, company, department)

    return _document(
        layer=layer,
        key=source,
        raw_text=page.text,
        url=source,
        company=company or page.site_name,
        department=department,
        title=title or page.title or _first_line(page.text),
        published_at=page.published_at,
        stamp=stamp,
    )


def is_uncollected(doc: SourceDocument) -> bool:
    """본문을 못 건진 문서인가.

    수집 실패를 예외가 아니라 **빈 본문**으로 표현하기 때문에 판별자가 필요하다.
    하드 게이트(T25)는 JD **건수**가 아니라 이 함수가 거짓인 문서의 수를 세야 한다 —
    빈 문서를 1건으로 세면 게이트가 헛통과한다.
    """
    return not doc.raw_text.strip()


# --- 층 판별 ----------------------------------------------------------------


def _is_http_url(value: str) -> bool:
    """통째로 하나의 http(s) URL인가.

    본문 안에 URL이 섞여 있는 붙여넣기를 URL로 오인하면 백본이 통째로 날아가므로,
    **공백이 하나라도 있으면 텍스트**로 본다. file:·javascript: 등 다른 스킴도 텍스트다
    (네트워크를 타지 않는 쪽이 안전하다).
    """
    if not value or any(ch.isspace() for ch in value):
        return False
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _is_platform(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.").removeprefix("m.")
    return any(host == p or host.endswith("." + p) for p in PLATFORM_HOSTS)


# --- robots.txt -------------------------------------------------------------


def robots_allows(url: str, fetch: Fetcher, user_agent: str = USER_AGENT) -> bool:
    """robots.txt를 같은 호스트 루트에서 읽어 이 URL을 가져와도 되는지 본다.

    가져오지 못했을 때의 처리는 RFC 9309 §2.3.1을 따른다 — 401·403은 **전면 금지**,
    그 밖의 4xx는 전면 허용(파일이 없는 정상 상태), 5xx·네트워크 오류는 알 수 없으므로
    금지. 상태 코드를 모르는 예외(주입된 fetcher 등)도 같은 이유로 금지 쪽이다.
    """
    parsed = urlparse(url)
    try:
        body = fetch(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    except FetchError as exc:
        status = exc.status
        return status is not None and 400 <= status < 500 and status not in (401, 403)
    except Exception:  # noqa: BLE001 — 알 수 없으면 보수적으로 금지
        return False

    parser = robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    return parser.can_fetch(user_agent, url)


# --- HTTP -------------------------------------------------------------------


def http_get(url: str, *, timeout: float = TIMEOUT_SECONDS) -> str:
    """기본 fetcher. 새 의존성을 들이지 않으려고 표준 라이브러리만 쓴다(DEVLOG D52).

    `br`(brotli)는 표준 라이브러리로 못 풀어서 **아예 요청하지 않는다** — 처리 못 할
    인코딩을 광고하면 서버가 그걸로 준다.
    """
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "ko,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, identity",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 — 스킴은 위에서 검사
            return decode_body(response.read(MAX_BYTES), response.headers)
    except HTTPError as exc:
        raise FetchError(f"HTTP {exc.code}", status=exc.code) from exc
    except (URLError, OSError, ValueError) as exc:
        raise FetchError(str(exc)) from exc


def decode_body(raw: bytes, headers) -> str:
    """압축을 풀고 문자셋을 정해 디코드한다.

    **이 함수는 실 HTTP 왕복 한 번에서 나왔다.** 오프라인 테스트는 fetcher를 갈아끼우니
    바이트 계층을 아예 안 지나가고, 그래서 gzip 응답을 그대로 utf-8로 읽어 본문이 통째로
    깨지는 걸 66건 초록불이 하나도 못 잡았다(D53 — D41·D51이 다른 층에서 한 말과 같다).

    `Content-Encoding`이 없어도 gzip 매직 바이트가 보이면 푼다 — CDN이 헤더를 흘리는
    경우가 실제로 있다. 상한(`MAX_BYTES`)에 잘린 스트림도 `decompressobj`가 있는 만큼
    돌려주므로 조각이라도 건진다.
    """
    encoding = (headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding or raw[:2] == b"\x1f\x8b":
        raw = _inflate(raw, 16 + zlib.MAX_WBITS)
    elif "deflate" in encoding:
        raw = _inflate(raw, zlib.MAX_WBITS) or _inflate(raw, -zlib.MAX_WBITS)

    charset = headers.get_content_charset() or _charset_from_meta(raw) or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:                       # 서버가 없는 인코딩 이름을 부른 경우
        return raw.decode("utf-8", errors="replace")


def _inflate(raw: bytes, wbits: int) -> bytes:
    try:
        return zlib.decompressobj(wbits).decompress(raw)
    except zlib.error:
        return b""


def _charset_from_meta(raw: bytes) -> str | None:
    """헤더에 문자셋이 없을 때 `<meta charset>`을 본다 — 한국어 사이트는 euc-kr이 흔하다."""
    match = re.search(rb"""charset=["']?\s*([\w-]+)""", raw[:4096], re.IGNORECASE)
    return match.group(1).decode("ascii", errors="replace") if match else None


# --- HTML → 텍스트 ----------------------------------------------------------

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "iframe"})
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "hr", "li", "ul", "ol", "dl", "dt", "dd",
        "table", "tr", "td", "th", "thead", "tbody",
        "section", "article", "header", "footer", "nav", "aside", "main",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "form",
    }
)
_PUBLISHED_KEYS = (
    "article:published_time",
    "og:published_time",
    "datepublished",
    "pubdate",
    "date",
)


class Page:
    """HTML에서 결정론적으로 건진 것들."""

    def __init__(self, text: str, title: str, site_name: str, published_at: date | None) -> None:
        self.text = text
        self.title = title
        self.site_name = site_name
        self.published_at = published_at


class _HtmlReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self.title_parts: list[str] = []
        self.meta: dict[str, str] = {}
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "meta":
            values = dict(attrs)
            key = (values.get("property") or values.get("name") or "").strip().lower()
            content = (values.get("content") or "").strip()
            if key and content and key not in self.meta:
                self.meta[key] = content     # 먼저 나온 것이 이긴다
            return
        if tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        self.chunks.append(data)


def read_html(payload: str) -> Page:
    """HTML(또는 평문)에서 본문·제목·메타를 뽑는다.

    태그가 없으면 평문으로 취급해 그대로 돌려준다 — 서버가 `text/plain`을 주는 경우다.
    본문 추출 품질은 ②③이 best-effort라서 여기까지가 적정선이다(DEVLOG D52).
    """
    reader = _HtmlReader()
    reader.feed(payload)
    reader.close()

    text = _collapse("".join(reader.chunks))
    title = " ".join("".join(reader.title_parts).split())
    return Page(
        text=text,
        title=title or reader.meta.get("og:title", ""),
        site_name=reader.meta.get("og:site_name", ""),
        published_at=_parse_published(reader.meta),
    )


def _collapse(text: str) -> str:
    """줄 안의 공백은 하나로, 빈 줄은 통째로 버린다.

    HTML은 여는 태그·닫는 태그가 각각 줄바꿈을 내므로 빈 줄을 남기면 `<li>` 사이마다
    빈 줄이 끼는 등 태그 구조가 그대로 새어 나온다. **블록 하나 = 한 줄**로 못 박으면
    출력이 마크업 형태와 무관해진다. 붙여넣기(①)는 이 함수를 타지 않으므로 사용자
    원문의 문단 구조는 그대로다.
    """
    lines = (re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _parse_published(meta: dict[str, str]) -> date | None:
    for key in _PUBLISHED_KEYS:
        value = meta.get(key)
        if not value:
            continue
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            continue     # 파싱 안 되면 없는 것으로 — 지어내지 않는다
    return None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "JD"


# --- 키워드 -----------------------------------------------------------------

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#._-]*")
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could do does for from had has have
    if in into is it its may might must not of on or our ours out over own same shall
    should so such than that the their them then there these they this those to too
    up was we were what when where which while who whom why will with would you your
    etc via per
    """.split()
)


def keywords(text: str, limit: int = DEFAULT_KEYWORD_LIMIT) -> list[str]:
    """결정론적 키워드 추출 — 빈도 내림차순, 동률이면 먼저 나온 순.

    **라틴 문자 토큰만 본다.** 한국어는 형태소 분석기 없이는 경계를 못 잘라 조사가 붙은
    쓰레기가 섞이는데, 그 사전을 들이는 값이 지금은 없다(하류에서 `keywords`를 쓰는
    모듈이 아직 없다 — 역량 추출은 `raw_text`를 본다). 표기는 처음 나온 형태를 쓴다.
    """
    if limit <= 0:
        return []

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    surface: dict[str, str] = {}

    for position, match in enumerate(_TOKEN.finditer(text)):
        token = match.group().rstrip("._-")
        if len(token) < 2:
            continue
        key = token.lower()
        if key in _STOPWORDS:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = position
            surface[key] = token

    ranked = sorted(counts, key=lambda k: (-counts[k], first_seen[k]))
    return [surface[k] for k in ranked[:limit]]


# --- 문서 조립 --------------------------------------------------------------


def _document(
    *,
    layer: int,
    key: str,
    raw_text: str,
    url: str | None,
    company: str,
    department: str | None,
    title: str,
    published_at: date | None,
    stamp: date,
) -> SourceDocument:
    return SourceDocument(
        doc_id=_doc_id(key),
        source_type=SourceType.JD,
        company=company,
        department=department,
        title=title,
        url=url,
        published_at=published_at,
        collected_at=stamp,
        raw_text=raw_text,
        keywords=keywords(raw_text),
        confidence=CONFIDENCE_BY_LAYER[layer],
    )


def _uncollected(
    url: str | None,
    reason: str,
    stamp: date,
    company: str,
    department: str | None,
) -> SourceDocument:
    """수집 실패 문서 — 예외 대신 이걸 돌려준다. 사유는 사람이 읽도록 제목에 싣는다."""
    return _document(
        layer=3,
        key=url or "",
        raw_text="",
        url=url,
        company=company,
        department=department,
        title=f"{UNCOLLECTED_TITLE} ({reason})",
        published_at=None,
        stamp=stamp,
    )


def _doc_id(key: str) -> str:
    return "jd-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

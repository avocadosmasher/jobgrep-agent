"""T21 · parse_resume 계층화.

완료 조건은 두 문장이다 — "텍스트 PDF는 **OCR 없이** 추출, 스캔 PDF는 **OCR 경로로
분기**". 둘 다 결과 텍스트만 봐서는 증명되지 않는다(OCR을 태우고도 같은 텍스트가
나올 수 있다). 그래서 **OCR 호출 횟수를 센다** — T12가 재개를 호출 횟수로 증명한
것과 같은 방식이다.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from contracts.enums import Confidence
from tools.parse_resume import (
    CONFIDENCE_BY_LAYER,
    LAYER_FAILED,
    LAYER_OCR,
    LAYER_TEXT,
    MIN_CHARS,
    MIN_READABLE_CHARS,
    count_hangul,
    count_latin,
    needs_manual_correction,
    parse_resume,
    text_quality_ok,
    whitespace_ratio,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
TEXT_PDF = FIXTURES / "resume_text.pdf"
SCAN_PDF = FIXTURES / "resume_scan.pdf"

OCR_TEXT = (
    "김도현 backend platform engineer 넥스트커머스 플랫폼팀 시니어 엔지니어 "
    "쿠버네티스 기반 사내 배포 플랫폼을 구축했고 배포 리드타임을 40분에서 "
    "6분으로 줄였다. 결제 정산 배치의 분산 트랜잭션을 재작성해 중복 정산 "
    "사고를 분기 3건에서 0건으로 만들었다. 관측 스택을 도입하고 서비스별 "
    "목표를 정의했다. 한국대학교 컴퓨터공학과 학사. 기술 블로그 연재 4편. "
    "사내 해커톤 대상 수상 이력이 있으며 로그 파이프라인 비용을 60% 줄였다."
)


# --------------------------------------------------------------------------
# OCR 이음매 — T22가 아직 없으므로 가짜 tools.ocr을 sys.modules에 꽂는다.
# 스텁이 아니라 "T22가 놓일 자리"를 그대로 흉내낸 것이라 R5에 걸리지 않는다:
# 검증 대상은 OCR의 품질이 아니라 **분기가 일어나는가**이다.
# --------------------------------------------------------------------------
@pytest.fixture
def ocr_calls(monkeypatch):
    """가짜 OCR을 설치하고 호출 인자를 모아 돌려준다."""
    calls: list[str] = []

    def _install(return_value: str = OCR_TEXT, raises: BaseException | None = None):
        module = types.ModuleType("tools.ocr")

        def ocr_document(file_path: str) -> str:
            calls.append(file_path)
            if raises is not None:
                raise raises
            return return_value

        module.ocr_document = ocr_document
        monkeypatch.setitem(sys.modules, "tools.ocr", module)
        return calls

    return _install


@pytest.fixture(autouse=True)
def no_stale_ocr(monkeypatch):
    """이 파일의 기본 상태는 "OCR 모듈이 없다"이다.

    **T22가 `tools/ocr.py`를 실제로 만들면서 이 픽스처의 방식이 바뀌었다.** 전에는
    `sys.modules`에서 지우기만 하면 됐지만, 이제 파일이 존재하므로 지우면 오히려 진짜
    모듈이 import돼 **실 vision API를 왕복한다**(§4 위반 + 조용한 과금). `None`을
    꽂아 두면 import 자체가 `ImportError`로 떨어져 T21이 의도한 "폴백 부재" 상태가
    정확히 재현된다. OCR이 있는 경우는 `ocr_calls` 픽스처가 따로 꽂는다.

    D59("배선 한 줄이 남의 테스트를 깬다")·D66(스텁 누수)의 같은 패턴이라 T22가 고쳤다.
    """
    monkeypatch.setitem(sys.modules, "tools.ocr", None)


# --------------------------------------------------------------------------
# 픽스처 자체가 전제를 만족하는지 — 여기가 깨지면 아래 전부가 무의미하다
# --------------------------------------------------------------------------
def test_픽스처_두_개가_존재한다():
    assert TEXT_PDF.exists(), "fixtures/resume_text.pdf 없음"
    assert SCAN_PDF.exists(), "fixtures/resume_scan.pdf 없음"


def test_스캔_픽스처는_진짜로_텍스트_레이어가_없다():
    """R5 — 스캔본을 흉내만 낸 파일이면 이 카드의 검증이 통째로 가짜가 된다."""
    pymupdf = pytest.importorskip("pymupdf")
    with pymupdf.open(SCAN_PDF) as doc:
        extracted = "".join(page.get_text() for page in doc)
    assert extracted.strip() == ""


def test_텍스트_픽스처는_한글이_왕복한다():
    """CJK 폰트가 깨져 나오면 품질 검사가 잡아야 할 것을 못 잡는다."""
    pymupdf = pytest.importorskip("pymupdf")
    with pymupdf.open(TEXT_PDF) as doc:
        extracted = "".join(page.get_text() for page in doc)
    assert count_hangul(extracted) > 300
    assert "김도현" in extracted


def test_픽스처는_두_쪽짜리다():
    """이력서는 한 장을 넘어간다 — 한 쪽짜리 픽스처는 페이지 회귀를 못 잡는다."""
    pymupdf = pytest.importorskip("pymupdf")
    for path in (TEXT_PDF, SCAN_PDF):
        with pymupdf.open(path) as doc:
            assert doc.page_count == 2, f"{path.name}이 두 쪽이 아니다"


# --------------------------------------------------------------------------
# 완료 조건 ① — 텍스트 PDF는 OCR 없이 추출된다
# --------------------------------------------------------------------------
def test_텍스트_PDF는_OCR을_한_번도_부르지_않는다(ocr_calls):
    calls = ocr_calls()  # OCR이 있어도 태우지 않아야 한다
    text, confidence = parse_resume(str(TEXT_PDF))

    assert calls == [], "텍스트 레이어가 멀쩡한데 OCR을 태웠다"
    assert confidence is Confidence.HIGH
    assert "Kubernetes" in text
    assert "배포 리드타임" in text


def test_두_번째_쪽까지_전부_추출한다():
    """첫 쪽만 읽으면 학력·기타가 통째로 사라지는데 분량은 여전히 임계를 넘어
    품질 검사가 초록불을 준다 — 조용한 데이터 손실이라 따로 걸어 둔다."""
    text, _ = parse_resume(str(TEXT_PDF))

    assert "넥스트커머스" in text, "1쪽이 비었다"
    for page2_only in ("OpenTelemetry", "한국대학교", "해커톤"):
        assert page2_only in text, f"2쪽의 {page2_only}이(가) 사라졌다"


def test_텍스트_PDF는_OCR_모듈이_없어도_성공한다():
    """T22 없이 T21이 완결이라는 뜻 — 폴백 모듈 부재가 실패가 되면 안 된다."""
    text, confidence = parse_resume(str(TEXT_PDF))
    assert confidence is Confidence.HIGH
    assert len(text) > MIN_CHARS
    assert not needs_manual_correction(text, confidence)


# --------------------------------------------------------------------------
# 완료 조건 ② — 스캔 PDF는 OCR 경로로 분기된다
# --------------------------------------------------------------------------
def test_스캔_PDF는_OCR_경로로_분기한다(ocr_calls):
    calls = ocr_calls()
    text, confidence = parse_resume(str(SCAN_PDF))

    assert calls == [str(SCAN_PDF)], "스캔본인데 OCR로 분기하지 않았다"
    assert confidence is Confidence.MID, "OCR 경로는 신뢰도가 한 단계 낮아야 한다"
    assert text == OCR_TEXT


def test_스캔_PDF는_OCR이_없으면_하_신뢰도로_H4에_넘긴다():
    text, confidence = parse_resume(str(SCAN_PDF))

    assert confidence is Confidence.LOW
    assert needs_manual_correction(text, confidence)


def test_이미지는_텍스트_레이어를_시도하지_않고_OCR로_직행한다(ocr_calls, tmp_path):
    calls = ocr_calls()
    # 내용은 볼 일이 없다 — 확장자만으로 3)으로 가야 한다
    image = tmp_path / "resume.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n not really a png")

    text, confidence = parse_resume(str(image))

    assert calls == [str(image)]
    assert confidence is Confidence.MID
    assert text == OCR_TEXT


def test_이미지는_텍스트_레이어_추출기를_아예_부르지_않는다(ocr_calls, tmp_path, monkeypatch):
    """결과만 보면 구분이 안 된다 — PDF 추출기에 이미지를 넣어도 빈 문자열이
    나와 어차피 OCR로 흘러가기 때문이다. 그래서 **호출 자체**를 본다."""
    import tools.parse_resume as parse_resume_module

    calls = ocr_calls()
    touched: list[str] = []
    monkeypatch.setattr(
        parse_resume_module,
        "_extract_pdf_text_layer",
        lambda path: (touched.append(path), "")[1],
    )

    image = tmp_path / "resume.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n not really a png")

    _, confidence = parse_resume(str(image))

    assert touched == [], "이미지에 PDF 텍스트 레이어 추출기를 태웠다"
    assert calls == [str(image)]
    assert confidence is Confidence.MID


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".tif", ".bmp", ".webp"])
def test_이미지_확장자_전부_OCR로_간다(ocr_calls, tmp_path, suffix):
    calls = ocr_calls()
    image = tmp_path / f"resume{suffix}"
    image.write_bytes(b"nonsense")

    _, confidence = parse_resume(str(image))

    assert len(calls) == 1
    assert confidence is Confidence.MID


def test_대문자_확장자도_같은_경로를_탄다(ocr_calls, tmp_path):
    calls = ocr_calls()
    image = tmp_path / "RESUME.PNG"
    image.write_bytes(b"nonsense")

    _, confidence = parse_resume(str(image))

    assert len(calls) == 1
    assert confidence is Confidence.MID


# --------------------------------------------------------------------------
# DOCX 경로
# --------------------------------------------------------------------------
def _write_docx(path: Path, paragraphs: list[str], table_rows: list[list[str]] | None = None):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    if table_rows:
        table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, cell in enumerate(row):
                table.cell(r, c).text = cell
    document.save(path)
    return path


def test_DOCX는_OCR_없이_본문을_추출한다(ocr_calls, tmp_path):
    calls = ocr_calls()
    path = _write_docx(
        tmp_path / "resume.docx",
        [
            "김도현 백엔드 엔지니어 경력기술서",
            "넥스트커머스 플랫폼팀에서 쿠버네티스 기반 사내 배포 플랫폼을 구축했다. "
            "서른 개 팀의 백이십여 개 서비스가 이 경로로 배포되며, 배포 리드타임을 "
            "사십 분에서 육 분으로 줄였고 평균 복구시간도 이십이 분에서 사 분으로 낮췄다.",
            "결제 정산 배치의 분산 트랜잭션을 재작성해 중복 정산 사고를 분기 세 건에서 "
            "영 건으로 만들었다. 관측 스택을 도입하고 서비스별 목표를 정의했으며, "
            "과다하게 울리던 알람을 번들링 규칙으로 정리해 주간 알람 수를 오분의 일로 줄였다.",
            "페이브릿지에서는 결제 API 서버를 개발·운영했다. 피크 트래픽은 초당 삼천 건이었고, "
            "샤딩을 도입해 정산 테이블의 단일 인스턴스 한계를 풀었다.",
        ],
    )

    text, confidence = parse_resume(str(path))

    assert calls == []
    assert confidence is Confidence.HIGH
    assert "쿠버네티스" in text


def test_DOCX_표_안의_글자도_본문이다(tmp_path):
    """이력서에서 기술 스택은 표에 들어가는 일이 흔하다."""
    path = _write_docx(
        tmp_path / "resume_table.docx",
        ["김도현 백엔드 엔지니어 경력기술서. " + "상세 경력은 아래 표에 정리했다. " * 4],
        table_rows=[["언어", "Java, Kotlin, Python"], ["인프라", "Kubernetes, Terraform"]],
    )

    text, _ = parse_resume(str(path))

    assert "Kubernetes" in text
    assert "Kotlin" in text


def test_DOCX_빈_문단이_공백_비율을_부풀리지_않는다(tmp_path):
    """이력서는 빈 줄로 레이아웃을 잡는다. 빈 문단을 그대로 이어 붙이면 개행만
    수백 개가 되어 **멀쩡한 문서가 공백 비율에서 탈락**하고 OCR로 흘러간다."""
    paragraphs = [
        "김도현 백엔드 엔지니어 경력기술서입니다. 아래에 주요 경력을 정리했습니다.",
        "넥스트커머스 플랫폼팀에서 쿠버네티스 기반 사내 배포 플랫폼을 구축했습니다. "
        "서른 개 팀의 백이십여 개 서비스가 이 경로로 배포되며, 배포 리드타임을 "
        "사십 분에서 육 분으로 줄였고 평균 복구시간도 크게 낮췄습니다.",
        "결제 정산 배치의 분산 트랜잭션을 재작성해 중복 정산 사고를 없앴습니다. "
        "관측 스택을 도입하고 서비스별 목표를 정의했으며, 과다하게 울리던 알람을 "
        "번들링 규칙으로 정리해 주간 알람 수를 오분의 일로 줄였습니다.",
    ]
    # 문단 사이사이를 빈 줄로 벌려 놓은 실제 이력서 모양
    spaced: list[str] = []
    for paragraph in paragraphs:
        spaced.append(paragraph)
        spaced.extend([""] * 100)

    path = _write_docx(tmp_path / "spaced.docx", spaced)
    text, confidence = parse_resume(str(path))

    assert confidence is Confidence.HIGH, "빈 문단 때문에 멀쩡한 DOCX가 탈락했다"
    assert "쿠버네티스" in text


def test_깨진_DOCX는_예외가_아니라_하_신뢰도다(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip archive at all")

    text, confidence = parse_resume(str(path))

    assert confidence is Confidence.LOW
    assert text == ""


# --------------------------------------------------------------------------
# 실패 처리 — 어느 경우에도 예외를 던지지 않는다
# --------------------------------------------------------------------------
def test_모르는_확장자는_파서를_추측하지_않는다(ocr_calls, tmp_path):
    calls = ocr_calls()
    path = tmp_path / "resume.hwp"
    path.write_bytes(b"\x00\x01\x02")

    text, confidence = parse_resume(str(path))

    assert calls == [], "모르는 확장자에 OCR을 태우면 조용히 돈이 나간다"
    assert (text, confidence) == ("", Confidence.LOW)


def test_깨진_PDF는_예외가_아니라_하_신뢰도다():
    text, confidence = parse_resume("존재하지_않는_파일.pdf")

    assert confidence is Confidence.LOW
    assert text == ""


def test_OCR이_예외를_던져도_전체_실패가_아니다(ocr_calls):
    """T22 불변식 — OCR 실패를 전체 실패로 만들지 않는다."""
    ocr_calls(raises=RuntimeError("엔진 죽음"))

    text, confidence = parse_resume(str(SCAN_PDF))

    assert confidence is Confidence.LOW
    assert needs_manual_correction(text, confidence)


def test_OCR_결과도_품질_미달이면_건진_쪽을_돌려준다(ocr_calls, tmp_path):
    """H4는 '추출 텍스트를 보여주고 고치게' 한다 — 빈 문자열을 주면 고칠 게 없다."""
    ocr_calls(return_value="김도현")  # 너무 짧아 품질 미달

    text, confidence = parse_resume(str(SCAN_PDF))

    assert confidence is Confidence.LOW
    assert text == "김도현", "품질 미달이어도 건진 텍스트는 넘겨야 한다"


# --------------------------------------------------------------------------
# 품질 검사 — 결정론적, LLM 없음
# --------------------------------------------------------------------------
def test_문자_수_임계_경계():
    assert not text_quality_ok("가" * (MIN_CHARS - 1))
    assert text_quality_ok("가" * MIN_CHARS)


def test_앞뒤_공백은_문자_수에_안_들어간다():
    assert not text_quality_ok("   " + "가" * (MIN_CHARS - 1) + "   ")


def test_공백_비율이_높으면_거절한다():
    """분량만 채운 추출물 — 글자 사이가 전부 벌어져 나오는 전형적 깨짐."""
    sparse = " ".join("가" * 3 for _ in range(MIN_CHARS))
    assert len(sparse.strip()) >= MIN_CHARS
    assert whitespace_ratio(sparse) > 0.2
    padded = "가" * MIN_CHARS + " " * (MIN_CHARS * 3)
    assert not text_quality_ok(padded)


def test_모지바케는_거절한다():
    """CJK 인코딩이 깨지면 한글도 라틴도 아닌 글자가 쏟아진다."""
    mojibake = "◆●▲■□▽" * 200
    assert len(mojibake) > MIN_CHARS
    assert count_hangul(mojibake) == 0
    assert count_latin(mojibake) == 0
    assert not text_quality_ok(mojibake)


def test_영문_이력서는_통과한다():
    """한글을 필수로 걸면 멀쩡한 영문 텍스트 레이어를 버리고 OCR을 탄다."""
    english = (
        "Dohyun Kim is a backend and platform engineer with six years of "
        "experience across commerce and fintech domains. He built an internal "
        "deployment platform on Kubernetes and reduced deployment lead time "
        "from forty minutes down to six minutes for over one hundred services."
    )
    assert len(english) > MIN_CHARS
    assert count_hangul(english) == 0
    assert text_quality_ok(english)


def test_판독_가능_문자_임계_경계():
    """MIN_READABLE_CHARS 정확히 = 통과. 경계에 테스트가 없으면 부등호 하나가
    바뀌어도 아무도 모른다."""
    filler = "." * MIN_CHARS  # 분량·공백 조건은 채우되 판독 가능 문자는 아니다
    assert text_quality_ok("가" * MIN_READABLE_CHARS + filler)
    assert not text_quality_ok("가" * (MIN_READABLE_CHARS - 1) + filler)


def test_빈_문자열_공백_비율은_1이다():
    assert whitespace_ratio("") == 1.0
    assert not text_quality_ok("")


def test_한글_자모와_음절을_모두_센다():
    assert count_hangul("가힣") == 2
    assert count_hangul("abc123 !@#") == 0
    assert count_latin("abcXYZ") == 6
    assert count_latin("가나다") == 0


# --------------------------------------------------------------------------
# 계약·규약
# --------------------------------------------------------------------------
def test_반환은_텍스트와_신뢰도_튜플이다():
    result = parse_resume(str(TEXT_PDF))
    assert isinstance(result, tuple) and len(result) == 2
    text, confidence = result
    assert isinstance(text, str)
    assert isinstance(confidence, Confidence)


def test_신뢰도는_층에서_나온다():
    """§5 수집 규약 ③ — CONFIDENCE_BY_LAYER가 유일한 정의 지점이다."""
    assert CONFIDENCE_BY_LAYER[LAYER_TEXT] is Confidence.HIGH
    assert CONFIDENCE_BY_LAYER[LAYER_OCR] is Confidence.MID
    assert CONFIDENCE_BY_LAYER[LAYER_FAILED] is Confidence.LOW
    assert len(set(CONFIDENCE_BY_LAYER.values())) == 3, "층마다 신뢰도가 달라야 한다"


def test_H4_판별자는_하_신뢰도와_빈_텍스트를_잡는다():
    assert needs_manual_correction("", Confidence.HIGH)
    assert needs_manual_correction("   ", Confidence.MID)
    assert needs_manual_correction("멀쩡한 텍스트", Confidence.LOW)
    assert not needs_manual_correction("멀쩡한 텍스트", Confidence.HIGH)
    assert not needs_manual_correction("멀쩡한 텍스트", Confidence.MID)


def test_LLM을_부르지_않는다(monkeypatch):
    """품질 검사는 코드다 — 불변식이 살아 있는지 어댑터를 막아 확인한다."""
    import llm.client

    def boom(*args, **kwargs):
        raise AssertionError("parse_resume이 LLM을 불렀다")

    monkeypatch.setattr(llm.client, "complete_structured", boom, raising=False)

    _, confidence = parse_resume(str(TEXT_PDF))
    assert confidence is Confidence.HIGH

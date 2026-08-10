"""스캔 이력서 OCR — `tools/parse_resume.py`가 텍스트 레이어에 실패했을 때의 폴백 (T22).

T21이 뚫어 둔 이음매를 채운다. 이 파일이 존재하는 순간 `parse_resume`이 자동으로
이 경로를 탄다 — 배선을 위해 그 파일을 고칠 필요가 없다(D68).

엔진
----
**전용 OCR API 대신 기존 비전 어댑터를 쓴다.** 후보는 Upstage Document Parse·CLOVA
OCR·Tesseract였고, 선택 근거는 DEVLOG D69에 적었다. 요지는 셋이다 — ① 키가 이미 있고
실제 채용공고 이미지로 관통이 확인됐다(D51) ② 회당 1콜이라 비용이 얕다(D49)
③ `llm/` 어댑터 경계가 이미 서 있어 R6를 새로 뚫을 필요가 없다.

실패 규약
--------
**실패는 예외가 아니라 빈 문자열이다.** `parse_resume`의 계약이 그렇고, 거기서 다시
H4(사용자 직접 보정)로 흐른다. 다만 **쪽 단위 부분 성공은 살린다** — 3쪽짜리 이력서의
2쪽에서 호출이 실패했다고 1쪽까지 버리면, 사용자가 화면에서 고칠 거리가 사라진다.
"""

from __future__ import annotations

from pathlib import Path

from llm.ocr_vision import extract_text_from_resume_image

# 쪽 예산. 이력서는 1~5쪽이 보통이고 쪽마다 vision 1콜이 나가므로, 잘못 올린 100쪽
# 스캔본이 조용히 100콜로 번지는 것을 막는다. **무한 루프를 막는 것이 예산이 아니라
# `tried`였던 T18과 달리(D58), 여기서 예산은 비용 그 자체가 목적이다.**
MAX_PAGES = 10

# PDF를 굽는 해상도. 200dpi는 본문 OCR의 관례적 하한이고, A4 한 쪽이 회색조로
# 대략 0.3~1.5MB라 어댑터의 15MB 상한에 한참 못 미친다.
#
# **회색조로 굽는다** — 스캔 이력서는 사실상 흑백이라 색을 살려도 읽는 글자가
# 달라지지 않는데 바이트만 3배가 된다. 픽스처 `resume_scan.pdf`도 같은 방식으로 만들었다.
RENDER_DPI = 200

# 확장자 → MIME. 어댑터가 받는 형식만 적는다.
MIME_BY_SUFFIX: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# pymupdf가 열어서 PNG로 다시 구워야 하는 것들. 어댑터가 안 받는 형식이라
# 그냥 넘기면 왕복 없이 거절당한다.
CONVERTED_SUFFIXES = frozenset({".tif", ".tiff", ".bmp"})

PDF_SUFFIXES = frozenset({".pdf"})


def _render_pdf_pages(file_path: str, max_pages: int) -> list[bytes]:
    """PDF를 쪽마다 PNG 바이트로 굽는다. 열리지 않으면 빈 목록."""
    import pymupdf

    try:
        with pymupdf.open(file_path) as doc:
            pages = []
            for index, page in enumerate(doc):
                if index >= max_pages:
                    break
                pixmap = page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csGRAY)
                pages.append(pixmap.tobytes("png"))
            return pages
    except Exception:
        return []


def _convert_to_png(file_path: str) -> list[bytes]:
    """어댑터가 안 받는 이미지 형식을 PNG로 다시 굽는다."""
    import pymupdf

    try:
        with pymupdf.open(file_path) as doc:
            page = doc[0]
            return [page.get_pixmap(colorspace=pymupdf.csGRAY).tobytes("png")]
    except Exception:
        return []


def _load_pages(file_path: str, max_pages: int) -> list[tuple[bytes, str]]:
    """파일을 (이미지 바이트, MIME) 목록으로 편다. 못 읽으면 빈 목록."""
    suffix = Path(file_path).suffix.lower()

    if suffix in PDF_SUFFIXES:
        return [(page, "image/png") for page in _render_pdf_pages(file_path, max_pages)]

    if suffix in CONVERTED_SUFFIXES:
        return [(page, "image/png") for page in _convert_to_png(file_path)]

    mime = MIME_BY_SUFFIX.get(suffix)
    if mime is None:
        return []  # 모르는 확장자에 콜을 태우지 않는다

    try:
        raw = Path(file_path).read_bytes()
    except OSError:
        return []
    return [(raw, mime)] if raw else []


def ocr_document(file_path: str, *, max_pages: int = MAX_PAGES, client=None) -> str:
    """이미지·스캔 PDF에서 텍스트를 추출한다.

    계약: `tools/parse_resume.py`의 OCR 이음매. **실패는 예외가 아니라 빈 문자열.**

    입력: 파일 경로. client는 테스트에서 SDK를 대체하기 위한 주입점이며 평소엔 None.
    출력: 쪽별로 읽은 텍스트를 줄바꿈 둘로 이어 붙인 문자열. 한 쪽도 못 읽으면 `""`.

    쪽 하나가 실패해도 나머지는 살린다 — 부분이라도 있어야 H4에서 고칠 수 있다.
    """
    chunks: list[str] = []
    for image_bytes, mime_type in _load_pages(file_path, max_pages):
        try:
            text = extract_text_from_resume_image(image_bytes, mime_type, client=client)
        except Exception:
            continue  # 이 쪽만 버리고 계속 — 전체 실패로 만들지 않는다
        chunks.append(text)

    # 쪽이 하나도 안 남으면 자연히 "" — 조기 반환 가드를 따로 두지 않는다.
    # **빈 텍스트 검사도 두지 않는다**: 어댑터가 빈 응답을 이미 `VisionError`로
    # 바꾸므로(그쪽이 "추출 성공, 내용 없음"과 구별되게 만든 지점) 여기서 다시 거르면
    # 도달할 수 없는 코드가 된다 — 뮤테이션 M08·M11이 그걸 드러냈다.
    return "\n\n".join(chunks)

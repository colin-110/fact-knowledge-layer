"""Page-level PDF extraction: text blocks, table detection, and complexity scoring.

PyMuPDF is the primary parser (cheap, gives block-level bounding boxes we need
for evidence highlighting later). pdfplumber is used only where PyMuPDF's own
signals suggest a page has a real table, since pdfplumber's table detection is
more reliable but noticeably slower.
"""

import re
from dataclasses import dataclass, field

import pdfplumber
import pymupdf

CHART_KEYWORDS = re.compile(
    r"\b(chart|figure|exhibit|graph|fig\.)\b", re.IGNORECASE
)


@dataclass
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]


@dataclass
class TableData:
    rows: list[list[str | None]]
    bbox: tuple[float, float, float, float] | None


@dataclass
class PageData:
    pdf_page_number: int  # 1-indexed
    printed_page_label: str | None
    raw_text: str
    width: float
    height: float
    blocks: list[TextBlock]
    tables: list[TableData] = field(default_factory=list)
    is_visually_complex: bool = False
    complexity_reason: str = ""


def _detect_printed_label(raw_text: str) -> str | None:
    """Best-effort guess at a printed page number/label near the top or bottom of the page text."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    for candidate in (lines[:2] + lines[-2:]) if lines else []:
        if re.fullmatch(r"[ivxlcdm]{1,6}", candidate, re.IGNORECASE) or re.fullmatch(r"\d{1,4}", candidate):
            return candidate
    return None


def _score_complexity(page: pymupdf.Page, raw_text: str) -> tuple[bool, str]:
    reasons = []
    images = page.get_images(full=True)
    if len(images) >= 1:
        reasons.append(f"{len(images)} embedded image(s)")

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    if len(drawings) >= 40:
        reasons.append(f"{len(drawings)} vector drawing ops")

    if CHART_KEYWORDS.search(raw_text):
        reasons.append("chart/figure keyword present")

    area = max(page.rect.width * page.rect.height, 1.0)
    text_density = len(raw_text) / area
    if text_density < 0.03 and len(raw_text.strip()) < 400:
        reasons.append(f"low text density ({text_density:.4f})")

    is_complex = len(reasons) >= 1 and (len(images) >= 1 or len(drawings) >= 40 or text_density < 0.03)
    return is_complex, "; ".join(reasons)


def _extract_tables_for_page(pdf_path: str, page_index_0based: int) -> list[TableData]:
    tables: list[TableData] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index_0based >= len(pdf.pages):
                return tables
            plumber_page = pdf.pages[page_index_0based]
            for t in plumber_page.find_tables():
                rows = t.extract()
                if rows and len(rows) >= 2:
                    tables.append(TableData(rows=rows, bbox=t.bbox))
    except Exception:
        return []
    return tables


def extract_document(pdf_path: str) -> list[PageData]:
    doc = pymupdf.open(pdf_path)
    pages: list[PageData] = []

    for i, page in enumerate(doc):
        raw_text = page.get_text()
        raw_blocks = page.get_text("blocks", sort=True)
        blocks = [
            TextBlock(text=b[4].strip(), bbox=(b[0], b[1], b[2], b[3]))
            for b in raw_blocks
            if isinstance(b[4], str) and b[4].strip()
        ]
        is_complex, reason = _score_complexity(page, raw_text)

        tables: list[TableData] = []
        if CHART_KEYWORDS.search(raw_text) or "table" in raw_text.lower()[:2000]:
            tables = _extract_tables_for_page(pdf_path, i)
            if tables:
                reason = (reason + "; " if reason else "") + f"{len(tables)} table(s) detected"

        pages.append(
            PageData(
                pdf_page_number=i + 1,
                printed_page_label=_detect_printed_label(raw_text),
                raw_text=raw_text,
                width=page.rect.width,
                height=page.rect.height,
                blocks=blocks,
                tables=tables,
                is_visually_complex=is_complex,
                complexity_reason=reason,
            )
        )

    page_count = doc.page_count
    doc.close()
    return pages


def render_page_png(pdf_path: str, pdf_page_number_1indexed: int, dpi: int = 150) -> bytes:
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[pdf_page_number_1indexed - 1]
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def serialize_table(table: list[list[str | None]]) -> str:
    rows = [" | ".join((c or "").strip() for c in row) for row in table]
    return "\n".join(rows)

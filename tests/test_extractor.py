"""Edge-case coverage for PDF extraction using tiny synthetic PDFs built with PyMuPDF itself -
no starter dataset or network access needed, so these run the same in CI as locally."""

import pymupdf
import pytest

from app.pipeline.extractor import extract_document


def _build_pdf(path, page_builders):
    doc = pymupdf.open()
    for build in page_builders:
        page = doc.new_page(width=400, height=300)
        build(page)
    doc.save(path)
    doc.close()


def _text_page(text):
    def build(page):
        page.insert_text((20, 20), text)
    return build


def _blank_page():
    def build(page):
        pass
    return build


def _image_page(text=""):
    def build(page):
        if text:
            page.insert_text((20, 20), text)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
        pix.set_rect(pix.irect, (255, 0, 0))
        page.insert_image(pymupdf.Rect(50, 50, 150, 150), stream=pix.tobytes("png"))
    return build


def _heavy_drawing_page():
    def build(page):
        # Each finish()+commit() pair produces one separate entry in page.get_drawings() -
        # a single shape with many draw_line() calls before one commit collapses into one path.
        for i in range(60):
            shape = page.new_shape()
            shape.draw_line((10, 10 + i * 3), (390, 10 + i * 3))
            shape.finish()
            shape.commit()
    return build


class TestPlainTextPage:
    def test_text_only_page_is_not_visually_complex(self, tmp_path):
        pdf_path = tmp_path / "text.pdf"
        _build_pdf(pdf_path, [_text_page("Revenue from services was 81,415.38 million in FY24.")])
        pages = extract_document(str(pdf_path))
        assert len(pages) == 1
        assert "81,415.38" in pages[0].raw_text
        assert pages[0].is_visually_complex is False

    def test_page_count_matches_number_of_pages(self, tmp_path):
        pdf_path = tmp_path / "multi.pdf"
        _build_pdf(pdf_path, [_text_page("Page one"), _text_page("Page two"), _text_page("Page three")])
        pages = extract_document(str(pdf_path))
        assert len(pages) == 3
        assert pages[0].pdf_page_number == 1
        assert pages[2].pdf_page_number == 3


class TestBlankPage:
    def test_blank_page_is_not_flagged_visually_complex(self, tmp_path):
        """A page with no text, no images, and no real drawings has nothing for a vision
        call to read - it must not be flagged complex just because text density is low."""
        pdf_path = tmp_path / "blank.pdf"
        _build_pdf(pdf_path, [_blank_page()])
        pages = extract_document(str(pdf_path))
        assert pages[0].raw_text.strip() == ""
        assert pages[0].is_visually_complex is False

    def test_blank_page_produces_no_crash_downstream(self, tmp_path):
        pdf_path = tmp_path / "blank2.pdf"
        _build_pdf(pdf_path, [_blank_page(), _text_page("Some real content here")])
        pages = extract_document(str(pdf_path))
        assert len(pages) == 2


class TestImagePage:
    def test_page_with_embedded_image_is_visually_complex(self, tmp_path):
        pdf_path = tmp_path / "image.pdf"
        _build_pdf(pdf_path, [_image_page("Figure 1: revenue chart")])
        pages = extract_document(str(pdf_path))
        assert pages[0].is_visually_complex is True

    def test_image_only_page_with_no_text_is_still_complex(self, tmp_path):
        pdf_path = tmp_path / "image_only.pdf"
        _build_pdf(pdf_path, [_image_page()])
        pages = extract_document(str(pdf_path))
        assert pages[0].raw_text.strip() == ""
        assert pages[0].is_visually_complex is True


class TestDrawingHeavyPage:
    def test_many_vector_drawings_flags_complex(self, tmp_path):
        pdf_path = tmp_path / "drawings.pdf"
        _build_pdf(pdf_path, [_heavy_drawing_page()])
        pages = extract_document(str(pdf_path))
        assert pages[0].is_visually_complex is True


class TestRenderAndSerialize:
    def test_render_page_png_produces_valid_png_bytes(self, tmp_path):
        from app.pipeline.extractor import render_page_png

        pdf_path = tmp_path / "render.pdf"
        _build_pdf(pdf_path, [_text_page("hello")])
        png_bytes = render_page_png(str(pdf_path), 1)
        assert png_bytes.startswith(b"\x89PNG")

    def test_serialize_table_handles_none_cells(self):
        from app.pipeline.extractor import serialize_table

        table = [["Header A", None, "Header C"], [None, "1,234", ""]]
        text = serialize_table(table)
        assert "Header A" in text and "1,234" in text
        assert "None" not in text

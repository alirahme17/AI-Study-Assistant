"""Chunks keep their PAGE so answers can cite "chap1.pdf, page 2"; scans go through OCR."""
import pymupdf

from process import chunk_pdf, clean_text
from quota import QuotaExhausted


def make_pdf(pages, scanned=()):
    """PDF bytes. Pages listed in `scanned` contain only an image (no text layer)."""
    doc = pymupdf.open()
    for n, text in enumerate(pages, start=1):
        page = doc.new_page()
        if n in scanned:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
            pix.clear_with(200)
            page.insert_image(pymupdf.Rect(72, 72, 300, 300), pixmap=pix)
        elif text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class FakeOCR:
    def __init__(self, text="Handwritten: the Doppler shift is 2v/lambda.", fail_after=None):
        self.text, self.fail_after, self.calls = text, fail_after, 0

    def read_page(self, page):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise QuotaExhausted("PerDay")
        return self.text


def test_chunks_have_pages_and_blank_pages_are_skipped():
    pdf = chunk_pdf(make_pdf(["Introduction to radar systems.", "",
                              "The radar equation relates received power to range."]))
    assert pdf.pages == 3
    assert [c.page for c in pdf.chunks] == [1, 3]
    assert "radar equation" in pdf.chunks[1].content
    assert not any(c.ocr for c in pdf.chunks)


def test_long_page_is_split_but_keeps_its_page():
    doc = pymupdf.open()
    page = doc.new_page()
    long_text = "\n".join(f"Line {i}: the Fourier transform of a sampled signal is periodic." for i in range(40))
    page.insert_textbox(pymupdf.Rect(36, 36, 560, 800), long_text, fontsize=7)
    pdf = chunk_pdf(doc.tobytes())
    assert len(pdf.chunks) > 1
    assert {c.page for c in pdf.chunks} == {1}
    assert [c.chunk_index for c in pdf.chunks] == list(range(len(pdf.chunks)))


def test_scanned_pages_are_read_with_ocr():
    ocr = FakeOCR()
    pdf = chunk_pdf(make_pdf(["Typed page about radar.", None], scanned={2}), ocr=ocr)
    assert ocr.calls == 1
    assert pdf.ocr_pages == 1 and pdf.ocr_skipped == 0
    scanned = [c for c in pdf.chunks if c.page == 2]
    assert scanned and scanned[0].ocr and "Doppler" in scanned[0].content


def test_without_ocr_scanned_pages_are_counted_as_skipped():
    pdf = chunk_pdf(make_pdf(["Typed.", None], scanned={2}), ocr=None)
    assert pdf.ocr_skipped == 1 and pdf.ocr_pages == 0
    assert {c.page for c in pdf.chunks} == {1}


def test_ocr_quota_stops_ocr_but_keeps_indexing():
    ocr = FakeOCR(fail_after=1)
    pdf = chunk_pdf(make_pdf([None, None, None, "Typed last page."], scanned={1, 2, 3}), ocr=ocr)
    assert ocr.calls == 2                  # the failing call, then no more attempts
    assert pdf.ocr_pages == 1 and pdf.ocr_skipped == 2
    assert {c.page for c in pdf.chunks} == {1, 4}


def test_clean_text_removes_garbage_glyphs_but_keeps_real_text():
    raw = "Alphabet of 2ଵ଺=65,536 symbols  � and énergie, Δf, ×, ≈, ½"
    cleaned = clean_text(raw)
    assert "ଵ" not in cleaned and "" not in cleaned and "�" not in cleaned
    assert "énergie" in cleaned and "Δf" in cleaned and "≈" in cleaned and "½" in cleaned
    assert "  " not in cleaned


def test_ocr_errors_skip_only_that_page():
    class Flaky(FakeOCR):
        def read_page(self, page):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("503 UNAVAILABLE: model is experiencing high demand")
            return self.text
    pdf = chunk_pdf(make_pdf([None, None, "Typed."], scanned={1, 2}), ocr=Flaky())
    assert pdf.ocr_errors == 1 and pdf.ocr_skipped == 1 and pdf.ocr_pages == 1
    assert {c.page for c in pdf.chunks} == {2, 3}

"""OCR of scanned (image-only) pages, caching, and progress reporting."""
import pymupdf
import pytest

import ocr
import process

rapidocr = pytest.importorskip("rapidocr_onnxruntime")


def make_scanned_pdf(path, text):
    """A PDF whose only page is a picture of text, like a scanned exam."""
    src = pymupdf.open()
    page = src.new_page(width=595, height=300)
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 260), text, fontsize=16)
    png = page.get_pixmap(dpi=150).tobytes("png")
    scan = pymupdf.open()
    scan.new_page(width=595, height=300).insert_image(pymupdf.Rect(0, 0, 595, 300), stream=png)
    scan.save(path)
    with pymupdf.open(path) as check:
        assert check[0].get_text().strip() == ""      # really no text layer


def test_scanned_page_is_read_with_ocr_and_marked(tmp_path):
    pdf = tmp_path / "exam.pdf"
    make_scanned_pdf(pdf, "The radar equation relates received power to range.")
    chunks = process.process_pdf(str(pdf), "Radar/exam.pdf", ocr.OcrCache(str(tmp_path / "cache")), use_ocr=True)
    text = " ".join(c.page_content for c in chunks).lower()
    assert "radar equation" in text and "range" in text
    assert chunks[0].metadata["ocr"] is True and chunks[0].metadata["page"] == 1


def test_ocr_results_are_cached_per_file(tmp_path, monkeypatch):
    pdf = tmp_path / "exam.pdf"
    make_scanned_pdf(pdf, "Doppler frequency shift")
    cache = ocr.OcrCache(str(tmp_path / "cache"))
    first = process.process_pdf(str(pdf), "Radar/exam.pdf", cache, use_ocr=True)

    def must_not_run(page):
        raise AssertionError("OCR ran again instead of using the cache")
    monkeypatch.setattr(process, "ocr_page", must_not_run)
    second = process.process_pdf(str(pdf), "Radar/exam.pdf", cache, use_ocr=True)
    assert [c.page_content for c in second] == [c.page_content for c in first]


def test_without_ocr_scanned_pages_are_skipped(tmp_path):
    pdf = tmp_path / "exam.pdf"
    make_scanned_pdf(pdf, "Hidden text")
    assert process.process_pdf(str(pdf), "Radar/exam.pdf", None, use_ocr=False) == []


def test_text_pages_are_not_ocrd(tmp_path, monkeypatch):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A normal slide with a real text layer, long enough to count.")
    doc.save(tmp_path / "slides.pdf")
    monkeypatch.setattr(process, "ocr_page", lambda page: (_ for _ in ()).throw(AssertionError("no OCR needed")))
    chunks = process.process_pdf(str(tmp_path / "slides.pdf"), "Radar/slides.pdf", None, use_ocr=True)
    assert chunks[0].metadata["ocr"] is False


def test_progress_messages_during_processing(isolated_workspaces):
    folder = isolated_workspaces / "workspaces" / "alice" / "data" / "Radar"
    folder.mkdir(parents=True)
    make_scanned_pdf(folder / "exam.pdf", "Pulse repetition frequency")
    messages = []
    chunks = process.process_user_pdfs("alice", progress=messages.append)
    assert chunks and messages and messages[0].startswith("Reading PDF 1/1")
    assert (isolated_workspaces / "workspaces" / "alice" / "ocr_cache").is_dir()

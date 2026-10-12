"""Chunks keep their file, course and PAGE so answers can cite "chap1.pdf, page 2"."""
import pymupdf as fitz

from process import process_user_pdfs


def _make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)


def test_chunks_have_course_file_and_page(isolated_workspaces):
    course_dir = isolated_workspaces / "workspaces" / "alice" / "data" / "Radar" / "Course"
    course_dir.mkdir(parents=True)
    _make_pdf(course_dir / "chapter1.pdf", [
        "Introduction to radar systems.",
        "",  # blank page (e.g. scanned) is skipped, numbering continues
        "The radar equation relates received power to range.",
    ])

    chunks = process_user_pdfs("alice")
    assert [c.metadata["page"] for c in chunks] == [1, 3]
    radar_eq = chunks[1]
    assert "radar equation" in radar_eq.page_content
    assert radar_eq.metadata["course"] == "Radar"
    assert radar_eq.metadata["source"] == "Radar/Course/chapter1.pdf"   # relative, portable
    assert radar_eq.metadata["file_name"] == "chapter1.pdf"


def test_long_page_is_split_but_keeps_its_page(isolated_workspaces):
    folder = isolated_workspaces / "workspaces" / "alice" / "data" / "DSP"
    folder.mkdir(parents=True)
    doc = fitz.open()
    page = doc.new_page()
    long_text = "\n".join(f"Line {i}: the Fourier transform of a sampled signal is periodic." for i in range(40))
    page.insert_textbox(fitz.Rect(36, 36, 560, 800), long_text, fontsize=7)
    doc.save(folder / "notes.pdf")

    chunks = process_user_pdfs("alice")
    assert len(chunks) > 1
    assert {c.metadata["page"] for c in chunks} == {1}


def test_clean_text_removes_garbage_glyphs_but_keeps_real_text():
    from process import clean_text
    raw = "Alphabet of 2ଵ଺=65,536 symbols � and énergie, Δf, ×, ≈, ½"
    cleaned = clean_text(raw)
    assert "ଵ" not in cleaned and "" not in cleaned and "�" not in cleaned
    assert "énergie" in cleaned and "Δf" in cleaned and "≈" in cleaned and "½" in cleaned
    assert "  " not in cleaned

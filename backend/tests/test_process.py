"""process.py imported `langchain.text_splitter`, which no longer exists in current LangChain."""
import fitz

from process import process_user_pdfs


def test_pdfs_are_chunked_with_course_metadata(isolated_workspaces):
    course_dir = isolated_workspaces / "workspaces" / "alice" / "data" / "Radar" / "Course"
    course_dir.mkdir(parents=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The radar equation relates received power to range. " * 3)
    doc.save(course_dir / "chapter1.pdf")

    chunks = process_user_pdfs("alice")
    assert chunks, "expected at least one chunk"
    assert chunks[0].metadata["course"] == "Radar"
    assert "radar equation" in chunks[0].page_content

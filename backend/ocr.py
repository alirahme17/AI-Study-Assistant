"""OCR for PDF pages that have no text layer (scanned exams, photographed notes).

Engine: RapidOCR (ONNX runtime). It installs with plain `pip` on Windows/Linux/macOS
and ships its models inside the package, so it needs no extra program and no
internet. It reads printed English/French well (accents are dropped, which the
keyword search ignores anyway); formulas come out approximate; handwriting is
mostly unreadable - that needs a vision LLM, not classic OCR.

Results are cached per PDF (keyed by file content hash), so re-indexing does not
OCR the same pages again.

Set OCR_ENGINE=none to disable OCR.
"""
import hashlib
import json
import os
from functools import lru_cache
from typing import Optional

OCR_ENGINE = os.getenv("OCR_ENGINE", "rapidocr")   # "rapidocr" or "none"
OCR_DPI = 200            # render resolution: higher = better accuracy, slower
MIN_TEXT_CHARS = 30      # a page with less embedded text than this is treated as scanned


def ocr_available() -> bool:
    if OCR_ENGINE == "none":
        return False
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def needs_ocr(page) -> bool:
    """No usable text layer, but there is an image to read."""
    return len(page.get_text().strip()) < MIN_TEXT_CHARS and bool(page.get_images())


def ocr_page(page) -> str:
    """Render a PyMuPDF page and return the recognized text, one line per text box."""
    import numpy as np
    pix = page.get_pixmap(dpi=OCR_DPI)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    result, _ = _engine()(image)
    return "\n".join(box[1] for box in (result or []) if box[1].strip())


class OcrCache:
    """{page_number: text} per PDF, stored as JSON under the user's workspace."""

    def __init__(self, cache_dir: Optional[str]):
        self.cache_dir = cache_dir

    @staticmethod
    def file_hash(path: str) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def _path(self, pdf_path: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{self.file_hash(pdf_path)}-{OCR_DPI}.json")

    def load(self, pdf_path: str) -> dict:
        path = self._path(pdf_path)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    def save(self, pdf_path: str, pages: dict) -> None:
        path = self._path(pdf_path)
        if path and pages:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(pages, f, ensure_ascii=False)

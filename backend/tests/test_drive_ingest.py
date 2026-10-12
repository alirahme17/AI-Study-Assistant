"""Bug 3: Drive listings were cut off after the first page of results."""
import pytest

import ingest


class FakeRequest:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


class FakeFiles:
    """Mimics Drive's files().list paging: returns `page_size` items at a time."""

    def __init__(self, tree, page_size=2):
        self.tree, self.page_size, self.calls = tree, page_size, []

    def list(self, q, fields, pageSize, pageToken, **kwargs):
        folder_id = q.split("'")[1]
        assert "trashed = false" in q
        self.calls.append((folder_id, pageToken))
        children = self.tree.get(folder_id, [])
        start = int(pageToken or 0)
        page = children[start:start + self.page_size]
        result = {"files": page}
        if start + self.page_size < len(children):
            result["nextPageToken"] = str(start + self.page_size)
        return FakeRequest(result)


class FakeService:
    def __init__(self, tree, page_size=2):
        self._files = FakeFiles(tree, page_size)

    def files(self):
        return self._files


def pdf(i):
    return {"id": f"pdf{i}", "name": f"file {i}.pdf", "mimeType": ingest.PDF_MIME}


def test_lists_every_page():
    tree = {"root": [pdf(i) for i in range(5)]}
    service = FakeService(tree, page_size=2)
    items = ingest.list_folder_children(service, "root")
    assert [it["id"] for it in items] == [f"pdf{i}" for i in range(5)]  # old code: only 2
    assert len(service.files().calls) == 3


def test_recursive_download_counts_all_pdfs_across_pages(tmp_path, monkeypatch):
    tree = {
        "root": [{"id": "sub", "name": "Radar", "mimeType": ingest.FOLDER_MIME}] + [pdf(i) for i in range(3)],
        "sub": [pdf(i) for i in range(10, 15)],
    }
    downloaded = []

    def fake_download(service, file_id, path):
        downloaded.append(file_id)
    monkeypatch.setattr(ingest, "_download_file", fake_download)

    count = ingest._download_pdfs_recursively(FakeService(tree, page_size=2), "root", str(tmp_path))
    assert count == 8
    assert len(downloaded) == 8


@pytest.mark.parametrize("url,expected", [
    ("https://drive.google.com/drive/folders/1AbC_dEf-123?usp=sharing", "1AbC_dEf-123"),
    ("https://drive.google.com/drive/u/0/folders/1AbC_dEf-123", "1AbC_dEf-123"),
    ("https://drive.google.com/open?id=1AbC_dEf-123", "1AbC_dEf-123"),
    ("1AbC_dEf-123456", "1AbC_dEf-123456"),
    ("not a link", None),
])
def test_extract_folder_id(url, expected):
    assert ingest.extract_folder_id(url) == expected


def test_safe_filename():
    assert ingest.safe_filename('Ch 1/2: "intro"?.pdf') == "Ch 1_2_ _intro__.pdf"
    assert ingest.safe_filename("...") == "_"

"""Where uploaded PDFs live: a local folder (development) or Supabase Storage (hosted).

Free hosts have no persistent disk, so in production every PDF is kept in a private
Supabase Storage bucket and downloaded when needed (indexing, rendering a cited page).
"""
import os
from collections import OrderedDict
from functools import lru_cache
from threading import Lock

import settings


class LocalStorage:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def _path(self, key: str) -> str:
        path = os.path.abspath(os.path.join(self.root, key))
        if not path.startswith(self.root + os.sep):
            raise ValueError("Invalid storage key.")
        return path

    def ensure_ready(self) -> None:
        os.makedirs(self.root, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass


class SupabaseStorage:
    """Supabase Storage REST API, authenticated with the project's secret key."""

    def __init__(self, url: str, service_key: str, bucket: str):
        import httpx
        if not url or not service_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required for Supabase storage.")
        headers = {"apikey": service_key}
        if service_key.count(".") == 2:          # legacy service_role JWT
            headers["Authorization"] = f"Bearer {service_key}"
        self.client = httpx.Client(base_url=f"{url}/storage/v1", headers=headers, timeout=120)
        self.bucket = bucket

    def ensure_ready(self) -> None:
        """Create the private bucket if it doesn't exist yet."""
        if self.client.get(f"/bucket/{self.bucket}").status_code == 200:
            return
        r = self.client.post("/bucket", json={"id": self.bucket, "name": self.bucket, "public": False,
                                              "allowed_mime_types": ["application/pdf"]})
        if r.status_code >= 400 and "already exists" not in r.text.lower():
            r.raise_for_status()

    def put(self, key: str, data: bytes) -> None:
        r = self.client.post(f"/object/{self.bucket}/{key}", content=data,
                             headers={"Content-Type": "application/pdf", "x-upsert": "true"})
        r.raise_for_status()

    def get(self, key: str) -> bytes:
        r = self.client.get(f"/object/{self.bucket}/{key}")
        r.raise_for_status()
        return r.content

    def delete(self, key: str) -> None:
        r = self.client.request("DELETE", f"/object/{self.bucket}", json={"prefixes": [key]})
        r.raise_for_status()


class CachedStorage:
    """Keeps recently used PDFs in memory (bounded), so paging through a cited
    document doesn't re-download it for every page."""

    def __init__(self, inner, max_bytes: int = 48 * 1024 * 1024):
        self.inner, self.max_bytes = inner, max_bytes
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._size = 0
        self._lock = Lock()

    def ensure_ready(self):
        self.inner.ensure_ready()

    def put(self, key, data):
        self.inner.put(key, data)
        self._drop(key)

    def delete(self, key):
        self.inner.delete(key)
        self._drop(key)

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        data = self.inner.get(key)
        if len(data) <= self.max_bytes // 2:
            with self._lock:
                self._cache[key] = data
                self._size += len(data)
                while self._size > self.max_bytes:
                    _, old = self._cache.popitem(last=False)
                    self._size -= len(old)
        return data

    def _drop(self, key):
        with self._lock:
            old = self._cache.pop(key, None)
            if old is not None:
                self._size -= len(old)


@lru_cache(maxsize=1)
def get_storage():
    if settings.STORAGE_BACKEND == "supabase":
        inner = SupabaseStorage(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY, settings.STORAGE_BUCKET)
    else:
        inner = LocalStorage(settings.LOCAL_STORAGE_DIR)
    return CachedStorage(inner)

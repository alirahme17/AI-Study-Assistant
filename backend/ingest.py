import io
import os
import re
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from workspace import data_path, workspace_path

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
PDF_MIME = "application/pdf"

_FOLDER_ID_PATTERNS = [
    re.compile(r"/folders/([A-Za-z0-9_-]+)"),   # .../drive/folders/<id>?usp=sharing
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),     # .../open?id=<id>
]
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")


def extract_folder_id(url_or_id: str) -> Optional[str]:
    """Get the folder ID from any common Drive link format, or a bare ID."""
    text = (url_or_id or "").strip()
    for pattern in _FOLDER_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return text if _BARE_ID.match(text) else None


def safe_filename(name: str) -> str:
    """Drive allows names like 'Ch 1/2: intro' that are invalid on Windows/Linux."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(".")
    return cleaned or "_"


def list_folder_children(service, folder_id: str) -> list:
    """Return ALL items in a folder, following Drive's pagination.

    files().list returns at most one page (100 items by default); without
    following nextPageToken, everything after the first page was silently skipped.
    """
    items, page_token = [], None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def _download_file(service, file_id: str, local_path: str) -> None:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    # Write to a temp name first so an interrupted download is never
    # mistaken for a finished file and skipped next time.
    tmp_path = local_path + ".part"
    with open(tmp_path, "wb") as f:
        f.write(fh.getvalue())
    os.replace(tmp_path, local_path)


def _download_pdfs_recursively(service, folder_id, local_path):
    """Recursively find and download PDFs. Returns the number of PDFs found."""
    os.makedirs(local_path, exist_ok=True)
    pdf_count = 0

    for item in list_folder_children(service, folder_id):
        current_local_path = os.path.join(local_path, safe_filename(item["name"]))

        if item["mimeType"] == FOLDER_MIME:
            print(f"Entering subfolder: '{item['name']}'...")
            pdf_count += _download_pdfs_recursively(service, item["id"], current_local_path)
        elif item["mimeType"] == PDF_MIME:
            pdf_count += 1
            if os.path.exists(current_local_path):
                print(f"Skipping '{item['name']}', already downloaded.")
                continue
            print(f"Downloading PDF: '{item['name']}'...")
            _download_file(service, item["id"], current_local_path)
    return pdf_count


def ingest_drive_data(username: str, drive_folder_id: str) -> int:
    """Download every PDF under the Drive folder into the user's workspace.

    Returns the number of PDFs found. Manages Google OAuth credentials per user.
    """
    os.makedirs(workspace_path(username), exist_ok=True)

    creds = None
    token_path = os.path.join(workspace_path(username), "token.json")
    credentials_path = "credentials.json"

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    "credentials.json is missing in backend/. "
                    "Create an OAuth client in Google Cloud Console (see README)."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    print(f"Starting recursive download for user '{username}'...")
    count = _download_pdfs_recursively(service, drive_folder_id, data_path(username))
    print(f"Found {count} PDF(s) for user '{username}'.")
    return count

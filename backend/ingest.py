import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _download_pdfs_recursively(service, folder_id, local_path):
    """Helper function to recursively find and download PDFs."""
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get("files", [])

    for item in items:
        item_name = item['name']
        item_id = item['id']
        item_mime_type = item['mimeType']
        current_local_path = os.path.join(local_path, item_name)

        if item_mime_type == 'application/vnd.google-apps.folder':
            print(f"Entering subfolder: '{item_name}'...")
            _download_pdfs_recursively(service, item_id, current_local_path)
        elif item_mime_type == 'application/pdf':
            if os.path.exists(current_local_path):
                print(f"Skipping '{item_name}', already downloaded.")
                continue
            print(f"Downloading PDF: '{item_name}'...")
            request = service.files().get_media(fileId=item_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"  Download {int(status.progress() * 100)}%.")
            with open(current_local_path, "wb") as f:
                f.write(fh.getvalue())

def ingest_drive_data(username: str, drive_folder_id: str):
    """
    Connects to a user's Google Drive and saves PDFs to their workspace.
    Manages Google OAuth credentials per user.
    """
    workspace_path = os.path.join("workspaces", username)
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
    
    creds = None
    token_path = os.path.join(workspace_path, "token.json")
    credentials_path = "credentials.json"

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)
    download_path = os.path.join(workspace_path, "data")
    
    print(f"Starting recursive download for user '{username}'...")
    _download_pdfs_recursively(service, drive_folder_id, download_path)
    print(f"\nAll files for user '{username}' downloaded successfully!")
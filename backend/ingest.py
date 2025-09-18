import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def download_pdfs_recursively(service, folder_id, local_path):
    """
    Recursively finds and downloads all PDFs from a Google Drive folder
    and its subfolders, mirroring the directory structure locally.
    """
    # Create the local directory if it doesn't exist
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # Query for all items (files and folders) inside the current folder_id
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get("files", [])

    for item in items:
        item_name = item['name']
        item_id = item['id']
        item_mime_type = item['mimeType']
        
        # Current item's local path
        current_local_path = os.path.join(local_path, item_name)

        # If the item is a folder, call this function again (recursion)
        if item_mime_type == 'application/vnd.google-apps.folder':
            print(f"Entering subfolder: '{item_name}'...")
            download_pdfs_recursively(service, item_id, current_local_path)
        
        # If the item is a PDF, download it
        elif item_mime_type == 'application/pdf':
            # Skip if file already exists to save time on re-runs
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

def main():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("drive", "v3", credentials=creds)
        
        semester_folder_id = "12KUVxcQIM3A5bk-qrPOqrhqQYhfzwVoP"
        download_path = "data"

        print("Starting recursive download of all PDFs...")
        download_pdfs_recursively(service, semester_folder_id, download_path)
        print("\nAll files downloaded successfully!")

    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()
import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_user_pdfs(username: str):
    """
    Processes all PDFs in a specific user's data directory, chunks them,
    and assigns correct metadata.
    """
    workspace_path = os.path.join("workspaces", username)
    data_directory = os.path.join(workspace_path, "data")
    all_chunks = []
    
    if not os.path.exists(data_directory):
        print(f"Data directory not found for user '{username}'. Skipping processing.")
        return []

    print(f"Starting PDF processing for user '{username}'...")
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                
                try:
                    relative_path = os.path.relpath(file_path, data_directory)
                    course_name = relative_path.split(os.sep)[0]
                except (IndexError, ValueError):
                    course_name = "Unknown"

                try:
                    doc = fitz.open(file_path)
                    full_text = "".join(page.get_text() for page in doc)
                    doc.close()
                    
                    if not full_text.strip():
                        continue

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.create_documents([full_text])
                    
                    for i, chunk in enumerate(chunks):
                        chunk.metadata = {
                            "source_file": file_path,
                            "course": course_name,
                            "chunk_index": i
                        }
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return all_chunks
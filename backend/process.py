import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_all_pdfs(data_directory="data"):
    all_chunks = []
    
    print("Starting PDF processing...")
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                
                # --- NEW LOGIC: Correctly parse the course name ---
                try:
                    # This splits the path relative to the data_directory
                    # e.g., "data/Mobile Communications/Lecture Notes/file.pdf"
                    # becomes "Mobile Communications/Lecture Notes/file.pdf"
                    relative_path = os.path.relpath(file_path, data_directory)
                    # The first part of the relative path is the course name
                    course_name = relative_path.split(os.sep)[0]
                except (IndexError, ValueError):
                    print(f"Warning: Could not determine course for {file_path}. Skipping.")
                    continue

                try:
                    doc = fitz.open(file_path)
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                    doc.close()
                    
                    if not full_text.strip():
                        print(f"Warning: No text found in {file_path}")
                        continue

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                    )
                    
                    chunks = text_splitter.create_documents([full_text])
                    
                    for i, chunk in enumerate(chunks):
                        chunk.metadata = {
                            "source_file": file_path,
                            "course": course_name, # <-- This will now be correct
                            "chunk_index": i
                        }
                    
                    all_chunks.extend(chunks)
                    print(f"Processed '{file_path}' for course '{course_name}'.")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return all_chunks

if __name__ == "__main__":
    chunks = process_all_pdfs()
    print(f"\nTotal chunks created: {len(chunks)}")
    
    if chunks:
        print("\n--- Example Chunk ---")
        print(chunks[0].page_content)
        print(f"\nMetadata: {chunks[0].metadata}")
        print("--------------------")
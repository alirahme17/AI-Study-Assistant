import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Import the processing function from our other script
from process import process_all_pdfs

# Define the directory where the vector database will be stored
PERSIST_DIRECTORY = "chroma_db"

def build_vector_database():
    """
    Builds a ChromaDB vector database from the processed PDF chunks.
    """
    # 1. Get all the text chunks from our documents
    print("Step 1: Processing PDFs into chunks...")
    chunks = process_all_pdfs()
    if not chunks:
        print("No chunks were created. Exiting.")
        return

    # 2. Initialize the local embedding model
    print("Step 2: Initializing local embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    
    # 3. Build and persist the ChromaDB database
    # This single command creates the embeddings and stores them in the database.
    print(f"Step 3: Building and persisting the vector database to '{PERSIST_DIRECTORY}'...")
    print(f"This will take a while. Processing {len(chunks)} chunks...")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print("\nVector database built successfully!")


if __name__ == "__main__":
    build_vector_database()
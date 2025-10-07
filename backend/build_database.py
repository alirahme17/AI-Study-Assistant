import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from process import process_user_pdfs

def build_user_database(username: str):
    """
    Builds a ChromaDB vector database for a specific user in their workspace.
    """
    workspace_path = os.path.join("workspaces", username)
    persist_directory = os.path.join(workspace_path, "chroma_db")
    
    chunks = process_user_pdfs(username)
    if not chunks:
        print(f"No chunks found for user '{username}'. Database not built.")
        return

    print(f"Building vector database for user '{username}'...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    print(f"\nVector database for user '{username}' built successfully!")
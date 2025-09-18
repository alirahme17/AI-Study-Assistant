import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import defaultdict

# Define the directory where the vector database is stored
PERSIST_DIRECTORY = "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def inspect_database_chunks():
    """
    Loads the ChromaDB database and prints the first 5 chunks from each course.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"Error: Database directory '{PERSIST_DIRECTORY}' not found.")
        print("Please run 'build_database.py' first.")
        return

    # 1. Initialize the embedding model (needed to load the DB)
    print("Loading embedding model...")
    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 2. Load the existing ChromaDB database
    print(f"Loading database from '{PERSIST_DIRECTORY}'...")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)

    # 3. Retrieve all documents from the database
    # We can use db.get() to fetch documents without performing a similarity search.
    print("Fetching all documents from the database...")
    results = db.get(include=["metadatas", "documents"])
    
    if not results or not results['ids']:
        print("Database is empty or could not be read.")
        return
        
    # 4. Group chunks by course
    # A defaultdict is a convenient way to handle grouping.
    chunks_by_course = defaultdict(list)
    
    metadatas = results['metadatas']
    documents = results['documents']

    for i in range(len(metadatas)):
        course = metadatas[i].get('course', 'Unknown Course')
        chunk_text = documents[i]
        chunks_by_course[course].append(chunk_text)

    print("\n--- First 5 Chunks From Each Course ---")

    # 5. Iterate through the grouped chunks and print the first 5
    for course, chunks in chunks_by_course.items():
        print(f"\n\n## Course: {course} ({len(chunks)} total chunks)")
        print("--------------------------------------------------")
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n[Chunk {i + 1}]")
            # Print a snippet of the chunk for brevity
            print(f"{chunk[:250]}...")
    
    print("\n\nInspection complete.")

if __name__ == "__main__":
    inspect_database_chunks()
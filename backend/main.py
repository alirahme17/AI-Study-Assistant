from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional

load_dotenv()

# Models and DB loading remains the same...
print("Loading embedding model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

print("Loading ChromaDB...")
PERSIST_DIRECTORY = "chroma_db"
db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)

print("Loading Gemini 1.5 Flash model...")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

app = FastAPI()

class Query(BaseModel):
    question: str
    course: Optional[str] = None

@app.post("/api/query")
def query_database(query: Query):
    question = query.question
    course_filter = query.course
    print(f"Received query: '{question}' for course: '{course_filter}'")

    search_kwargs = {"k": 4}

    # --- THIS IS THE UPDATED FILTER LOGIC ---
    # We now use an exact match ('$eq') on the 'course' metadata field.
    if course_filter:
        search_kwargs['filter'] = {"course": {"$eq": course_filter}}

    retrieved_docs = db.similarity_search(query=question, **search_kwargs)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # The rest of the function remains the same...
    print("Retrieved context from database.")

    prompt_template = f"""
    Answer the following question based only on the context provided.
    If the answer is not in the context, say that the information is not available in the provided documents.
    

    Context:
    {context}

    Question:
    {question}
    """
    print("Generating answer with Gemini...")
    response = llm.invoke(prompt_template)

    return {"answer": response.content, "sources": [doc.metadata.get("source_file", "unknown") for doc in retrieved_docs]}

@app.get("/")
def read_root():
    return {"message": "Teleco-Tutor Backend is running with Gemini!"}
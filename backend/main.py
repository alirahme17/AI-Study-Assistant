import os
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import threading
from dotenv import load_dotenv

# Import refactored functions
from ingest import ingest_drive_data
from build_database import build_user_database

# Langchain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

class ConfigurePayload(BaseModel):
    username: str
    drive_folder_url: str
    gemini_api_key: str

class QueryPayload(BaseModel):
    username: str
    question: str
    course: Optional[str] = None

def run_data_pipeline(username: str, drive_folder_id: str):
    """Function to run the long-running data processing in a background thread."""
    try:
        print(f"Starting data pipeline for user: {username}")
        ingest_drive_data(username, drive_folder_id)
        build_user_database(username)
        print(f"Data pipeline for user '{username}' completed successfully.")
    except Exception as e:
        print(f"Error in data pipeline for user '{username}': {e}")

@app.post("/api/configure")
def configure_workspace(payload: ConfigurePayload):
    workspace_path = os.path.join("workspaces", payload.username)
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)
    
    with open(os.path.join(workspace_path, ".env"), "w") as f:
        f.write(f"GEMINI_API_KEY={payload.gemini_api_key}\n")

    try:
        drive_folder_id = payload.drive_folder_url.split('/')[-1].split('?')[0]
    except IndexError:
        raise HTTPException(status_code=400, detail="Invalid Google Drive URL format.")

    pipeline_thread = threading.Thread(target=run_data_pipeline, args=(payload.username, drive_folder_id))
    pipeline_thread.start()
    return {"message": f"Configuration received for '{payload.username}'. Data processing started."}

@app.get("/api/status/{username}")
def get_status(username: str):
    db_path = os.path.join("workspaces", username, "chroma_db")
    return {"status": "ready" if os.path.exists(db_path) else "processing"}

@app.get("/api/courses/{username}")
def get_courses(username: str):
    data_dir = os.path.join("workspaces", username, "data")
    try:
        courses = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        return {"courses": sorted(courses)}
    except FileNotFoundError:
        return {"courses": []}

@app.post("/api/query")
def query_database(payload: QueryPayload):
    username = payload.username
    workspace_path = os.path.join("workspaces", username)
    db_path = os.path.join(workspace_path, "chroma_db")
    env_path = os.path.join(workspace_path, ".env")

    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found for this user. Please configure first.")
    
    load_dotenv(dotenv_path=env_path)
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=400, detail="Gemini API key not found for this user.")

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings_model)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

    search_kwargs = {"k": 4}
    if payload.course:
        search_kwargs['filter'] = {"course": {"$eq": payload.course}}

    retrieved_docs = db.similarity_search(query=payload.question, **search_kwargs)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt_template = f"Answer based only on the context provided.\n\nContext:\n{context}\n\nQuestion:\n{payload.question}"

    def stream_generator():
        try:
            for chunk in llm.stream(prompt_template):
                yield chunk.content
        except Exception as e:
            yield f"Error during response generation: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")
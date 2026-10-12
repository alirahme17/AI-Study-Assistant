"""All configuration, from environment variables (see .env.example).

For local development, a backend/.env file is loaded too (never commit it).
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)
except ImportError:
    pass

# --- database / storage ---------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5433/tutor")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")          # "local" | "supabase"
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", "storage")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")     # secret key: server only, never in the browser
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET", "pdfs")
AUTO_MIGRATE = os.getenv("AUTO_MIGRATE", "1") == "1"            # create tables/bucket at startup (idempotent)

# --- AI providers -----------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")           # writes the answers
EMBEDDINGS = os.getenv("EMBEDDINGS", "gemini")                         # "gemini" | "fake" (offline tests/demos)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMS = 768                                                   # must match tt_chunks.embedding
OCR = os.getenv("OCR", "gemini")                                       # "gemini" | "none"
OCR_MODEL = os.getenv("OCR_MODEL", "gemini-3.5-flash-lite")            # reads scanned/handwritten pages
FAKE_LLM = os.getenv("FAKE_LLM") == "1"                                # offline demo answers (devllm.py)

# --- retrieval --------------------------------------------------------------------------
TOP_K = 5
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")                 # "hybrid" | "vector" | "keyword"

# --- web --------------------------------------------------------------------------------
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# --- secrets ----------------------------------------------------------------------------
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")                       # Fernet key for stored Gemini keys

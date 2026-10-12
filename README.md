# Teleco-Tutor: a study assistant for your own course PDFs 📡

Upload your course material (slides, lecture notes, past exams, even scanned or handwritten pages) and ask questions in English or French. Every answer comes **only** from your PDFs, with numbered citations: click one and the exact page opens with the passage highlighted.

## Features

- **Answers grounded in your material**: the assistant cites `[1]`, `[2]`… after each claim, and says so when the answer isn't in your PDFs.
- **Cited pages, highlighted**: a citation opens the PDF page with the cited passage marked, with page-by-page navigation.
- **Chat memory**: follow-ups like *"and what about its range?"* are understood in context.
- **Hybrid search**: meaning-based vector search (multilingual Gemini embeddings, so English questions find French slides) plus exact keyword search for technical terms like `DSDV` or `S11`, merged with Reciprocal Rank Fusion.
- **Scanned and handwritten pages** are read by a Gemini vision model.
- **Upload a whole folder**: each sub-folder becomes a course. Indexing starts by itself, file by file, with live status.
- **Free to run** for a few people: Supabase, Render and Vercel free tiers, and each user's own free Gemini key.
- **Measured quality**: a 25-question evaluation set with page-level ground truth (`backend/eval/`).

## Architecture

```
 Browser ──► Next.js app (Vercel)
                │  sign-in token (Supabase Auth)
                ▼
         FastAPI backend (Render, 512 MB)
           │            │              │
           ▼            ▼              ▼
   Postgres + pgvector  Storage     Gemini API
   (chunks, embeddings, (the PDFs,  (answers, embeddings,
    full-text index)     private)    OCR) with the user's key
   └──────── Supabase ───────┘
```

- **Indexing queue** (`backend/indexer.py`): one row per PDF, `pending → processing → ready / error`. Nothing is lost when the free server sleeps or restarts, or when a Gemini key hits its daily limit: finished PDFs stay searchable and the rest continues the next time the app is opened.
- **Security**: every API call is authenticated with the Supabase token and scoped to that user's id. Users' Gemini keys are stored encrypted. The database tables are not exposed to the browser (Row Level Security on, no policies). The storage bucket is private.

| Folder | What's inside |
|---|---|
| `backend/` | FastAPI API, RAG pipeline, indexing, tests, evaluation |
| `backend/db/schema.sql` | Database schema (created automatically at startup) |
| `web/` | Next.js + React + Tailwind frontend |
| `render.yaml` | Render blueprint for the backend |

## Deploy (free tiers)

You need accounts on [Supabase](https://supabase.com), [Render](https://render.com) and [Vercel](https://vercel.com), all signed in with GitHub, no credit card.

### 1. Supabase (database, file storage, sign-in)

1. Create a project and note its **database password**.
2. **Project Settings → Database → Connection string → Transaction pooler**: copy the URI (port `6543`) and put your password in it. This is `DATABASE_URL`.
3. **Project Settings → API Keys**: copy the **secret** key (`sb_secret_…`). This is `SUPABASE_SERVICE_KEY`. Never put it in the frontend or in git.
4. **Authentication → URL Configuration**: set the Site URL to your Vercel address (step 3) once you have it.

The backend creates its tables and the private `pdfs` bucket by itself on first start.

### 2. Render (backend)

1. **New → Blueprint**, pick this repository. Render reads `render.yaml`.
2. Fill in the secret values it asks for:
   - `DATABASE_URL` and `SUPABASE_SERVICE_KEY` from step 1.
   - `ENCRYPTION_KEY`: generate once with
     `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
     Keep it: if it changes, users have to enter their Gemini key again.
   - `CORS_ORIGINS`: your Vercel address, e.g. `https://teleco-tutor.vercel.app`.
3. Check `https://<your-service>.onrender.com/api/health` shows `"db": "ok"`.

The free instance sleeps after 15 minutes without requests; the first request after that takes about a minute.

### 3. Vercel (web app)

1. **Add New → Project**, import this repository, set **Root Directory** to `web`.
2. Environment variables:
   - `NEXT_PUBLIC_API_URL` = the Render address.
   - `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` (the **publishable** key, which is safe in the browser).
3. Deploy, then put the Vercel address in Supabase (step 1.4) and Render's `CORS_ORIGINS`.

### Using it

Create an account, add a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) in Settings, then upload PDFs or a whole folder. Keep the tab open while a large library is indexing: the open tab keeps the free server awake. If the Gemini free daily limit runs out, indexing pauses and resumes the next day. A free Supabase project is paused after a week without any use; restore it from the Supabase dashboard.

## Local development

Requirements: Python 3.11+, Node 20+, and a Postgres 15+ with the `pgvector` extension (for example the `pgvector/pgvector:pg16` Docker image).

```bash
# backend
cd backend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then set DATABASE_URL, STORAGE_BACKEND=local
uvicorn main:app --reload                            # http://127.0.0.1:8000

# web app (second terminal)
cd web
npm install
cp .env.example .env.local
npm run dev                                          # http://localhost:3000
```

Without `SUPABASE_URL`, the backend runs in **dev mode**: no passwords, the sign-in page asks for a test user name. With `EMBEDDINGS=fake OCR=none FAKE_LLM=1` everything works offline, without a Gemini key (answers then quote the retrieved passages).

## Tests and evaluation

```bash
cd backend
TEST_DATABASE_URL=postgresql://postgres@localhost:5432/tutor_test pytest -q
python eval/run_eval.py --user you@example.com --retrieval-only
```

The tests use a real Postgres (emptied between tests) and offline fakes for Gemini. See `backend/eval/README.md` for the evaluation set and metrics.

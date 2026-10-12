# Teleco-Tutor web app

Next.js frontend: sign-in, chat with streaming answers, citations that open the cited PDF page with the passage highlighted, course library with PDF upload, settings, light/dark mode.

## Run locally

```bash
cd web
npm install
cp .env.example .env.local   # Windows: copy .env.example .env.local
npm run dev                  # http://localhost:3000
```

The backend must be running (see the main README). Without Supabase settings the app runs in **local dev mode**: the sign-in page just asks for a workspace name (e.g. an existing one such as `mhmd123`).

### Try the UI without a Gemini key

Start the backend with `FAKE_LLM=1`: answers then quote the retrieved passages with real citations instead of calling Gemini, so you can click through everything offline.

## Structure

| Path | What it does |
|---|---|
| `src/app/login/page.tsx` | Sign in / create account (Supabase), or pick a workspace in dev mode |
| `src/app/page.tsx` | Workspace: course sidebar, onboarding, conversation, source panel |
| `src/components/Answer.tsx` | Markdown + KaTeX answer; `[n]` citations become highlighter chips |
| `src/components/SourcePanel.tsx` | Rendered PDF page with the cited passage highlighted, page navigation |
| `src/components/LibraryDialog.tsx` | Upload/remove PDFs per course, rebuild the knowledge base, Drive import |
| `src/components/SettingsDialog.tsx` | Gemini API key, theme, sign out |
| `src/lib/api.ts` | Typed API client, NDJSON streaming reader, friendly Gemini error messages |
| `src/lib/auth.tsx` | Supabase session (or dev workspace) and the headers sent to the backend |

# Evaluation set

`questions.csv` holds questions about the course PDFs, each with a known answer and the exact PDF and page it comes from. It is used to measure the RAG pipeline's quality: retrieval, citations and answer correctness.

## Columns

| Column | Meaning |
|---|---|
| `id` | Stable identifier (`q01`…) |
| `course` | Top-level course folder; used as the course filter, like in the chat UI |
| `type` | `fact`, `definition`, `numeric`, `formula`, `reasoning`, or `negative` |
| `language` | Language of the question (`en` / `fr`) |
| `question` | What the student asks |
| `reference_answer` | The correct answer, written from the cited page |
| `source`, `page` | `<course>/<path inside the course>` exactly as shown in the app's Sources panel, and 1-based page number |
| `also_ok` | Other `file#page` locations that also contain the answer (duplicate slides), separated by `;` |
| `verified` | Set to `yes` after a human has checked the row against the PDF |
| `notes` | What the question is designed to test |

`negative` questions have no source: their topic is deliberately **not** in the materials, and the assistant must say it couldn't find the answer instead of using general knowledge.

## Adding questions

Open the CSV in Excel or LibreOffice, add a row, keep the same columns, and save as **CSV UTF-8**. Take `source` from the Sources panel in the chat (it shows the same path). When a folder is uploaded with each sub-folder as a course, it is the path inside the uploaded folder.

## Running the evaluation

From the `backend` folder, with the virtual environment active and `DATABASE_URL` pointing at the database (the same value as on Render, e.g. in `backend/.env`):

```powershell
python eval/run_eval.py --user you@example.com                  # full run (uses the account's Gemini key)
python eval/run_eval.py --user you@example.com --retrieval-only # retrieval metrics only (embeds the 25 questions)
python eval/run_eval.py --user you@example.com --compare        # vector vs keyword vs hybrid, side by side
```

`--user` is the account's e-mail, its user id, or a dev-mode test user name.

**Free-tier quota:** free Gemini keys allow only a small number of requests *per model per day* (20 for `gemini-3.8-flash` at the time of writing), and a full run needs about 50. The judge therefore uses a different model (`--judge-model`, default `gemini-3.5-flash-lite`) so it draws on a separate quota. When a daily quota runs out, the run stops cleanly and keeps every finished question; continue it the next day with:

```powershell
python eval/run_eval.py --user you@example.com --resume
```

Each run writes `eval/results/<timestamp>_<model>/report.md` (readable summary) and `results.json` (everything, for comparing runs). The answer judge is the same Gemini model that writes the answers, so treat the answer score as a strong signal, not ground truth; the retrieval and citation metrics are exact.

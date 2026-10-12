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
| `source`, `page` | PDF path relative to the user's `data/` folder, and 1-based page number |
| `also_ok` | Other `file#page` locations that also contain the answer (duplicate slides), separated by `;` |
| `verified` | Set to `yes` after a human has checked the row against the PDF |
| `notes` | What the question is designed to test |

`negative` questions have no source: their topic is deliberately **not** in the materials, and the assistant must say it couldn't find the answer instead of using general knowledge.

## Adding questions

Open the CSV in Excel or LibreOffice, add a row, keep the same columns, and save as **CSV UTF-8**. Take `source` from the Sources panel in the chat (it shows the same path), or from the folder structure under `workspaces/<user>/data/`.

## Running the evaluation

From the `backend` folder, with the virtual environment active:

```powershell
python eval/run_eval.py --user mhmd123                  # full run (uses the user's Gemini key)
python eval/run_eval.py --user mhmd123 --retrieval-only # retrieval metrics only, no API calls
```

**Free-tier quota:** free Gemini keys allow only a small number of requests *per model per day* (20 for `gemini-3.8-flash` at the time of writing), and a full run needs about 50. The judge therefore uses a different model (`--judge-model`, default `gemini-3.5-flash-lite`) so it draws on a separate quota. When a daily quota runs out, the run stops cleanly and keeps every finished question; continue it the next day with:

```powershell
python eval/run_eval.py --user mhmd123 --resume
```

Each run writes `eval/results/<timestamp>_<model>/report.md` (readable summary) and `results.json` (everything, for comparing runs). The answer judge is the same Gemini model that writes the answers, so treat the answer score as a strong signal, not ground truth; the retrieval and citation metrics are exact.

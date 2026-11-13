# Evidence-Grounded Fact Knowledge Layer

Turns PDFs into a queryable layer of facts, where every fact links back to the exact page, text, or chart it
came from, and facts from different documents are automatically checked against each other: do they
corroborate, contradict, or only differ because of context?

<p align="center">
  <a href="https://drive.google.com/file/d/11Ns9VgLIaBC0U2QHDBuwt0iK0OEf-YCN/view?usp=drive_link">
    <img src="https://img.shields.io/badge/%E2%96%B6%20Watch%20the%20Demo-Video-E53935?style=for-the-badge" alt="Watch the demo video" height="48">
  </a>
</p>

<p align="center">
  <b>A real PDF processed end-to-end, plus the four showcase cases below, shown live against the running system.</b>
</p>

## Problem

Reports state the same number many ways: ₹81,415 million in an annual report, ₹8,142 Cr in an investor deck,
59% on one page and 60% on another. A plain RAG chatbot will happily quote either. To trust an answer you need
to know **where each fact came from** and **whether other sources agree with it**.

This project treats facts as first-class records: extracted by an LLM, normalized deterministically, grounded
in evidence, and linked to each other by typed relationships.

## Architecture

```
Upload (API or web UI) -> FastAPI -> 202 + job_id
                                |
                     background worker (per-page, resumable)
                                |
        +-----------------------+------------------------+
        |                       |                        |
   PyMuPDF text           pdfplumber tables      chart-heavy page?
   + block bboxes                                -> spatial-text reflow
                                                 -> vision LLM (Groq -> Gemini)
        +-----------------------+------------------------+
                                |
                 EVIDENCE (page, bbox, method, confidence)
                                |
                 LLM fact extraction -> deterministic normalization
                                |
             SQLite (source of truth) + Chroma (embeddings)
                                |
                 RELATIONSHIP ENGINE: nearest neighbours
                 -> rules for clear cases -> LLM for ambiguous ones
```

**Query path:** question -> dense (Chroma) + lexical (SQLite FTS5 / BM25) retrieval -> Reciprocal Rank Fusion
-> cross-encoder rerank -> facts + evidence + relationships -> LLM answer grounded only in what was retrieved.

## Engineering decisions

- **Evidence is mandatory.** Every fact carries `evidence_ids` (page, exact quote or chart read, bbox, method,
  confidence). A fact without evidence can't be inserted.
- **LLM decides *what* a fact is; code decides *what it means*.** Extraction is generic (subject, predicate,
  value, unit, period, scope, status), with no fact types hard-coded. Money magnitudes, fiscal periods, and
  negatives in `(parentheses)` are normalized with plain Python, so two runs never disagree on whether two
  numbers match.
- **Rules first, LLM second, for relationships.** Same metric, period, scope, and matching values ->
  `CORROBORATES`, no LLM call. Only genuinely ambiguous pairs go to the LLM, along with both facts' evidence.
- **Cheap before paid for charts.** Chart-heavy pages first try a free spatial-text reflow of PDF word positions;
  vision models (Groq, then Gemini) run only when that isn't enough.
- **SQLite is the source of truth.** Chroma is a derived index that can be rebuilt from SQLite at any time.
- **Idempotent, resumable ingestion.** IDs are content hashes: re-uploading is a no-op, a crash only redoes the
  page it died on, and unfinished jobs resume automatically on the next startup.
- **Built for free-tier limits.** Pages run concurrently (6 workers), calls are pre-paced under Groq's
  tokens-per-minute cap, multiple Groq keys round-robin with failover, and Gemini takes over when Groq's quota
  runs out.
- **Ask never waits behind ingestion.** The rate limiter reserves a separate interactive lane for `/query`.
- **Small on purpose.** No Redis, Docker, or external vector DB: an in-process queue and embedded Chroma give the
  same async behaviour with one process to run.

### Relationship rules

| Same period | Same scope | Same status | Values within 1% | Result |
|---|---|---|---|---|
| yes | yes | yes | yes | `CORROBORATES` |
| yes | yes | yes | no, gap ≤ 15% | `LIKELY_CONTRADICTION` |
| yes | yes | yes | no, gap > 15% | `CONTRADICTS` |
| yes | no | - | - | `CONTEXT_RECONCILES` (scope) |
| yes | yes | no | - | `CONTEXT_RECONCILES` (estimate vintage) |
| anything else | | | | LLM decides, given both facts' evidence |

## Showcase cases

All produced by the running system on real reports (Delhivery's FY24 Annual Report and Q4 FY24 earnings deck).

| Case | What the documents say | Result |
|---|---|---|
| **Corroborates** | Annual Report p.22: revenue ₹81,415.38 million. Q4 deck p.6: ₹8,142 Cr. | Normalizes to ₹8,141.5 Cr vs ₹8,142 Cr (0.01% apart). `CORROBORATES`, no LLM call. |
| **Likely contradiction** | Annual Report p.8 says female workers grew 60% YoY; the same page also says 59%. | `LIKELY_CONTRADICTION` (confidence 0.6). The 59% figure is backed three more times in the report (5,594 vs 3,519 headcount). |
| **Context reconciles** | Q4 deck: FY24 revenue ₹5,077 Cr and ₹1,429 Cr. | Different business segments (Express Parcel vs PTL freight). `CONTEXT_RECONCILES` on scope. |
| **Failures, surfaced** | See below. | Every extraction issue is logged and shown in the Failures tab. |

Real failures found by running the system, and what was done about them:

- **Lost sign.** `(4,516.08)` (accounting notation for a loss) was extracted as positive, creating a false
  contradiction. *Fixed* in the extraction prompt.
- **Lookalike metrics.** "Adjusted EBITDA" vs "EBITDA" and YoY vs QoQ growth were flagged `CONTRADICTS`.
  *Reduced*: rules now fire only on near-identical wording, with regression tests for each false positive.
- **Silent skips.** Re-uploading skipped pages that had failed on a rate limit. *Fixed*: failed pages are retried.
- **Open:** the Prospectus's side-by-side standalone and consolidated columns lose their headers in text
  extraction, producing a false `CONTRADICTS`. This needs better table detection.

## Measured

| What | Result |
|---|---|
| Chart-heavy pages that skipped paid vision calls | 17 / 20 (Q4 deck), 83 / 83 (Annual Report excerpt) |
| `/query` while a bulk upload runs | 9.3 s before the interactive lane, normally ~1–3 s |
| Test suite | 120 tests, no API key or network needed, run in CI on every push |

## Tech stack

| Layer | Technologies |
|---|---|
| API | FastAPI, Pydantic, in-process job queue |
| PDF parsing | PyMuPDF, pdfplumber, spatial-text reflow |
| Storage | SQLite + FTS5, Chroma (embedded) |
| Retrieval | `BAAI/bge-small-en-v1.5` embeddings, BM25, RRF, `ms-marco-MiniLM-L-6-v2` cross-encoder |
| LLMs | Groq (`openai/gpt-oss-120b` text, Llama 4 Scout vision), Gemini fallback (`gemini-3.6-flash`) |
| Frontend | Plain HTML / CSS / JS, no build step |
| CI | GitHub Actions + pytest |

## Run locally

Requires Python 3.11+ and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/colin-110/fact-knowledge-layer.git
cd fact-knowledge-layer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # set GROQ_API_KEY
uvicorn app.main:app --reload
```

Open **http://localhost:8000**, which has tabs for Documents, Facts, Relationships, Ask, and Failures. Interactive
API docs are at `/docs`.

Optional `.env` settings: `GROQ_API_KEYS` (extra comma-separated keys for more throughput) and `GEMINI_API_KEY`
([free](https://aistudio.google.com/apikey), used as a fallback for vision and text).

### API

```bash
curl -F "file=@report.pdf" http://localhost:8000/documents    # -> {"document_id", "job_id", "status": "queued"}
curl http://localhost:8000/jobs/<job_id>                       # progress
curl "http://localhost:8000/facts?document_id=<document_id>"
curl http://localhost:8000/relationships
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What was Delhivery'"'"'s FY24 revenue?"}'
```

### Tests

```bash
pytest tests/ -v
```

Unit tests for normalization, IDs, rules, and fusion; an end-to-end pipeline test with mocked LLM calls; a
retrieval-accuracy test against the real embedding model; and API smoke tests.

## Limitations and next steps

- **Groq's free tier allows 200K tokens a day**, and a large multi-document corpus can use all of it. Extra keys
  and the Gemini fallback help, but a paid key is the real fix.
- **Text evidence points to the whole page.** Tables and charts get precise boxes; matching quotes to text
  blocks is next.
- **Documents are processed one at a time** (pages within a document run in parallel).
- **Extraction favours numbers** over role and membership facts, such as committee tables.
- **Retrieval was checked by hand** against the showcase cases, not with a scored recall@k benchmark.

## What I focused on

Making every answer **traceable and checkable**: each fact points to its source, cross-document conflicts are
surfaced rather than hidden, and failures are logged and shown, not swallowed. Wherever arithmetic could replace
an LLM call (normalization, clear-cut relationships, chart pages that already contain text), it does: that makes
results cheaper and reproducible.

# Evidence-Grounded Fact Knowledge Layer

Built for the Superjoin VIT 2026 Engineering Intern assignment: extract facts from PDFs, ground every fact in its
source evidence, and detect when facts across documents corroborate, contradict, or reconcile through context.

## Setup and Run Instructions

Requires Python 3.11+ and a [Groq API key](https://console.groq.com/keys) (free tier is enough).

```bash
git clone https://github.com/colin-110/fact-knowledge-layer.git
cd fact-knowledge-layer
python -m venv .venv
pip install -r requirements.txt

cp .env.example .env             # then edit .env and set GROQ_API_KEY
```

Optional in `.env`: `GROQ_API_KEYS` (extra comma-separated Groq keys, round-robined for more throughput/quota)
and `GEMINI_API_KEY` (free tier at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) - used only
as a vision fallback for chart/figure pages when Groq has no working vision model on your account/tier).

Run it directly with the venv's own interpreter - this works the same in cmd, PowerShell, and bash, and
sidesteps PowerShell's script-execution-policy issues with `.venv\Scripts\Activate.ps1`:

```bash
# Windows
.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# macOS / Linux
.venv/bin/python -m uvicorn app.main:app --reload
```

One process serves the API, the background ingestion worker, and the UI. Open **http://localhost:8000** - a
plain HTML/CSS/JS frontend (no build step, no framework) with Documents, Facts, Relationships, Ask, and Failures
tabs, polling the API every few seconds so it stays live without a manual refresh. Upload a PDF (any PDF - not
limited to the starter dataset) and watch it process. Or drive it directly via the API:

```bash
curl -F "file=@some-report.pdf" http://localhost:8000/documents
# -> {"document_id": "...", "job_id": "...", "status": "queued"}

curl http://localhost:8000/jobs/<job_id>
curl http://localhost:8000/facts?document_id=<document_id>
curl http://localhost:8000/relationships
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What was Delhivery'"'"'s FY24 revenue?"}'
```

Run the test suite (no API key needed - it never makes a real LLM call):

```bash
pytest tests/ -v
```

## Video Demo

_TODO: link to a ≤3 minute demo video showing a PDF being processed and the four required cases below._

## Approach

### Architecture

```
                    UPLOAD (API or web UI)
                              |
                              v
                    FastAPI  ->  202 + job_id
                              |
                              v
                    in-process queue -> background worker thread
                              |
                              v
                    PDF PARSER (per page, persisted incrementally)
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
         PyMuPDF text     pdfplumber        page flagged
         + block bboxes    tables            visually complex?
              |               |                    |
              |               |                    v
              |               |            render full page -> PNG
              |               |                    |
              |               |            vision LLM (Groq, Llama 4
              |               |            Scout/Maverick) extracts
              |               |            chart/figure data, or
              |               |            reports low confidence
              |               |                    |
              +---------------+--------------------+
                              |
                              v
                          EVIDENCE  (SQLite: document, page, bbox where
                                      available, extraction method, confidence)
                              |
                              v
                      LLM FACT EXTRACTION (generic subject/predicate/value/
                      unit/period/scope/status/qualifiers schema - the model
                      decides what's a fact, not a document-specific rule)
                              |
                              v
                       NORMALIZATION (deterministic: money magnitude ->
                       canonical crore, fiscal period -> start/end dates,
                       status keywords)
                              |
                +-------------+--------------+
                v                             v
            SQLite (source of truth)     embeddings -> Chroma
                                          (facts_index, evidence_index)
                              |
                              v
                    RELATIONSHIP ENGINE
              (dense nearest-neighbor candidate generation
               -> deterministic rules for confident cases
               -> LLM classification, with both facts' evidence,
                  for everything ambiguous)
                              |
                              v
                       relationships table
                 (CORROBORATES / CONTRADICTS /
                  LIKELY_CONTRADICTION / CONTEXT_RECONCILES)
```

Query path:

```
question -> hybrid retrieval (dense Chroma search + SQLite FTS5 BM25,
            fused with Reciprocal Rank Fusion) -> cross-encoder rerank
            -> facts + evidence + any relationships touching them
            -> LLM answer, grounded only in what was actually retrieved
```

### Why this architecture

- **SQLite is the source of truth, not the vector index.** Every fact and evidence row lives in SQLite with a
  deterministic, content-hashed ID. Chroma only holds embeddings + enough metadata to look the row back up -
  it's a derived retrieval index that could be deleted and rebuilt from SQLite at any time.
- **No Redis/RQ/Qdrant/Docker.** The assignment explicitly says *"a smaller, understandable prototype is better
  than a large system whose behavior is unclear."* A background thread + an in-process queue gives the same
  observable behavior as a real task queue (upload returns immediately with a job id, status is pollable, a
  crash doesn't take down the API) without an extra service to run, configure, or explain. Chroma runs embedded
  (no server) for the same reason. This is called out explicitly as the first easy scale-out point in
  Limitations below.
- **Vision LLM instead of a chart-parsing library, with a two-provider fallback chain.** The starter PDFs lean
  heavily on charts, bar graphs, and infographics with no accompanying text table. A dedicated chart-to-table
  parser is brittle and still needs a fallback for anything it can't handle. Sending the full page image to a
  vision-capable model and asking it to report low confidence rather than guess is both simpler to implement
  and more honest about uncertainty. In practice, Groq's free tier didn't expose a working vision model on the
  key used for this submission (confirmed via `client.models.list()`), so `app/pipeline/vision.py` tries Groq
  first and falls over to Gemini (`GEMINI_API_KEY`, free tier) if Groq has none - and if neither is configured,
  falls back further to a free, zero-key spatial-text reflow (`app/pipeline/spatial_text.py`) that clusters
  PyMuPDF's word positions by proximity instead of raw reading order, which recovers a fair amount of structure
  for any chart whose numbers are real PDF text objects rather than a rasterized image.
- **Deterministic normalization, LLM-driven extraction.** *What* counts as a fact and its subject/predicate/value
  is entirely up to the LLM (the assignment is explicit that facts shouldn't be hard-coded to any document).
  But *how* ₹81,415 million relates to ₹8,142 crore, or how "FY24" maps to a date range, is arithmetic and
  calendar logic - regex and Python, not a model, so two runs never disagree about whether two numbers match.
- **Rules first, LLM second, for relationships.** Comparing every fact to every other fact is wasteful and most
  pairs are trivially unrelated. Chroma's nearest-neighbor search on the fact's own embedding finds plausible
  candidates cheaply; a same-subject/metric/period/scope/status pair with matching normalized values is
  confidently CORROBORATES without ever calling an LLM. Only genuinely ambiguous pairs (different scope, status,
  or a numeric gap with no obvious cause) go to the LLM, and it's given both facts' actual source evidence, not
  just the two numbers, so it can reason about *why* they might differ instead of just noticing that they do.
- **Concurrent page processing, multi-key failover.** Pages are I/O-bound (waiting on Groq), so a thread pool
  (`INGESTION_WORKERS`, default 6) processes them concurrently instead of one at a time - this alone took a
  100-page document from ~20 minutes to under a minute in testing, the rest being a retry bug (see Limitations).
  `GROQ_API_KEYS` accepts multiple comma-separated keys; calls round-robin across them and fail over to the next
  key when one is rate-limited or a model is unavailable on it, which both multiplies throughput and daily quota.
- **Plain HTML/CSS/JS frontend, no framework.** Matches the "smaller, understandable prototype" instruction and
  removes a dependency (and the auto-refresh problems of Streamlit's rerun model) - the page polls the API every
  few seconds so job progress and new facts/relationships show up without a manual refresh, with zero build step.
- **Hybrid retrieval + RRF, not dense-only.** Financial/economic text is full of exact tokens - "FY24", "EBITDA",
  ticker-like codes - that dense embeddings routinely miss and BM25 catches immediately. Reciprocal Rank Fusion
  combines the two ranked lists by rank rather than raw score, so cosine similarity and BM25 scores (different
  scales, not directly comparable) never need to be reconciled.
- **Canonical retrieval text, not raw PDF fragments.** A fact is embedded as `"Subject: Delhivery. Metric:
  revenue from services. Value: ...FY24. Scope: consolidated."` rather than a raw sentence copy-pasted from the
  PDF. This makes both semantic search and the "why did this get retrieved" experience much more legible.
- **Idempotent by content hash, not by re-running from scratch.** Document IDs are derived from the file's
  SHA-256; page/evidence/fact IDs are derived from (document, page, content). Uploading the same PDF twice is a
  no-op; reprocessing after a crash only redoes the page it was on when it died (`pages.processed_at` gates
  re-extraction). This is also what "new documents incrementally, without rebuilding all existing knowledge"
  falls out of for free - ingesting document #7 never touches documents #1-6.

### Fact schema

Deliberately generic - no field is specific to Delhivery, GDP, or any other document:

```json
{
  "id": "fa_9f2c1a...",
  "subject": "Delhivery",
  "predicate": "revenue from services",
  "raw_value": "81,415.38",
  "raw_unit": "INR million",
  "normalized_value": 8141.538,
  "normalized_unit": "INR crore",
  "period_label": "FY24",
  "period_start": "2023-04-01",
  "period_end": "2024-03-31",
  "scope": "consolidated",
  "status": "actual",
  "qualifiers": {},
  "evidence_ids": ["ev_..."],
  "extraction_confidence": 0.8
}
```

### Evidence / provenance

Every fact carries `evidence_ids` pointing at rows in the `evidence` table, each with `document_id`, `page`,
`evidence_type` (text / table / chart), the exact source text (or the vision model's structured read of a
chart), an `extraction_method`, a `confidence`, and - for chart/figure evidence - a saved crop of the actual
page image, servable via `GET /evidence/{id}/artifact` and shown inline in the Facts tab. A fact with
no evidence is a schema violation the code cannot produce - `insert_fact` always takes `evidence_ids`.

### Retrieval

Query -> dense search over `facts_index`/`evidence_index` (Chroma, embedding = canonical retrieval text) in
parallel with SQLite FTS5 BM25 lexical search -> Reciprocal Rank Fusion -> cross-encoder rerank
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, degrades gracefully to the RRF order if it can't load) -> top facts +
evidence + any relationships touching those facts -> handed to the LLM as context, which is told to answer only
from what's given and to say so when it isn't enough.

### Relationship reasoning

See **Four Required Cases** below for real output. Summary of the decision logic
(`app/pipeline/relationship_engine.py`):

| Same period? | Same scope? | Same status? | Values match (≤1%)? | Result |
|---|---|---|---|---|
| yes | yes | yes | yes | `CORROBORATES` |
| yes | yes | yes | no, gap ≤15% | `LIKELY_CONTRADICTION` |
| yes | yes | yes | no, gap >15% | `CONTRADICTS` |
| yes | no | — | — | `CONTEXT_RECONCILES` (scope) |
| yes | yes | no | — | `CONTEXT_RECONCILES` (estimate vintage) |
| anything else | | | | LLM classifies, given both facts' source evidence |

### AI tools used

Claude Code (Sonnet 5) wrote essentially all of the implementation in this repository, working from an
architecture spec I gave it, iterating phase by phase with a commit after each. It also read the starter PDFs
directly to locate the real facts used in the four required-case demonstrations below (all quotes below are
verbatim from the source PDFs, not invented), and diagnosed a couple of live issues by actually running the
pipeline end-to-end against a real Groq key (a wasted-retry-on-permanent-404 bug, and confirming which models
the account could actually call via `client.models.list()`, rather than assuming from documentation).

The runtime LLMs the *application itself* calls (never hard-coded, always from `.env`): Groq's
`openai/gpt-oss-120b` for text (fact extraction, normalization hints, relationship classification) and its
configured vision model for chart/figure reads, falling over to Google Gemini (`gemini-2.0-flash`) if Groq has
no working vision model available, and further to a zero-key spatial-text heuristic if neither is configured.

## Four Required Cases

_All quotes below are verbatim from the starter dataset PDFs (page numbers as printed in the excerpts)._

### 1. CORROBORATES - Delhivery FY24 revenue, stated two ways

- **Annual Report FY24** (`02-delhivery-annual-report-fy24-excerpt.pdf`, p.22): *"The revenue from operations on
  consolidated basis for FY24 stood at ₹81,415.38 million as against ₹72,253.01 million for FY23..."*
- **Q4 FY24 Earnings Presentation** (`03-delhivery-q4-fy24-earnings-presentation.pdf`, p.6): *"₹8,142 Cr FY24
  revenue from services / YoY: 12.7%"*

₹81,415.38 million normalizes to ₹8,141.538 crore - within 0.01% of the ₹8,142 Cr figure (rounding to the
nearest crore in the investor deck). Same subject, metric, period, and scope; the rule engine fires
`CORROBORATES` deterministically, no LLM call needed.

### 2. LIKELY_CONTRADICTION - female workforce growth, same report, same page

Both statements appear on **page 8** of the Annual Report:

- *"The number of female workers in our combined on-roll and off-roll workforce increased 60% year-on-year."*
- *"Increasing female workforce • Number of female workers increased by 59% year-on-year"*

The 59% figure is independently corroborated three more times elsewhere in the same report (pp.17, 26, 38,
including the underlying headcount math: 5,594 vs 3,519 = 59% growth), making the 60% on p.8 the clear outlier.
Same subject/metric/period, no scope or status distinction visible in either passage, and the numeric gap
(1.7% relative) is too large for the corroboration threshold but too small to confidently call a real
contradiction - the rule engine classifies this `LIKELY_CONTRADICTION` and says so without inventing an
explanation it can't see in the text.

### 3. CONTEXT_RECONCILES - India real GDP growth, FY25

- **Economic Survey 2024-25** (`01-india-economic-survey-2024-25-excerpt.pdf`, p.4): *"As per the first advance
  estimates of national accounts, India's real GDP is estimated to grow by 6.4 per cent in FY25."*
- **RBI Annual Report 2024-25** (`02-rbi-annual-report-2024-25-excerpt.pdf`, p.8): *"...real gross domestic
  product (GDP) growth moderated to 6.5 per cent in 2024-25..."*
- **IMF Article IV 2025** (`03-imf-india-2025-article-iv-excerpt.pdf`, p.10): *"India's real GDP grew by 6.5
  percent in FY2024/25."*

Same subject and period once "FY25" / "2024-25" / "FY2024/25" are all normalized to the same fiscal year, but
the Economic Survey explicitly flags its number as a *First Advance Estimate* (published before the fiscal year
closed) while RBI and IMF report later, more complete data vintages. Different `status` -> the rule engine
fires `CONTEXT_RECONCILES` with `context_dimension: estimate_vintage` rather than flagging a contradiction.

### 4. Extraction failure

**Q4 FY24 Earnings Presentation, page 13**: a combined quarterly/yearly "Adjusted EBITDA (₹ Cr)" bar chart
(Q1 FY23 - Q4 FY24 plus FY23/FY24 totals) with value and margin labels positioned around the bars and no
accompanying data table on that page. Plain PyMuPDF text extraction returns the numbers stripped of their
axis association and partially concatenated - e.g. four distinct quarterly values collapse into a single
token like `"(125) (67) (25)6"`. The page is correctly flagged `is_visually_complex` and routed to the vision
LLM fallback; when the model's own confidence for a given chart region comes back low, the system does not
silently drop it or guess - it stores the evidence row as-is (so you can see exactly what was extracted) and
logs an `extraction_issue` with the reported confidence and uncertainty note, surfaced unfiltered in the
web UI's **Failures** tab and via `GET /extraction-issues`. This is the intended failure-handling behavior:
show what was extracted, how confident the system is, and why it might be wrong, rather than hiding it.

## Limitations and Next Steps

**Works:**
- End-to-end pipeline: upload -> async processing -> evidence -> facts -> relationships -> query, all backed by
  a real job status you can poll.
- Deterministic, idempotent ingestion (safe to re-upload or resume after a crash).
- Hybrid retrieval with RRF fusion and reranking.
- Rule-based relationship classification for the confident cases, LLM fallback for the rest, always shown its
  source evidence rather than reasoning from bare numbers.

**Does not work yet / known gaps:**
- **Text evidence bbox is page-level, not block-level.** Tables and charts get precise bounding boxes; plain
  narrative text evidence currently points at the whole page rather than the specific sentence, so evidence
  highlighting for text facts is coarser than for tabular/chart facts. The block-level text is captured in
  `extractor.py` but not yet threaded through to fact-evidence linking - matching an LLM's `supporting_quote`
  back to its originating text block is the natural next step.
- **Documents queue one at a time, even though pages within a document run concurrently.** `app/jobs.py`'s
  worker loop pulls one document at a time from the queue; within a document, `run_document_pipeline` processes
  pages (and later, relationship candidates) through a thread pool (`INGESTION_WORKERS`, default 6), which is
  what took a 100-page document from ~20 minutes to under a minute in testing. Uploading five documents back to
  back still processes them one after another rather than all five concurrently - the interface in
  `app/jobs.py` is narrow enough that swapping the single worker for a small pool of document-level workers
  later wouldn't touch the pipeline code at all.
- **No dynamic schema evolution UI.** The fact schema is already generic enough that new kinds of facts don't
  require code changes (qualifiers is free-form JSON), but there's no explicit mechanism for detecting *new
  categories* of predicate and surfacing that to a user - it just silently accumulates whatever predicates the
  LLM chooses.
- **Relationship candidate generation is capped** (top-5 nearest neighbors per fact) to bound LLM call volume.
  A fact whose true match isn't in its nearest 5 embedding neighbors won't be compared at all. Works well in
  practice because the canonical retrieval text puts subject+metric+period+scope right in the embedding, but
  it's a real recall ceiling on a much larger corpus.
- **No formal retrieval evaluation set.** Retrieval quality was checked by hand against the four required cases
  rather than a scored recall@k benchmark.
- Would like to add: block-level evidence highlighting on the actual PDF page (not just the page image), a
  small worker pool, and a lightweight UI affordance for "these look like a new predicate we haven't seen
  before" during ingestion.

## Additional Notes

The repository is currently private (`https://github.com/colin-110/fact-knowledge-layer`) - grant the reviewer
access or flip it to public before submitting.

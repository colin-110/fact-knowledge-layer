# Evidence-Grounded Fact Knowledge Layer

Built for the Superjoin VIT 2026 Engineering Intern assignment: extract facts from PDFs, ground every fact in its
source evidence, and detect when facts across documents corroborate, contradict, or reconcile through context.

## Video Demo

[Watch the demo](https://drive.google.com/file/d/11Ns9VgLIaBC0U2QHDBuwt0iK0OEf-YCN/view?usp=drive_link) - a PDF
being processed end-to-end and the four required cases below, shown live against the running system.

## Setup and Run Instructions

Requires Python 3.11+ and a [Groq API key](https://console.groq.com/keys) (free tier is enough).

```bash
git clone https://github.com/colin-110/fact-knowledge-layer.git
cd fact-knowledge-layer
python -m venv .venv
pip install -r requirements.txt

cp .env.example .env             # then edit .env and set GROQ_API_KEY
```

Optional in `.env`: `GROQ_API_KEYS` (extra comma-separated Groq keys, round-robined for more throughput/quota) and
`GEMINI_API_KEY` (free tier at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) - a fallback for
both vision and text calls whenever Groq can't serve one).

```bash
# Windows
.venv\Scripts\python.exe -m uvicorn app.main:app --reload

# macOS / Linux
.venv/bin/python -m uvicorn app.main:app --reload
```

One process serves the API, the background ingestion worker, and the UI. Open **http://localhost:8000** - a
plain HTML/CSS/JS frontend (no build step) with Documents, Facts, Relationships, Ask, and Failures tabs, polling
the API so it stays live. Upload any PDF (not limited to the starter dataset) and watch it process, or drive it
directly:

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
              |               |            render page -> PNG -> vision LLM
              |               |            (Groq -> Gemini -> spatial-text reflow)
              +---------------+--------------------+
                              |
                              v
                          EVIDENCE  (SQLite: document, page, bbox, method, confidence)
                              |
                              v
                      LLM FACT EXTRACTION (generic subject/predicate/value/
                      unit/period/scope/status/qualifiers - model decides what's a fact)
                              |
                              v
                       NORMALIZATION (deterministic: money magnitude -> canonical
                       crore, fiscal period -> start/end dates, status keywords)
                              |
                +-------------+--------------+
                v                             v
            SQLite (source of truth)     embeddings -> Chroma
                                          (facts_index, evidence_index)
                              |
                              v
                    RELATIONSHIP ENGINE
              (nearest-neighbor candidates -> deterministic rules for
               confident cases -> LLM classification for ambiguous ones)
                              |
                              v
                       relationships table
                 (CORROBORATES / CONTRADICTS / LIKELY_CONTRADICTION / CONTEXT_RECONCILES)
```

Query path: `question -> hybrid retrieval (dense Chroma + SQLite FTS5 BM25, fused with RRF) -> cross-encoder
rerank -> facts + evidence + touching relationships -> LLM answer grounded only in what was retrieved`.

### Key decisions

- **SQLite is the source of truth, not the vector index.** Chroma only holds embeddings + enough metadata to look
  the row back up; it's a derived index that could be rebuilt from SQLite at any time.
- **No Redis/Qdrant/Docker.** The brief explicitly favors "a smaller, understandable prototype." An in-process
  queue + background thread gives the same observable behavior (async upload, pollable status, crash-safe) as a
  real task queue without an extra service to run or explain. Chroma runs embedded for the same reason.
- **Vision LLM, not a chart parser, with a fallback chain.** The starter PDFs lean on charts with no accompanying
  text table. `app/pipeline/vision.py` tries Groq's vision model, falls to Gemini if unavailable/rate-limited, and
  falls further to a zero-key spatial-text reflow (clusters PyMuPDF word positions by proximity) if neither vision
  provider is configured.
- **Deterministic normalization, LLM-driven extraction.** *What* counts as a fact is entirely up to the LLM (the
  brief is explicit facts shouldn't be hard-coded). *How* ₹81,415 million relates to ₹8,142 crore, or "FY24" maps
  to a date range, is arithmetic - regex and Python, so two runs never disagree on whether two numbers match.
- **Rules first, LLM second, for relationships.** Chroma's nearest-neighbor search finds plausible candidates
  cheaply; a same-subject/metric/period/scope/status pair with matching values is `CORROBORATES` with no LLM
  call. Only genuinely ambiguous pairs go to the LLM, given both facts' actual evidence so it can reason about
  *why* they differ, not just that they do.
- **Concurrent pages, multi-key failover.** A thread pool (`INGESTION_WORKERS`, default 6) processes pages
  concurrently since they're I/O-bound waiting on Groq. `GROQ_API_KEYS` round-robins across multiple keys and
  fails over when one is rate-limited, multiplying both throughput and daily quota.
- **Idempotent by content hash.** Document/page/fact IDs derive from content hashes, so re-uploading is a no-op
  and reprocessing after a crash only redoes the page it died on. This is also what "new documents incrementally"
  falls out of for free - ingesting document #7 never touches #1-6. On startup, the app also re-enqueues any job
  an unclean shutdown left mid-run, so an interrupted document resumes automatically.
- **Ask gets its own rate-limit lane.** `/query` and background ingestion share the same Groq account, so a bulk
  upload could make a live question wait behind it (measured: 9.3s instead of ~1-3s). The token bucket in
  `app/llm.py` reserves a separate `"interactive"` slice so asking a question never queues behind ingestion.

### Fact schema

Deliberately generic - no field is specific to Delhivery or any other document:

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

### Evidence, retrieval, and relationship logic

Every fact carries `evidence_ids` into the `evidence` table (page, exact text or chart read, extraction method,
confidence, and a saved page-crop for charts, servable via `GET /evidence/{id}/artifact`). `insert_fact` always
requires evidence - a fact with none is a schema violation the code can't produce.

Retrieval: dense Chroma search + SQLite FTS5 BM25 lexical search, fused with Reciprocal Rank Fusion, then a
cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`, degrades gracefully if it can't load).

Relationship decision table (`app/pipeline/relationship_engine.py`):

| Same period? | Same scope? | Same status? | Values match (≤1%)? | Result |
|---|---|---|---|---|
| yes | yes | yes | yes | `CORROBORATES` |
| yes | yes | yes | no, gap ≤15% | `LIKELY_CONTRADICTION` |
| yes | yes | yes | no, gap >15% | `CONTRADICTS` |
| yes | no | — | — | `CONTEXT_RECONCILES` (scope) |
| yes | yes | no | — | `CONTEXT_RECONCILES` (estimate vintage) |
| anything else | | | | LLM classifies, given both facts' source evidence |

### AI tools used

Claude Code (Sonnet 5) wrote the implementation in this repository, working from an architecture spec I
directed, iterating in commits. It read the starter PDFs directly to locate the real examples used in the Four
Required Cases below (all quotes are verbatim from the source PDFs), and diagnosed several live issues by
actually running the pipeline against a real Groq key rather than assuming from documentation.

Runtime LLMs the *application itself* calls (from `.env`, never hard-coded): Groq's `openai/gpt-oss-120b` for
text and its configured vision model for charts, falling back to Google Gemini (`gemini-2.0-flash`) when Groq
can't serve a call, and further to a zero-key spatial-text heuristic if neither vision provider is configured.

## Four Required Cases

_All quotes below are verbatim from the starter dataset PDFs (page numbers as printed in the excerpts)._

### 1. CORROBORATES - Delhivery FY24 revenue, stated two ways

- **Annual Report** (p.22): *"The revenue from operations on consolidated basis for FY24 stood at ₹81,415.38
  million as against ₹72,253.01 million for FY23..."*
- **Q4 FY24 Earnings Presentation** (p.6): *"₹8,142 Cr FY24 revenue from services / YoY: 12.7%"*

₹81,415.38 million normalizes to ₹8,141.538 crore - within 0.01% of ₹8,142 Cr (rounding in the investor deck).
Same subject, metric, period, and scope; the rule engine fires `CORROBORATES` deterministically, no LLM call
needed. Confirmed on a live run producing this exact pair.

### 2. LIKELY_CONTRADICTION - female workforce growth, same report, same page

Both statements appear on **page 8** of the Annual Report:

- *"The number of female workers in our combined on-roll and off-roll workforce increased 60% year-on-year."*
- *"Increasing female workforce • Number of female workers increased by 59% year-on-year"*

The 59% figure is independently corroborated three more times elsewhere in the report (pp.17, 26, 38, including
the headcount math: 5,594 vs 3,519 = 59% growth), making 60% on p.8 the clear outlier. The numeric gap (1.7%
relative) is too large for corroboration but too small to confidently call a real contradiction. Verified live -
uploading the real Annual Report produced exactly this pair and classified it automatically:

```json
{
  "relationship_type": "LIKELY_CONTRADICTION",
  "confidence": 0.6,
  "fact_a": { "predicate": "female workers increase year-on-year", "raw_value": "60%" },
  "fact_b": { "predicate": "female workers increase year-on-year (bullet)", "raw_value": "59%" },
  "reason": "Same subject, metric, period (FY24), scope, and status, but normalized values differ (60.0 vs 59.0 %) with no distinguishing qualifier found - possible rounding or reporting inconsistency."
}
```

### 3. CONTEXT_RECONCILES - Delhivery revenue, split by business segment

The Q4 FY24 deck reports "revenue" for multiple business lines on the same page, same fiscal year. Verified live
- re-ingesting the Q4 deck produced exactly this relationship, unprompted:

```json
{
  "relationship_type": "CONTEXT_RECONCILES",
  "confidence": 0.7,
  "fact_a": { "subject": "Delhivery", "predicate": "revenue", "raw_value": "5,077", "raw_unit": "₹ Cr", "period_label": "FY24", "scope": "Express Parcel" },
  "fact_b": { "subject": "Delhivery", "predicate": "revenue", "raw_value": "1,429", "raw_unit": "₹ Cr", "period_label": "FY24", "scope": "PTL freight" },
  "reason": "Same metric and period but different reporting scope ('Express Parcel' vs 'PTL freight')."
}
```

Same subject, predicate, and period; the numbers differ by 3x, but the evidence names a different business
segment for each - `CONTEXT_RECONCILES` fires deterministically on the scope difference, no LLM call needed.

### 4. Extraction/reasoning failures - three real ones, found by actually running the system

**(a) Sign lost on a parenthesized negative value.** The Annual Report states FY23 EBITDA as "(4,516.08)" ₹
million - accounting notation for a loss. A live run extracted `predicate: "EBITDA loss FY23", numeric_value:
+4516` - the model captured the *word* "loss" but kept the *number* positive, producing a false `CONTRADICTS`
against the Q4 deck's correctly-signed figure. Found by inspecting live relationship output, not code. *Fixed*:
the extraction prompt now states the parentheses-mean-negative convention explicitly.

**(b) Deterministic relationship rules firing on superficially similar but unrelated metrics.** Auditing live
output surfaced false `CONTRADICTS` results: "Adjusted EBITDA" vs. "EBITDA" (different metrics by definition),
YoY vs. QoQ growth (different time bases), "total borrowings" vs. "total income" (sharing only the word
"total"). Character-level string similarity alone can't reliably separate "same metric, paraphrased" from
"different metric, coincidental overlap." *Fixed*: the deterministic shortcut now only fires on near-identical
wording (≥0.92 similarity, with a check for asymmetric qualifiers like "adjusted"); everything else defers to
the LLM. Four regression tests lock in the exact false-positive pairs found - documented as reduced, not solved,
since the underlying signal is inherently imperfect and a different corpus's jargon could surface a new gap.

**(c) A page marked "processed" isn't the same as a page that succeeded.** By design, a failed extraction still
marks a page as attempted (so one bad page doesn't block a document) - but that meant re-uploading an
already-ingested document silently skipped every page that had previously failed to a rate limit, forever. A
live run hit Groq's actual daily token ceiling (`Rate limit reached ... Limit 200000, Used 199879`), and
re-uploading afterward retried nothing. *Fixed*: reprocessing now separately checks which "done" pages actually
logged a `fact_extraction_error` and retries those, covered by an integration test.

**Not yet fixed, found while auditing:** the Prospectus's restated financials present standalone and
consolidated figures in adjacent columns with no other label distinguishing them; raw text extraction scrambles
the headers, so both bases get extracted with the same predicate and `scope: null`, producing a real
`CONTRADICTS` between two figures that aren't actually in tension. This is a table-detection gap, not a
prompt fix - the natural next step is improving `pdfplumber`'s handling of this multi-header layout.

## Limitations and Next Steps

**Works:**
- End-to-end pipeline: upload -> async processing -> evidence -> facts -> relationships -> query, with a real
  job status you can poll. Confirmed on live runs against all three starter PDFs, not just unit tests.
- Deterministic, idempotent ingestion - safe to re-upload, resume after a crash, or resume automatically on the
  next server startup.
- Hybrid retrieval (RRF + rerank), confirmed against a real embedding model, including disambiguating facts that
  share a superficial pattern but mean different things.
- Rule-based relationship classification for confident cases, LLM fallback for the rest, always shown its source
  evidence. Ask (`/query`) stays fast even while a bulk upload runs in the background (own rate-limit lane).
- 120 tests passing without a real API key (deterministic-logic unit tests, an end-to-end pipeline test with
  LLM/embedding calls mocked, a retrieval-accuracy test against the real embedding model, API smoke tests).

**Does not work yet / known gaps:**
- **Groq's free-tier daily token quota (200K/day) is a real constraint, hit repeatedly in testing.** A single
  100-page document can burn through a meaningful fraction of it; a full three-document corpus can exhaust it
  entirely. Multiple keys and the Gemini fallback reduce how often this happens but don't eliminate it - a paid
  key is the real fix for processing many large documents in one session.
- **Text evidence bbox is page-level, not block-level.** Tables/charts get precise bounding boxes; narrative text
  evidence points at the whole page. Matching an LLM's `supporting_quote` back to its originating text block is
  the natural next step.
- **Documents queue one at a time**, even though pages within a document run concurrently. The worker interface
  is narrow enough that a small pool of document-level workers wouldn't touch the pipeline code.
- **No dynamic schema evolution UI.** The schema is generic enough that new fact kinds need no code change, but
  there's no mechanism surfacing "this looks like a new predicate" to a user.
- **Relationship candidates are capped** at the top-5 nearest embedding neighbors per fact, to bound LLM call
  volume - a real recall ceiling on a much larger corpus.
- **No formal retrieval evaluation set** - checked by hand against the four required cases, not a scored
  recall@k benchmark.
- **Extraction is recall-biased toward numeric facts over role/relationship facts.** Auditing the Annual Report
  page-by-page found a genuine miss on a CSR committee table (director names + roles) - exactly the kind of fact
  the assignment calls out as an example. A prompt change with more role-table examples is the natural next
  step.
- Would like to add: block-level evidence highlighting, a small worker pool, and a UI affordance for flagging
  new predicate categories during ingestion.

## Additional Notes

Repository: `https://github.com/colin-110/fact-knowledge-layer` (public).

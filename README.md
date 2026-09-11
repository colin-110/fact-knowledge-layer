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
and `GEMINI_API_KEY` (free tier at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) - a fallback
for *both* vision (chart/figure pages) and text (fact extraction, relationship classification) whenever Groq
can't serve a call, whether that's a missing vision model, a per-minute rate limit, or a daily quota).

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
`CORROBORATES` deterministically, no LLM call needed. Confirmed on a live run producing this exact pair; a
*different* live run (documented in Case 4c below) instead produced a live CORROBORATES between the two
documents' FY23 EBITDA figures when the revenue page happened to fall inside a rate-limited window - which
document/page pairing shows up depends on which pages a given run's Groq quota allowed through, not on whether
the mechanism works.

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

**Verified against the live system**, not just by reading the PDF - uploading the real Annual Report produced
exactly this pair as extracted facts, and the rule engine classified them automatically:

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

The Q4 FY24 Earnings Presentation reports "revenue" for multiple business lines on the same page, for the same
fiscal years. Verified live against the actual running system (not just reasoning about the PDF) - re-ingesting
the real Q4 deck after the false-positive fix below produced exactly this relationship, unprompted:

```json
{
  "relationship_type": "CONTEXT_RECONCILES",
  "confidence": 0.7,
  "fact_a": { "subject": "Delhivery", "predicate": "revenue", "raw_value": "5,077", "raw_unit": "₹ Cr", "period_label": "FY24", "scope": "Express Parcel" },
  "fact_b": { "subject": "Delhivery", "predicate": "revenue", "raw_value": "1,429", "raw_unit": "₹ Cr", "period_label": "FY24", "scope": "PTL freight" },
  "reason": "Same metric and period but different reporting scope ('Express Parcel' vs 'PTL freight')."
}
```

Same subject, predicate, and period; the numbers differ by more than 3x, but the evidence explicitly names a
different business segment for each (`scope` differs) - the rule engine fires `CONTEXT_RECONCILES` with
`context_dimension: scope` deterministically, no LLM call needed. The same pair recurs correctly for FY22 and
FY23 too (three separate relationships, one per fiscal year, all in the live corpus).

### 4. Extraction/reasoning failure - five real ones, all found by actually running the system

**(a) Sign lost on a parenthesized negative value.** The Annual Report states FY24 EBITDA as "1,266.41" and FY23
as "(4,516.08)" (₹ million) - accounting notation where parentheses mean a loss. A live run extracted this as
`predicate: "EBITDA loss FY23", raw_value: "₹4,516 million", numeric_value: +4516` - the model correctly captured
the *word* "loss" in the predicate but extracted the *number* as positive, so it normalized to +451.6 crore
instead of -451.6 crore. Comparing against the Q4 deck's correctly-signed "₹(452) Cr" produced a **CONTRADICTS**
relationship that isn't a real contradiction at all - both sources agree it's a ~452 crore loss, but one fact's
sign was silently wrong. *How this was found:* by actually uploading the real PDF and inspecting the resulting
relationships, not by inspecting code. *Fix applied:* the fact-extraction prompt (`app/pipeline/facts.py`) now
explicitly states the parentheses-mean-negative convention and to sign `numeric_value` accordingly - documented
here rather than silently patched over, since this class of error (an LLM getting the qualitative sense right
but the quantitative encoding wrong) can recur elsewhere and the honest fix is a better prompt, not a claim that
it's now impossible. Re-run the pipeline to see whether it recurs; if it does, that's still an accurate
depiction of current recall, not a broken pipeline.

**(b) A genuinely unreadable chart.** Q4 FY24 Earnings Presentation, page 13: a combined quarterly/yearly
"Adjusted EBITDA (₹ Cr)" bar chart with value and margin labels positioned around the bars and no accompanying
data table on that page. Plain PyMuPDF text extraction returns the numbers stripped of their axis association
and partially concatenated - e.g. four distinct quarterly values collapse into a single token like
`"(125) (67) (25)6"`. The page is correctly flagged `is_visually_complex` and routed through the vision
provider chain (Groq -> Gemini -> spatial-text reflow); when confidence comes back low, the system does not
silently drop it or guess - it stores the evidence row as-is (so you can see exactly what was extracted) and
logs an `extraction_issue` with the reported confidence and uncertainty note, surfaced unfiltered in the
web UI's **Failures** tab and via `GET /extraction-issues`.

**(c) Rate limits, surfaced accurately.** A live 100-page run hit Groq's free-tier quota partway through and
95 of 100 pages failed fact extraction - and an early version of the failure-logging code was *itself* buggy:
it built the `extraction_issue` id without the page number, so every page failing with the identical error
message collided on the same id and got silently deduped by `INSERT OR IGNORE`, making the Failures tab report
2 issues when the real number was 95. Caught by comparing the job's fact count against an earlier successful
run's, not by code review. Fixed in `app/pipeline/ingestion.py` - the failure-handling behavior is only useful
if the failure count itself is trustworthy.

**(d) Relationship reasoning: deterministic classification firing on superficially similar but unrelated
metrics.** Auditing the live relationship output (not the code) surfaced real false `CONTRADICTS` results: two
different people's remuneration compared as if contradictory, "Adjusted EBITDA" vs. "EBITDA" (different metrics
by definition), YoY vs. QoQ growth rates (different time bases, not comparable), and "total borrowings" vs.
"total income" (unrelated metrics sharing only the word "total"). Root cause: `rule_based_classify()`
(`app/pipeline/relationship_engine.py`) fired deterministically from scope+status+period+numeric-diff alone,
without ever confirming the two predicates were the same real-world metric - and character-level string
similarity can't reliably tell "same metric, paraphrased" from "different metric, coincidental wording overlap"
at any single threshold (verified: legitimate paraphrases scored as low as 0.50 similarity; false positives
scored as high as 0.94). *Fix applied:* the deterministic shortcut now only fires on near-identical wording
(≥0.92 similarity, with an explicit check for asymmetric qualifiers like "adjusted"/"YoY"/"QoQ" appearing in
only one side); everything else defers to the LLM, which reads the actual evidence text. 4 regression tests
lock in the exact false-positive pairs found. Documented here rather than claimed fully solved - the underlying
signal (predicate wording) is inherently imperfect, and a corpus with different jargon could surface a new gap
in the same family.

**(e) A genuinely hard table layout: standalone vs. consolidated figures losing their scope tag.** The
Prospectus's restated financial statements present standalone and consolidated figures in adjacent multi-row
column groups (e.g. "As at and for the Financial Year ended March 31, 2021" repeated across several columns
with no other visible label distinguishing which column is which basis). Raw text extraction scrambles these
headers into a single run-on block, so the fact-extraction LLM has no reliable way to recover which column a
number came from - both a standalone and a consolidated "total income" for the same period get extracted with
the same predicate and `scope: null`, and the relationship engine (correctly, given identical wording) compares
them, producing a real `CONTRADICTS` between two figures that aren't actually in tension - it's an extraction
gap (garbled table structure), not a relationship-reasoning gap. *Not fixed*: the honest state is that some of
this document's multi-basis tables don't yield a reliable `scope` today; improving pdfplumber's table detection
specifically for this multi-header layout is the natural next step, not a prompt tweak.

**(f) Groq's free-tier daily token quota, hit mid-session - and a real recovery gap it exposed.**
Re-processing the full three-document corpus to verify (d) and (e) above ran the account's `openai/gpt-oss-120b`
usage to its actual daily ceiling: `Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199879,
Requested 2138`. This is a distinct constraint from the per-minute TPM limit fixed elsewhere in this doc (see
`GROQ_TPM_LIMIT`) - no amount of per-minute pacing prevents a *daily* ceiling, and the system correctly stops
trying rather than looping: once a key/model combination reports a long wait, every further call for it fails
fast for that duration instead of retrying into a wall. While recovering from this, a second real bug surfaced:
a page is marked "processed" the moment it's *attempted*, not when its facts are successfully extracted (by
design - one bad page shouldn't block a whole document) - but that meant a plain re-upload of an already-ingested
document silently skipped every page that had previously failed, forever, since "processed" and "succeeded" were
being treated as the same thing. Live evidence: a rate-limited run left ~95% of a document's pages permanently
stuck at zero facts, and re-uploading the same PDF did not retry a single one of them. *Fixed*: reprocessing now
separately checks which "done" pages actually logged a `fact_extraction_error` (`storage.
pages_with_fact_extraction_errors`) and retries those too, clearing the stale failure record first so a
successful retry doesn't leave a misleading old entry behind - covered by a new integration test
(`test_reprocessing_retries_pages_that_previously_failed_fact_extraction`). `GROQ_API_KEYS` (a second free-tier
account) remains the practical way to avoid the daily ceiling entirely, since quota is tracked per key.

## Limitations and Next Steps

**Works:**
- End-to-end pipeline: upload -> async processing -> evidence -> facts -> relationships -> query, all backed by
  a real job status you can poll. Confirmed on live runs against the actual starter PDFs, not just unit tests.
- Deterministic, idempotent ingestion (safe to re-upload or resume after a crash) - confirmed with an
  integration test that reprocesses a fully-completed document and asserts zero new LLM calls.
- Hybrid retrieval with RRF fusion and reranking - confirmed against a real embedding model + FTS5, including
  disambiguating two facts that share a superficial "X%" pattern but mean completely different things.
- Rule-based relationship classification for the confident cases, LLM fallback for the rest, always shown its
  source evidence rather than reasoning from bare numbers. The 59%/60% LIKELY_CONTRADICTION case fired exactly
  as designed on a live upload of the real Annual Report (see Case 2 above).
- 109 tests, all passing without a real API key (unit tests for the deterministic pieces, an end-to-end
  pipeline test with only the LLM/embedding calls mocked, a retrieval-accuracy test against the real embedding
  model, and API smoke tests covering malformed input, idempotency, and empty-state behavior).

**Does not work yet / known gaps:**
- **Groq's free-tier daily token quota (200K tokens/day) is a real, hit-in-testing constraint**, not a
  theoretical one - a single 100-page document can burn through a meaningful fraction of it, and processing a
  full three-document corpus in one session hit the account's actual daily ceiling (see Case 4f). Once hit,
  every subsequent page fails fact extraction until the quota resets - re-uploading the same PDF after that now
  correctly retries only the pages that actually failed (see Case 4f's fix), rather than needing a full
  re-ingestion or silently staying incomplete forever. `GROQ_API_KEYS` (multiple comma-separated keys,
  round-robined with failover) and the Gemini vision fallback both help reduce how often the ceiling is hit, but
  don't eliminate it - a heavier daily-quota tier or a paid key is the real fix for processing many large
  documents in one session.
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

"""LLM-driven atomic fact extraction from evidence text.

The schema (subject/predicate/value/unit/period/scope/status/qualifiers) is
generic on purpose - nothing here is specific to Delhivery, GDP, or any other
document. The model decides what counts as a fact; we just constrain the
*shape* it comes back in.
"""

from app.llm import complete_json
from app.models import ExtractedFactLLM, FactExtractionResult

SYSTEM_PROMPT = """You extract atomic, verifiable facts from a passage of a source document (a company report, \
prospectus, earnings deck, or a macroeconomic/government report).

A fact is ONE claim with a subject, a predicate (what is being said about the subject), and a value - typically \
numeric (a financial figure, a percentage, a count, a growth rate) but a clear qualitative claim \
(e.g. "director X resigned") is also acceptable if it is materially checkable against other documents.

Rules:
- Extract only what is explicitly stated. Never infer, compute, or guess a value that is not written down.
- Split compound statements into separate atomic facts. "Revenue was X and EBITDA was Y" is two facts, not one.
- Do not extract vague or non-checkable statements ("the company performed well").
- Do not extract table-of-contents entries, section headers, running heads/footers, or navigational text (e.g. \
  "FORWARD-LOOKING STATEMENTS ... 20" listing a section name next to a page number). A page number next to a \
  heading is not a checkable claim about the document's subject matter - it is structure, not content. Only \
  extract facts from the substantive body text itself.
- Do not extract the presentation/report's own boilerplate about itself (safe-harbour notices, "this document is \
  for information purposes only", copyright lines, disclaimers about forward-looking statements) as a fact about \
  the company - these are legal disclaimers, not checkable claims.
- Do not extract more than roughly 15 facts from a single passage - pick the ones that are most concrete and \
  most likely to be checkable against another document (financial figures, growth rates, headcounts, dates, \
  named entities and their status, addresses, percentages).
- For each fact, "subject" is the entity the fact is about (e.g. a company name, a country, a person's role), \
  and "predicate" is the metric or claim (e.g. "revenue from services", "real GDP growth rate", "female employee \
  headcount growth year-on-year"). Keep predicates short and reusable, not full sentences.
- "period_label" should be exactly as written (e.g. "FY24", "FY 2023-24", "Q4 FY24", "2024-25", "as of March 31, 2024").
- "scope" captures what the number applies to if stated (e.g. "consolidated", "standalone", "India", "global", a \
  specific business segment) - null if not stated.
- "status" should reflect whether the source calls this out as an estimate/forecast/actual/provisional/target if \
  stated (e.g. "first advance estimate") - null if not stated.
- Put any other qualifying detail that could distinguish this fact from a superficially similar one elsewhere \
  (e.g. price basis, currency, methodology note, restated/revised) into "qualifiers" as free-form key-value pairs.
- "supporting_quote" MUST be a short verbatim substring copied exactly from the source text below (not paraphrased) \
  that contains the number/claim - this is used to locate the evidence, so it must match the source text exactly.
- Financial reporting convention: a number in parentheses, e.g. "(452)" or "₹(4,516.08)", means NEGATIVE - a loss \
  or an outflow. Set "numeric_value" to the negative number (e.g. -452, -4516.08) even though "raw_value" can \
  keep the parenthesized form as written. Do not extract a parenthesized negative value as if it were positive.
- A value prefixed with ">" or "≥" (e.g. ">2.8Bn", ">33,200") is still a real, checkable number - set "numeric_value" \
  to that stated number (e.g. 2800000000, 33200), not null. It is a stated lower bound, not an unreadable value; \
  note the "more than" qualifier in "qualifiers" if you want to preserve that nuance, but do not omit numeric_value \
  just because the source phrased it as an inequality.

This document may be about anything - do not assume it is about any particular company or topic. Let the text \
itself tell you what facts matter."""


# A page's raw text plus several serialized tables can easily run past several thousand
# characters - a 6000-char cap was silently dropping tables (or even the tail of raw text)
# on exactly the dense financial-statement pages most likely to hold the headline numbers a
# cross-document comparison needs (found by checking a real page's length against the cap
# directly). openai/gpt-oss-120b's context window comfortably fits this; bounded generously
# rather than left unbounded so one pathological page can't blow out latency/cost.
MAX_INPUT_CHARS = 20000


def extract_facts_from_text(evidence_text: str, document_context: str = "") -> list[ExtractedFactLLM]:
    if not evidence_text or len(evidence_text.strip()) < 20:
        return []
    user_prompt = (
        f"Document context: {document_context or 'unknown'}\n\n"
        f"Source text:\n---\n{evidence_text[:MAX_INPUT_CHARS]}\n---\n\n"
        f"Extract atomic facts from this passage as a JSON object with a 'facts' array."
    )
    result = complete_json(SYSTEM_PROMPT, user_prompt, FactExtractionResult)
    return result.facts

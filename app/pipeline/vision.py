"""Chart / figure / infographic extraction via a vision-capable LLM.

Used for pages the extractor flagged as visually complex, where plain text
extraction is unlikely to capture the actual data (bar/line/pie charts,
infographics with icons, etc.). The model is explicitly told never to invent
numbers it cannot read, and to report low confidence / null instead.

Provider chain: Groq's configured vision model first, falling over to Gemini
(GEMINI_API_KEY) if Groq has none available. Both raise a subclass of
RuntimeError when they can't serve the call, so callers can treat "no vision
provider worked" as one condition regardless of which one(s) were tried.
"""

from app import llm
from app.config import GROQ_VISION_MODEL
from app.models import ChartExtractionLLM

SYSTEM_PROMPT = """You are extracting data from a single page image of a financial or economic report \
(company annual report, prospectus, earnings deck, or a government/IMF/RBI macroeconomic report). \
The page has been flagged as visually complex - it likely contains a chart, graph, table-as-image, or infographic.

Rules:
- Extract ONLY numbers, labels, and text that are explicitly visible and legible on the page.
- Never estimate, interpolate, or invent a value you cannot actually read.
- If a chart's exact values are not labeled (e.g. bars with no data labels), set has_extractable_data to false \
  or leave series values null, and explain why in uncertainty_note.
- If the page is mostly plain paragraph text with no real chart, set has_extractable_data to false.
- Report your own confidence (0.0-1.0) honestly - most infographic/icon-based charts without axis labels \
  deserve a LOW confidence score.
- Capture any footnote, source citation, or caveat near the chart in source_note."""


def vision_available() -> bool:
    """Whether at least one vision provider is currently usable - lets callers skip the whole
    attempt cheaply when neither Groq nor Gemini can serve it, instead of paying for a call
    (and its base64 encoding) that's already known to fail."""
    return not llm.is_model_dead(GROQ_VISION_MODEL) or llm.gemini_configured()


def extract_chart_data(image_bytes: bytes, page_context_text: str | None = None) -> ChartExtractionLLM:
    context = (page_context_text or "").strip()[:1500] or "(no machine-readable text was extracted from this page)"
    user_prompt = (
        f"Raw text machine-extracted from this same page (may be partial, out of order, or empty):\n"
        f"---\n{context}\n---\n\n"
        f"Now look at the attached page image and extract any chart/figure/table data it contains."
    )

    groq_error: Exception | None = None
    if not llm.is_model_dead(GROQ_VISION_MODEL):
        try:
            return llm.complete_json_with_image(SYSTEM_PROMPT, user_prompt, image_bytes, ChartExtractionLLM)
        except llm.ModelUnavailableError as exc:
            groq_error = exc  # fall through to Gemini

    if llm.gemini_configured():
        try:
            return llm.complete_json_with_image_gemini(SYSTEM_PROMPT, user_prompt, image_bytes, ChartExtractionLLM)
        except llm.GeminiUnavailableError as exc:
            raise llm.ModelUnavailableError(
                f"No vision provider could serve this page (Groq: {groq_error}; Gemini: {exc})"
            ) from exc

    raise llm.ModelUnavailableError(
        f"No vision provider available (Groq: {groq_error or 'no working vision model'}; "
        f"Gemini: not configured - set GEMINI_API_KEY to enable it)"
    )

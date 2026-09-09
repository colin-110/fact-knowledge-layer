"""Chart / figure / infographic extraction via a vision-capable LLM.

Used for pages the extractor flagged as visually complex, where plain text
extraction is unlikely to capture the actual data (bar/line/pie charts,
infographics with icons, etc.). The model is explicitly told never to invent
numbers it cannot read, and to report low confidence / null instead.
"""

from app.llm import complete_json_with_image
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


def extract_chart_data(image_bytes: bytes, page_context_text: str | None = None) -> ChartExtractionLLM:
    context = (page_context_text or "").strip()[:1500] or "(no machine-readable text was extracted from this page)"
    user_prompt = (
        f"Raw text machine-extracted from this same page (may be partial, out of order, or empty):\n"
        f"---\n{context}\n---\n\n"
        f"Now look at the attached page image and extract any chart/figure/table data it contains."
    )
    return complete_json_with_image(SYSTEM_PROMPT, user_prompt, image_bytes, ChartExtractionLLM)

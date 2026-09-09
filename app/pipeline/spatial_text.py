"""Zero-key fallback for chart-heavy pages: reflow PyMuPDF's word-level output
by spatial position instead of raw reading order.

PyMuPDF's default text extraction reads roughly top-to-bottom, left-to-right
in block order, which scrambles a chart's value labels away from their axis
labels when they're positioned side-by-side or scattered around bars/points
(the exact failure mode behind the Q4 earnings-deck EBITDA chart in the
README's Case 4). Clustering words into rows by y-position, then rows into
visually-separated groups by vertical gaps, keeps a bar's value near its
label and separates unrelated chart regions - useful on its own, and as a
free second input to the vision LLM alongside the actual page image.

This only ever helps or is a no-op: for a real chart with numbers embedded
as PDF text (not rasterized), it recovers structure raw extraction destroys.
For scanned/rasterized charts with no text layer at all, `get_text("words")`
returns nothing and this degrades gracefully to an empty string.
"""

import pymupdf

Word = tuple[float, float, float, float, str, int, int, int]


def _group_into_rows(words: list[Word], y_tolerance: float) -> list[list[Word]]:
    rows: list[list[Word]] = []
    current_row: list[Word] = []
    current_y: float | None = None
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        y0 = w[1]
        if current_y is None or abs(y0 - current_y) <= y_tolerance:
            current_row.append(w)
            current_y = y0 if current_y is None else current_y
        else:
            rows.append(current_row)
            current_row = [w]
            current_y = y0
    if current_row:
        rows.append(current_row)
    for row in rows:
        row.sort(key=lambda w: w[0])
    return rows


def _group_rows_into_clusters(rows: list[list[Word]], cluster_gap: float) -> list[list[list[Word]]]:
    clusters: list[list[list[Word]]] = []
    current_cluster: list[list[Word]] = []
    prev_bottom: float | None = None
    for row in rows:
        row_top = min(w[1] for w in row)
        row_bottom = max(w[3] for w in row)
        if prev_bottom is not None and (row_top - prev_bottom) > cluster_gap and current_cluster:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(row)
        prev_bottom = row_bottom
    if current_cluster:
        clusters.append(current_cluster)
    return clusters


def extract_clustered_text(pdf_path: str, pdf_page_number: int, y_tolerance: float = 3.0, cluster_gap: float = 14.0) -> str:
    """Return page text reflowed into spatially-coherent groups, blank-line separated."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[pdf_page_number - 1]
        words: list[Word] = page.get_text("words")
    finally:
        doc.close()

    if not words:
        return ""

    rows = _group_into_rows(words, y_tolerance)
    clusters = _group_rows_into_clusters(rows, cluster_gap)

    lines: list[str] = []
    for cluster in clusters:
        for row in cluster:
            lines.append(" ".join(w[4] for w in row))
        lines.append("")
    return "\n".join(lines).strip()

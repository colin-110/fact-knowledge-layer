"""Deterministic normalization: money/unit magnitudes and fiscal-period parsing.

Kept separate from the LLM calls on purpose - anything that can be parsed with
a regex should be, so the system's numeric comparisons don't depend on an LLM
being consistent about arithmetic.
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Unit / magnitude normalization
# ---------------------------------------------------------------------------

# All magnitudes expressed as multiples of one unit, for money-like values.
_MAGNITUDE = {
    "thousand": 1_000,
    "lakh": 100_000,
    "lac": 100_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "crore": 10_000_000,
    "cr": 10_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
}

_CANONICAL_MONEY_UNIT = "crore"  # common convention in Indian financial reporting
_CANONICAL_MONEY_MAGNITUDE = _MAGNITUDE[_CANONICAL_MONEY_UNIT]

_CURRENCY_SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}


def _find_magnitude_word(unit_text: str) -> str | None:
    unit_lower = unit_text.lower()
    for word in sorted(_MAGNITUDE, key=len, reverse=True):
        if re.search(rf"\b{word}s?\b", unit_lower):
            return word
    return None


def detect_currency(text: str) -> str | None:
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    m = re.search(r"\b(INR|USD|EUR|GBP|Rs\.?|Rupees?)\b", text, re.IGNORECASE)
    if m:
        token = m.group(1).upper()
        return "INR" if token.startswith("RS") or token.startswith("RUPEE") else token
    return None


@dataclass
class NormalizedUnit:
    normalized_value: float | None
    normalized_unit: str | None


def normalize_value_unit(numeric_value: float | None, raw_unit: str | None, raw_value_text: str = "") -> NormalizedUnit:
    """Convert a money-like value to canonical crore, pass percentages/counts through unchanged."""
    if numeric_value is None:
        return NormalizedUnit(None, None)

    unit_text = raw_unit or ""
    combined = f"{unit_text} {raw_value_text}"

    if "%" in combined or re.search(r"\bpercent(age)?\b", combined, re.IGNORECASE):
        return NormalizedUnit(numeric_value, "%")

    magnitude_word = _find_magnitude_word(unit_text) or _find_magnitude_word(raw_value_text)
    currency = detect_currency(combined)

    if magnitude_word and currency:
        base_value = numeric_value * _MAGNITUDE[magnitude_word]
        normalized_value = base_value / _CANONICAL_MONEY_MAGNITUDE
        return NormalizedUnit(round(normalized_value, 4), f"{currency} {_CANONICAL_MONEY_UNIT}")

    if magnitude_word and not currency:
        # a bare magnitude word with no currency (e.g. "1.2 million users") - normalize to a plain count
        return NormalizedUnit(numeric_value * _MAGNITUDE[magnitude_word], "count")

    # nothing to normalize - pass through with a cleaned-up unit string
    return NormalizedUnit(numeric_value, unit_text.strip() or None)


# ---------------------------------------------------------------------------
# Fiscal period normalization (India FY = April 1 - March 31)
# ---------------------------------------------------------------------------


@dataclass
class NormalizedPeriod:
    period_start: str | None
    period_end: str | None
    period_label: str | None


def _fy_bounds(end_year_2digit_or_4digit: str) -> tuple[str, str]:
    year = int(end_year_2digit_or_4digit)
    if year < 100:
        year += 2000
    start = f"{year - 1}-04-01"
    end = f"{year}-03-31"
    return start, end


_MONTHS = {
    m.lower(): i + 1
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
    )
}


def normalize_period(period_text: str | None) -> NormalizedPeriod:
    if not period_text:
        return NormalizedPeriod(None, None, None)
    text = period_text.strip()

    m = re.search(r"\bH([12])\s*FY\s*(\d{2,4})\b", text, re.IGNORECASE)
    if m:
        half, fy = m.group(1), m.group(2)
        fy_start, fy_end = _fy_bounds(fy)
        if half == "1":
            return NormalizedPeriod(fy_start, f"{fy_start[:4]}-09-30", text)
        return NormalizedPeriod(f"{fy_start[:4]}-10-01", fy_end, text)

    m = re.search(r"\bQ([1-4])\s*FY\s*(\d{2,4})\b", text, re.IGNORECASE)
    if m:
        q, fy = int(m.group(1)), m.group(2)
        fy_start, _ = _fy_bounds(fy)
        start_year = int(fy_start[:4])
        q_start_month = [4, 7, 10, 1][q - 1]
        q_start_year = start_year if q <= 3 else start_year + 1
        q_end_month = q_start_month + 2
        q_end_year = q_start_year
        if q_end_month > 12:
            q_end_month -= 12
            q_end_year += 1
        last_day = 31 if q_end_month in (1, 3, 5, 7, 8, 10, 12) else (28 if q_end_month == 2 else 30)
        start = f"{q_start_year:04d}-{q_start_month:02d}-01"
        end = f"{q_end_year:04d}-{q_end_month:02d}-{last_day:02d}"
        return NormalizedPeriod(start, end, text)

    m = re.search(r"\bFY\s*(\d{4})\s*[-/]\s*(\d{2,4})\b", text, re.IGNORECASE)
    if m:
        end_part = m.group(2)
        end_year = end_part if len(end_part) == 4 else m.group(1)[:2] + end_part
        start, end = _fy_bounds(end_year)
        return NormalizedPeriod(start, end, text)

    m = re.search(r"\bFY\s*(\d{2,4})\b", text, re.IGNORECASE)
    if m:
        start, end = _fy_bounds(m.group(1))
        return NormalizedPeriod(start, end, text)

    m = re.search(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})\b", text, re.IGNORECASE
    )
    if m:
        month = _MONTHS[m.group(1).lower()]
        day, year = int(m.group(2)), int(m.group(3))
        date = f"{year:04d}-{month:02d}-{day:02d}"
        return NormalizedPeriod(date, date, text)

    m = re.fullmatch(r"(19|20)\d{2}", text)
    if m:
        return NormalizedPeriod(f"{text}-01-01", f"{text}-12-31", text)

    # Bare "2024-25" style range with no "FY" prefix - common in Indian govt/RBI reporting,
    # assume Indian fiscal year convention (April-March) rather than a calendar range.
    m = re.fullmatch(r"(20\d{2})[-/](\d{2})", text)
    if m:
        start, end = _fy_bounds(m.group(1)[:2] + m.group(2))
        return NormalizedPeriod(start, end, text)

    return NormalizedPeriod(None, None, text)


_STATUS_KEYWORDS = [
    ("first advance estimate", "first_advance_estimate"),
    ("second advance estimate", "second_advance_estimate"),
    ("advance estimate", "advance_estimate"),
    ("provisional", "provisional"),
    ("projection", "projection"),
    ("projected", "projection"),
    ("forecast", "forecast"),
    ("target", "target"),
    ("estimate", "estimate"),
    ("actual", "actual"),
]


def infer_status(*texts: str | None) -> str | None:
    combined = " ".join(t for t in texts if t).lower()
    for phrase, status in _STATUS_KEYWORDS:
        if phrase in combined:
            return status
    return None

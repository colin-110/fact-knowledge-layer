import pytest

from app.pipeline.normalize import infer_status, normalize_period, normalize_value_unit


class TestMoneyNormalization:
    def test_million_to_crore(self):
        # Delhivery Annual Report: "Revenue from services ... 81,415.38" million INR
        result = normalize_value_unit(81415.38, "INR million", "₹81,415.38 million")
        assert result.normalized_unit == "INR crore"
        assert result.normalized_value == pytest.approx(8141.538, rel=1e-6)

    def test_crore_passthrough(self):
        # Q4 earnings presentation: "₹8,142 Cr"
        result = normalize_value_unit(8142, "Cr", "₹8,142 Cr")
        assert result.normalized_unit == "INR crore"
        assert result.normalized_value == pytest.approx(8142)

    def test_corroboration_pair_matches_within_rounding(self):
        annual_report = normalize_value_unit(81415.38, "INR million", "₹81,415.38 million")
        earnings_deck = normalize_value_unit(8142, "Cr", "₹8,142 Cr")
        rel_diff = abs(annual_report.normalized_value - earnings_deck.normalized_value) / earnings_deck.normalized_value
        assert rel_diff < 0.001

    def test_lakh(self):
        result = normalize_value_unit(50, "lakh", "₹50 lakh")
        assert result.normalized_unit == "INR crore"
        assert result.normalized_value == pytest.approx(0.5)

    def test_billion(self):
        result = normalize_value_unit(1.2, "USD billion", "$1.2 billion")
        assert result.normalized_unit == "USD crore"
        assert result.normalized_value == pytest.approx(120)

    def test_percentage_passthrough(self):
        result = normalize_value_unit(6.4, "%", "6.4 per cent")
        assert result.normalized_unit == "%"
        assert result.normalized_value == 6.4

    def test_percentage_detected_from_value_text_without_unit(self):
        result = normalize_value_unit(60, None, "increased 60% year-on-year")
        assert result.normalized_unit == "%"

    def test_no_currency_or_magnitude_passes_through(self):
        result = normalize_value_unit(5594, "employees", "5,594 employees")
        assert result.normalized_value == 5594
        assert result.normalized_unit == "employees"

    def test_magnitude_abbreviation_glued_directly_to_the_number(self):
        # Real finding from auditing live output: "18.8Mn Sq ft" and "2.8Bn" (no space before the
        # abbreviation) failed to normalize at all - a plain \b word boundary never matches right
        # after a digit, since digits and letters are both \w with no boundary between them. This
        # pattern is common in decks/reports where the magnitude is glued onto the number.
        result = normalize_value_unit(18.8, "Sq ft", "18.8Mn Sq ft")
        assert result.normalized_value == 18_800_000
        assert result.normalized_unit == "count"

    def test_billion_abbreviation_glued_directly_to_the_number(self):
        result = normalize_value_unit(2.8, "Bn", ">2.8Bn")
        assert result.normalized_value == 2_800_000_000
        assert result.normalized_unit == "count"

    def test_none_value_returns_none(self):
        result = normalize_value_unit(None, "INR million", "")
        assert result.normalized_value is None
        assert result.normalized_unit is None

    def test_negative_value_in_parentheses(self):
        # Indian financial reporting convention: "Rs. (452 Cr)" for a loss/negative EBITDA.
        # The LLM is responsible for parsing "(452)" -> numeric_value=-452; normalize_value_unit
        # just needs to carry the sign correctly through the magnitude conversion.
        result = normalize_value_unit(-452, "Cr", "Rs. (452 Cr)")
        assert result.normalized_unit == "INR crore"
        assert result.normalized_value == pytest.approx(-452)

    def test_negative_value_corroborates_with_matching_negative(self):
        a = normalize_value_unit(-452, "Cr", "(452 Cr)")
        b = normalize_value_unit(-4520, "million", "(4,520) million")
        assert a.normalized_value == pytest.approx(b.normalized_value)

    def test_zero_value(self):
        result = normalize_value_unit(0, "%", "0%")
        assert result.normalized_value == 0
        assert result.normalized_unit == "%"


class TestPeriodNormalization:
    def test_fy_two_digit(self):
        p = normalize_period("FY24")
        assert p.period_start == "2023-04-01"
        assert p.period_end == "2024-03-31"

    def test_fy_range_four_digit(self):
        p = normalize_period("FY2023-24")
        assert p.period_start == "2023-04-01"
        assert p.period_end == "2024-03-31"

    def test_fy_slash_four_digit(self):
        # IMF's "FY2024/25"
        p = normalize_period("FY2024/25")
        assert p.period_start == "2024-04-01"
        assert p.period_end == "2025-03-31"

    def test_bare_year_range_assumes_indian_fy(self):
        # RBI's "2024-25"
        p = normalize_period("2024-25")
        assert p.period_start == "2024-04-01"
        assert p.period_end == "2025-03-31"

    def test_q4_fy(self):
        p = normalize_period("Q4 FY24")
        assert p.period_start == "2024-01-01"
        assert p.period_end == "2024-03-31"

    def test_h1_fy(self):
        p = normalize_period("H1 FY25")
        assert p.period_start == "2024-04-01"

    def test_explicit_date(self):
        p = normalize_period("March 31, 2024")
        assert p.period_start == p.period_end == "2024-03-31"

    def test_calendar_year(self):
        p = normalize_period("2024")
        assert p.period_start == "2024-01-01"
        assert p.period_end == "2024-12-31"

    def test_unparseable_period_keeps_label_only(self):
        p = normalize_period("as reported")
        assert p.period_start is None
        assert p.period_label == "as reported"

    def test_none_input(self):
        p = normalize_period(None)
        assert p.period_start is None and p.period_label is None


class TestStatusInference:
    def test_first_advance_estimate(self):
        # Economic Survey 2024-25 phrasing
        assert infer_status("As per the first advance estimates, real GDP is estimated to grow by 6.4 per cent") == (
            "first_advance_estimate"
        )

    def test_actual(self):
        assert infer_status("the actual figure for FY23 was") == "actual"

    def test_no_keyword_returns_none(self):
        assert infer_status("revenue from services") is None

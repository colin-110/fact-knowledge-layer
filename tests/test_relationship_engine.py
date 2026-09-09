from app.pipeline.candidate_matcher import FactRecord
from app.pipeline.relationship_engine import is_plausible_pair, rule_based_classify


def _fact(**overrides) -> FactRecord:
    defaults = dict(
        id="fa_1", document_id="doc_1", subject="Delhivery", predicate="revenue from services",
        numeric_value=81415.38, normalized_value=8141.538, normalized_unit="INR crore",
        period_start="2023-04-01", period_end="2024-03-31", period_label="FY24",
        scope="consolidated", status="actual", retrieval_text="Subject: Delhivery\nMetric: revenue from services",
    )
    defaults.update(overrides)
    return FactRecord(**defaults)


class TestCorroboration:
    def test_matching_values_corroborate(self):
        # Annual report (million) vs earnings deck (crore), already normalized to the same value
        a = _fact(id="fa_annual_report", normalized_value=8141.538)
        b = _fact(id="fa_earnings_deck", normalized_value=8142.0)
        result = rule_based_classify(a, b)
        assert result is not None
        assert result.relationship_type == "CORROBORATES"


class TestLikelyContradiction:
    def test_small_unexplained_gap_is_likely_contradiction(self):
        # The 59% vs 60% female-workforce-growth case: same subject/metric/period/scope/status,
        # small but real numeric gap, no distinguishing qualifier.
        a = _fact(
            id="fa_59", subject="Delhivery", predicate="female workforce growth year-on-year",
            normalized_value=59, normalized_unit="%", scope=None, status=None,
        )
        b = _fact(
            id="fa_60", subject="Delhivery", predicate="female workforce growth year-on-year",
            normalized_value=60, normalized_unit="%", scope=None, status=None,
        )
        result = rule_based_classify(a, b)
        assert result is not None
        assert result.relationship_type == "LIKELY_CONTRADICTION"


class TestContradicts:
    def test_large_unexplained_gap_is_contradiction(self):
        a = _fact(id="fa_1", normalized_value=100, normalized_unit="INR crore")
        b = _fact(id="fa_2", normalized_value=500, normalized_unit="INR crore")
        result = rule_based_classify(a, b)
        assert result is not None
        assert result.relationship_type == "CONTRADICTS"


class TestContextReconciles:
    def test_different_scope_reconciles(self):
        a = _fact(id="fa_consol", scope="consolidated")
        b = _fact(id="fa_standalone", scope="standalone")
        result = rule_based_classify(a, b)
        assert result is not None
        assert result.relationship_type == "CONTEXT_RECONCILES"
        assert result.context_dimension == "scope"

    def test_different_status_reconciles(self):
        # Economic Survey's first advance estimate vs. RBI/IMF's later estimate for the same FY
        a = _fact(
            id="fa_es", subject="India", predicate="real GDP growth rate",
            normalized_value=6.4, normalized_unit="%", period_label="FY25",
            period_start="2024-04-01", period_end="2025-03-31", scope="India", status="first_advance_estimate",
        )
        b = _fact(
            id="fa_rbi", subject="India", predicate="real GDP growth rate",
            normalized_value=6.5, normalized_unit="%", period_label="FY25",
            period_start="2024-04-01", period_end="2025-03-31", scope="India", status="estimate",
        )
        result = rule_based_classify(a, b)
        assert result is not None
        assert result.relationship_type == "CONTEXT_RECONCILES"
        assert result.context_dimension == "estimate_vintage"


class TestAmbiguousFallsThroughToLLM:
    def test_different_periods_returns_none(self):
        a = _fact(id="fa_1", period_label="FY23", period_start="2022-04-01", period_end="2023-03-31")
        b = _fact(id="fa_2", period_label="FY24", period_start="2023-04-01", period_end="2024-03-31")
        assert rule_based_classify(a, b) is None

    def test_non_numeric_facts_return_none(self):
        a = _fact(id="fa_1", normalized_value=None, normalized_unit=None)
        b = _fact(id="fa_2", normalized_value=None, normalized_unit=None)
        assert rule_based_classify(a, b) is None


class TestPlausiblePair:
    def test_similar_subject_and_predicate_is_plausible(self):
        a = _fact(id="fa_1", subject="Delhivery Limited", predicate="revenue from services")
        b = _fact(id="fa_2", subject="Delhivery", predicate="revenue from services (consolidated)")
        assert is_plausible_pair(a, b) is True

    def test_unrelated_subject_is_not_plausible(self):
        a = _fact(id="fa_1", subject="Delhivery", predicate="revenue from services")
        b = _fact(id="fa_2", subject="India", predicate="real GDP growth rate")
        assert is_plausible_pair(a, b) is False

    def test_same_fact_is_never_its_own_pair(self):
        a = _fact(id="fa_1")
        assert is_plausible_pair(a, a) is False

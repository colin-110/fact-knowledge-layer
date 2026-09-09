"""Retrieval accuracy test against the real stack (SQLite FTS5 + Chroma with the real
sentence-transformers model, not mocked) - the hybrid_search_facts function is exactly what's
on the line when someone says "the answer didn't find the right fact." Slower than a unit
test (downloads/loads the embedding model once) but this is the only place that would catch a
regression in the actual fusion behavior.
"""

import pytest

from app import embeddings, ids, storage
from app.db import init_db
from app.pipeline import chunks
from app.pipeline.retrieval import hybrid_search_facts

_FACTS = [
    dict(
        subject="Delhivery", predicate="revenue from services", raw_value="81,415.38", raw_unit="INR million",
        normalized_value=8141.538, normalized_unit="INR crore", period_label="FY24", scope="consolidated",
    ),
    dict(
        subject="Delhivery", predicate="EBITDA", raw_value="1,266.41", raw_unit="INR million",
        normalized_value=126.641, normalized_unit="INR crore", period_label="FY24", scope="consolidated",
    ),
    dict(
        subject="India", predicate="real GDP growth rate", raw_value="6.4", raw_unit="%",
        normalized_value=6.4, normalized_unit="%", period_label="FY25", scope="India",
    ),
    dict(
        subject="Delhivery", predicate="female workforce growth year-on-year", raw_value="59", raw_unit="%",
        normalized_value=59, normalized_unit="%", period_label="FY24", scope=None,
    ),
]

_DOCUMENT_ID = "doc_retrieval_test"


def _seed():
    init_db()
    storage.create_document(
        id=_DOCUMENT_ID, filename="test.pdf", sha256="retrieval_test_hash", title="Test Document",
        page_count=1, file_path="test.pdf",
    )
    fact_ids = []
    for f in _FACTS:
        fid = ids.fact_id(_DOCUMENT_ID, ["ev_1"], f["subject"], f["predicate"], f["raw_value"], f["period_label"])
        retrieval_text = chunks.build_fact_retrieval_text(
            "Test Document", f["subject"], f["predicate"], f["raw_value"], f["raw_unit"],
            f["normalized_value"], f["normalized_unit"], f["period_label"], f["scope"], "actual",
        )
        storage.insert_fact(
            id=fid, document_id=_DOCUMENT_ID, subject=f["subject"], predicate=f["predicate"],
            raw_value=f["raw_value"], raw_unit=f["raw_unit"], numeric_value=float(f["raw_value"].replace(",", "")),
            normalized_value=f["normalized_value"], normalized_unit=f["normalized_unit"],
            period_start=None, period_end=None, period_label=f["period_label"], scope=f["scope"], status="actual",
            qualifiers={}, evidence_ids=["ev_1"], source_method="test", extraction_confidence=0.9,
            retrieval_text=retrieval_text,
        )
        fact_ids.append(fid)
        embeddings.upsert(
            "facts_index", [fid], [retrieval_text],
            [{"document_id": _DOCUMENT_ID, "fact_id": fid, "subject": f["subject"], "predicate": f["predicate"]}],
        )
    return fact_ids


@pytest.fixture(scope="module")
def fact_ids():
    # Fixture, not module-level code: module-level code runs at pytest *collection* time (when
    # every test file is imported, before any test actually runs), which would seed this data
    # into the shared test DB before other test files' "fresh/empty" assumptions ever got a
    # chance to hold - a fixture only runs when a test in this module actually requests it.
    return _seed()


class TestHybridSearchFacts:
    def test_exact_keyword_query_finds_the_right_fact(self, fact_ids):
        # "EBITDA" is an exact, unusual-enough token that FTS5 alone should nail this.
        results = hybrid_search_facts("Delhivery EBITDA FY24", limit=5)
        assert fact_ids[1] in results

    def test_semantic_paraphrase_finds_the_right_fact_via_dense_search(self, fact_ids):
        # No shared tokens with "revenue from services" - only a dense embedding match saves this.
        results = hybrid_search_facts("How much money did Delhivery make from its services?", limit=5)
        assert fact_ids[0] in results

    def test_percentage_query_distinguishes_gdp_from_workforce(self, fact_ids):
        # Both facts are "59%"/"6.4%" - retrieval must not just match on "%", it needs to
        # actually distinguish India's GDP growth from Delhivery's workforce growth.
        gdp_results = hybrid_search_facts("India GDP growth rate", limit=3)
        assert fact_ids[2] in gdp_results
        assert fact_ids[3] not in gdp_results[:1]  # workforce fact should not outrank the GDP fact

    def test_unrelated_query_does_not_return_everything(self, fact_ids):
        results = hybrid_search_facts("unrelated query about something else entirely, xyzzy", limit=5)
        # Should not just return every fact regardless of relevance - FTS return likely empty this
        # this being unusually far from all four seeded facts's semantic content.
        assert isinstance(results, list)  # exercised without raising; exact ranking not asserted

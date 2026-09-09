from app.ids import document_id_from_hash, evidence_id, fact_id, page_id, relationship_id


def test_document_id_deterministic():
    assert document_id_from_hash("abc123") == document_id_from_hash("abc123")


def test_document_id_differs_by_hash():
    assert document_id_from_hash("abc123") != document_id_from_hash("def456")


def test_page_id_deterministic_and_scoped_to_document():
    assert page_id("doc_a", 3) == page_id("doc_a", 3)
    assert page_id("doc_a", 3) != page_id("doc_b", 3)
    assert page_id("doc_a", 3) != page_id("doc_a", 4)


def test_evidence_id_deterministic():
    a = evidence_id("doc_a", 1, "text", 0, "some source text")
    b = evidence_id("doc_a", 1, "text", 0, "some source text")
    assert a == b


def test_evidence_id_differs_by_type():
    text_ev = evidence_id("doc_a", 1, "text", 0, "same seed")
    table_ev = evidence_id("doc_a", 1, "table", 0, "same seed")
    assert text_ev != table_ev


def test_evidence_id_differs_by_page_even_with_identical_seed():
    """Regression: an extraction_issue id built with a hardcoded page_number=0 collided across
    every page that failed with the same error message (e.g. a sustained rate limit), so
    INSERT OR IGNORE silently dropped all but the first - undercounting real failures. The id
    must vary by page even when the seed text (the exception message) is identical."""
    page_5 = evidence_id("doc_a", 5, "fact_extraction_error", 0, "rate limited")
    page_6 = evidence_id("doc_a", 6, "fact_extraction_error", 0, "rate limited")
    assert page_5 != page_6


def test_fact_id_idempotent_reprocessing():
    """Reprocessing the same document/evidence/claim must produce the same fact id,
    so INSERT OR REPLACE overwrites rather than duplicates."""
    ids_first_run = fact_id("doc_a", ["ev_1"], "Delhivery", "revenue from services", "81,415.38", "FY24")
    ids_second_run = fact_id("doc_a", ["ev_1"], "Delhivery", "revenue from services", "81,415.38", "FY24")
    assert ids_first_run == ids_second_run


def test_fact_id_differs_for_different_claims():
    a = fact_id("doc_a", ["ev_1"], "Delhivery", "revenue from services", "81,415.38", "FY24")
    b = fact_id("doc_a", ["ev_1"], "Delhivery", "EBITDA", "1,266.41", "FY24")
    assert a != b


def test_relationship_id_symmetric():
    """A relationship between fact A and B should get the same id regardless of argument order,
    so we never store the same pair twice from opposite directions."""
    assert relationship_id("fa_1", "fa_2") == relationship_id("fa_2", "fa_1")

"""All raw SQL lives here. Pipeline and API code call these functions rather
than writing SQL inline, so the schema only has one place it's touched from.
"""

import json
from datetime import datetime, timezone

from app.db import db_session


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


def get_document_by_hash(sha256: str):
    with db_session() as conn:
        return conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()


def get_document(document_id: str):
    with db_session() as conn:
        return conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()


def create_document(
    id: str, filename: str, sha256: str, title: str | None, page_count: int, file_path: str,
    publisher: str | None = None, publication_date: str | None = None, document_type: str | None = None,
):
    with db_session() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO documents
               (id, filename, sha256, title, publisher, publication_date, document_type, page_count, file_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, filename, sha256, title, publisher, publication_date, document_type, page_count, file_path, now()),
        )


def list_documents():
    with db_session() as conn:
        return conn.execute(
            """SELECT d.*,
                      (SELECT COUNT(*) FROM facts f WHERE f.document_id = d.id) AS fact_count,
                      (SELECT COUNT(*) FROM relationships r
                         JOIN facts fa ON fa.id = r.fact_a_id
                         WHERE fa.document_id = d.id) AS relationship_count,
                      (SELECT status FROM jobs j WHERE j.document_id = d.id ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status
               FROM documents d ORDER BY d.created_at DESC"""
        ).fetchall()


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def get_page(document_id: str, pdf_page_number: int):
    with db_session() as conn:
        return conn.execute(
            "SELECT * FROM pages WHERE document_id = ? AND pdf_page_number = ?", (document_id, pdf_page_number)
        ).fetchone()


def upsert_page(id: str, document_id: str, pdf_page_number: int, printed_page_label: str | None,
                 raw_text: str, width: float, height: float, is_visually_complex: bool):
    with db_session() as conn:
        conn.execute(
            """INSERT INTO pages (id, document_id, pdf_page_number, printed_page_label, raw_text, width, height, is_visually_complex)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(document_id, pdf_page_number) DO UPDATE SET
                 raw_text=excluded.raw_text, width=excluded.width, height=excluded.height,
                 is_visually_complex=excluded.is_visually_complex""",
            (id, document_id, pdf_page_number, printed_page_label, raw_text, width, height, int(is_visually_complex)),
        )


def mark_page_processed(page_id: str):
    with db_session() as conn:
        conn.execute("UPDATE pages SET processed_at = ? WHERE id = ?", (now(), page_id))


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


def insert_evidence(id: str, document_id: str, page_id: str, evidence_type: str, text: str | None,
                     bbox: tuple[float, float, float, float] | None, artifact_path: str | None,
                     extraction_method: str | None, confidence: float | None):
    bbox = bbox or (None, None, None, None)
    with db_session() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO evidence
               (id, document_id, page_id, evidence_type, text, bbox_x0, bbox_y0, bbox_x1, bbox_y1,
                artifact_path, extraction_method, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, document_id, page_id, evidence_type, text, *bbox, artifact_path, extraction_method, confidence, now()),
        )
        if text:
            conn.execute("DELETE FROM evidence_fts WHERE evidence_id = ?", (id,))
            conn.execute("INSERT INTO evidence_fts (evidence_id, text) VALUES (?, ?)", (id, text))


def get_evidence(evidence_id: str):
    with db_session() as conn:
        return conn.execute(
            """SELECT e.*, p.pdf_page_number, d.filename AS document_filename
               FROM evidence e
               JOIN pages p ON p.id = e.page_id
               JOIN documents d ON d.id = e.document_id
               WHERE e.id = ?""",
            (evidence_id,),
        ).fetchone()


def get_evidence_batch(evidence_ids: list[str]):
    if not evidence_ids:
        return []
    with db_session() as conn:
        placeholders = ",".join("?" * len(evidence_ids))
        return conn.execute(
            f"""SELECT e.*, p.pdf_page_number, d.filename AS document_filename
                FROM evidence e
                JOIN pages p ON p.id = e.page_id
                JOIN documents d ON d.id = e.document_id
                WHERE e.id IN ({placeholders})""",
            evidence_ids,
        ).fetchall()


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------


def insert_fact(id: str, document_id: str, subject: str, predicate: str, raw_value: str | None,
                 raw_unit: str | None, numeric_value: float | None, normalized_value: float | None,
                 normalized_unit: str | None, period_start: str | None, period_end: str | None,
                 period_label: str | None, scope: str | None, status: str | None, qualifiers: dict,
                 evidence_ids: list[str], source_method: str, extraction_confidence: float | None,
                 retrieval_text: str):
    with db_session() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO facts
               (id, document_id, subject, predicate, raw_value, raw_unit, numeric_value, normalized_value,
                normalized_unit, period_start, period_end, period_label, scope, status, qualifiers_json,
                evidence_ids_json, source_method, extraction_confidence, retrieval_text, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, document_id, subject, predicate, raw_value, raw_unit, numeric_value, normalized_value,
             normalized_unit, period_start, period_end, period_label, scope, status, json.dumps(qualifiers),
             json.dumps(evidence_ids), source_method, extraction_confidence, retrieval_text, now()),
        )
        conn.execute("DELETE FROM facts_fts WHERE fact_id = ?", (id,))
        conn.execute("INSERT INTO facts_fts (fact_id, retrieval_text) VALUES (?, ?)", (id, retrieval_text))


def get_fact(fact_id: str):
    with db_session() as conn:
        return conn.execute(
            """SELECT f.*, d.filename AS document_filename FROM facts f
               JOIN documents d ON d.id = f.document_id WHERE f.id = ?""",
            (fact_id,),
        ).fetchone()


def get_facts_batch(fact_ids: list[str]):
    if not fact_ids:
        return []
    with db_session() as conn:
        placeholders = ",".join("?" * len(fact_ids))
        return conn.execute(
            f"""SELECT f.*, d.filename AS document_filename FROM facts f
                JOIN documents d ON d.id = f.document_id WHERE f.id IN ({placeholders})""",
            fact_ids,
        ).fetchall()


def list_facts(document_id: str | None = None, search: str | None = None, limit: int = 200):
    query = """SELECT f.*, d.filename AS document_filename FROM facts f
               JOIN documents d ON d.id = f.document_id WHERE 1=1"""
    params: list = []
    if document_id:
        query += " AND f.document_id = ?"
        params.append(document_id)
    if search:
        query += " AND (f.subject LIKE ? OR f.predicate LIKE ? OR f.retrieval_text LIKE ?)"
        like = f"%{search}%"
        params.extend([like, like, like])
    query += " ORDER BY f.created_at DESC LIMIT ?"
    params.append(limit)
    with db_session() as conn:
        return conn.execute(query, params).fetchall()


def list_facts_for_document(document_id: str):
    with db_session() as conn:
        return conn.execute("SELECT * FROM facts WHERE document_id = ?", (document_id,)).fetchall()


def list_all_facts():
    with db_session() as conn:
        return conn.execute("SELECT * FROM facts").fetchall()


# ---------------------------------------------------------------------------
# relationships
# ---------------------------------------------------------------------------


def insert_relationship(id: str, fact_a_id: str, fact_b_id: str, relationship_type: str,
                         confidence: float | None, context_dimension: str | None, reason: str | None, method: str) -> bool:
    """Returns True if a new row was actually inserted, False if this pair already existed
    (e.g. the other worker thread for the reverse fact pair won the race first) - callers use
    this rather than assuming every call persists something, since relationship candidates are
    found from both facts' perspective and processed concurrently."""
    with db_session() as conn:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO relationships
               (id, fact_a_id, fact_b_id, relationship_type, confidence, context_dimension, reason, method, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, fact_a_id, fact_b_id, relationship_type, confidence, context_dimension, reason, method, now()),
        )
        return cursor.rowcount > 0


def relationship_exists(fact_a_id: str, fact_b_id: str) -> bool:
    a, b = sorted([fact_a_id, fact_b_id])
    with db_session() as conn:
        row = conn.execute(
            "SELECT 1 FROM relationships WHERE (fact_a_id = ? AND fact_b_id = ?) OR (fact_a_id = ? AND fact_b_id = ?)",
            (a, b, b, a),
        ).fetchone()
    return row is not None


def list_relationships(relationship_type: str | None = None, limit: int = 200):
    query = "SELECT * FROM relationships WHERE 1=1"
    params: list = []
    if relationship_type:
        query += " AND relationship_type = ?"
        params.append(relationship_type)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db_session() as conn:
        return conn.execute(query, params).fetchall()


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def create_job(id: str, document_id: str):
    with db_session() as conn:
        conn.execute(
            "INSERT INTO jobs (id, document_id, status, created_at) VALUES (?, ?, 'queued', ?)",
            (id, document_id, now()),
        )


def update_job(id: str, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    with db_session() as conn:
        conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", (*fields.values(), id))


def get_job(job_id: str):
    with db_session() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def get_latest_job_for_document(document_id: str):
    with db_session() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE document_id = ? ORDER BY created_at DESC LIMIT 1", (document_id,)
        ).fetchone()


# ---------------------------------------------------------------------------
# extraction issues (surfaced in the Failures UI tab)
# ---------------------------------------------------------------------------


def insert_extraction_issue(id: str, document_id: str, page_id: str | None, evidence_id: str | None,
                             issue_type: str, description: str, confidence: float | None):
    with db_session() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO extraction_issues
               (id, document_id, page_id, evidence_id, issue_type, description, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, document_id, page_id, evidence_id, issue_type, description, confidence, now()),
        )


def list_extraction_issues(document_id: str | None = None, limit: int = 200):
    query = "SELECT * FROM extraction_issues WHERE 1=1"
    params: list = []
    if document_id:
        query += " AND document_id = ?"
        params.append(document_id)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with db_session() as conn:
        return conn.execute(query, params).fetchall()


def pages_with_fact_extraction_errors(document_id: str) -> set[str]:
    """Page ids for this document that logged a fact_extraction_error - a page marked
    'processed' does not mean its facts were actually extracted (a transient LLM failure
    still lets the page finish and move on, by design, so one bad page can't block a whole
    document), so a normal reprocess needs this to know which 'done' pages are actually worth
    retrying. fact_extraction_error rows carry evidence_id, not page_id directly (the failure
    happens per evidence unit, before a page-level id would be known), so this joins through
    evidence to resolve it."""
    with db_session() as conn:
        rows = conn.execute(
            """SELECT DISTINCT e.page_id AS page_id
               FROM extraction_issues ei
               JOIN evidence e ON e.id = ei.evidence_id
               WHERE ei.document_id = ? AND ei.issue_type = 'fact_extraction_error' AND e.page_id IS NOT NULL""",
            (document_id,),
        ).fetchall()
    return {r["page_id"] for r in rows}


def clear_fact_extraction_errors(document_id: str, page_ids: set[str]) -> None:
    """Delete stale fact_extraction_error rows for pages about to be retried, so a successful
    retry doesn't leave a misleading 'this page failed' entry sitting in the Failures tab
    forever alongside (or instead of) the fresh result."""
    if not page_ids:
        return
    with db_session() as conn:
        placeholders = ",".join("?" * len(page_ids))
        conn.execute(
            f"""DELETE FROM extraction_issues
                WHERE document_id = ? AND issue_type = 'fact_extraction_error'
                AND evidence_id IN (SELECT id FROM evidence WHERE page_id IN ({placeholders}))""",
            (document_id, *page_ids),
        )

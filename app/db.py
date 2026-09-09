import sqlite3
from contextlib import contextmanager

from app.config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT UNIQUE NOT NULL,
    title TEXT,
    publisher TEXT,
    publication_date TEXT,
    document_type TEXT,
    page_count INTEGER,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    pdf_page_number INTEGER NOT NULL,
    printed_page_label TEXT,
    raw_text TEXT,
    width REAL,
    height REAL,
    is_visually_complex INTEGER DEFAULT 0,
    processed_at TEXT,
    UNIQUE(document_id, pdf_page_number)
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    page_id TEXT NOT NULL REFERENCES pages(id),
    evidence_type TEXT NOT NULL,
    text TEXT,
    bbox_x0 REAL, bbox_y0 REAL, bbox_x1 REAL, bbox_y1 REAL,
    artifact_path TEXT,
    extraction_method TEXT,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    raw_value TEXT,
    raw_unit TEXT,
    numeric_value REAL,
    normalized_value REAL,
    normalized_unit TEXT,
    period_start TEXT,
    period_end TEXT,
    period_label TEXT,
    scope TEXT,
    status TEXT,
    qualifiers_json TEXT,
    evidence_ids_json TEXT NOT NULL,
    source_method TEXT,
    extraction_confidence REAL,
    retrieval_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact_id UNINDEXED, retrieval_text
);

CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    evidence_id UNINDEXED, text
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    fact_a_id TEXT NOT NULL REFERENCES facts(id),
    fact_b_id TEXT NOT NULL REFERENCES facts(id),
    relationship_type TEXT NOT NULL,
    confidence REAL,
    context_dimension TEXT,
    reason TEXT,
    method TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(fact_a_id, fact_b_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    status TEXT NOT NULL DEFAULT 'queued',
    stage TEXT,
    progress INTEGER DEFAULT 0,
    total_pages INTEGER DEFAULT 0,
    pages_processed INTEGER DEFAULT 0,
    facts_extracted INTEGER DEFAULT 0,
    relationships_found INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS extraction_issues (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    page_id TEXT,
    evidence_id TEXT,
    issue_type TEXT NOT NULL,
    description TEXT,
    confidence REAL,
    created_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_session():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_session() as conn:
        conn.executescript(SCHEMA)

"""Embedded vector index (Chroma, local persistence - no server) plus the
local sentence-transformers embedding model.

Two collections: `facts_index` and `evidence_index`. Both are upserted with
deterministic IDs (the fact/evidence row's own id), so reprocessing a document
overwrites rather than duplicates vectors.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, EMBEDDING_MODEL

_model: SentenceTransformer | None = None
_chroma_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_collection(name: str):
    return get_chroma_client().get_or_create_collection(name)


def _clean_metadata(meta: dict) -> dict:
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


def upsert(collection_name: str, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    if not ids:
        return
    collection = get_collection(collection_name)
    embeddings = get_model().encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[_clean_metadata(m) for m in metadatas],
    )


def query(collection_name: str, query_text: str, n_results: int = 20, where: dict | None = None) -> dict:
    collection = get_collection(collection_name)
    if collection.count() == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    embedding = get_model().encode([query_text], normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=embedding,
        n_results=min(n_results, max(collection.count(), 1)),
        where=where,
    )

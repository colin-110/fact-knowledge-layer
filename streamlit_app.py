"""Streamlit UI for the Fact Knowledge Layer.

Talks to the FastAPI backend over HTTP (no direct DB access) so the UI
process and the API/worker process can run independently. Run the API
first (`uvicorn app.main:app`), then this (`streamlit run streamlit_app.py`).
"""

import os
import time

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")


def api_get(path: str, **params):
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json=None, files=None):
    resp = requests.post(f"{API_BASE}{path}", json=json, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


RELATIONSHIP_COLORS = {
    "CORROBORATES": "#1a7f37",
    "CONTRADICTS": "#cf222e",
    "LIKELY_CONTRADICTION": "#bf8700",
    "CONTEXT_RECONCILES": "#0969da",
}


def relationship_badge(rel_type: str) -> str:
    color = RELATIONSHIP_COLORS.get(rel_type, "#57606a")
    return f":{'green' if color=='#1a7f37' else 'orange' if color=='#bf8700' else 'red' if color=='#cf222e' else 'blue'}[**{rel_type}**]"


st.title("Fact Knowledge Layer")
st.caption("Upload a PDF, extract evidence-grounded facts, and see how facts corroborate, contradict, or reconcile across documents.")

try:
    api_get("/health")
except Exception:
    st.error(f"Cannot reach the API at {API_BASE}. Start it with `uvicorn app.main:app --reload` first.")
    st.stop()

tab_docs, tab_facts, tab_rels, tab_ask, tab_failures = st.tabs(
    ["📄 Documents", "🔎 Facts", "🔗 Relationships", "💬 Ask", "⚠️ Failures / Extraction Quality"]
)

# ---------------------------------------------------------------------------
# Documents tab
# ---------------------------------------------------------------------------

with tab_docs:
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Any PDF - not limited to the starter dataset", type=["pdf"])
    if uploaded is not None and st.button("Upload and process", type="primary"):
        with st.spinner("Uploading..."):
            result = api_post("/documents", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
        st.session_state["last_job_id"] = result["job_id"]
        st.success(f"Queued. document_id={result['document_id']}  job_id={result['job_id']}")

    if "last_job_id" in st.session_state:
        job_id = st.session_state["last_job_id"]
        placeholder = st.empty()
        if st.button("Refresh job status"):
            pass
        job = api_get(f"/jobs/{job_id}")
        with placeholder.container():
            st.progress(min(job["progress"], 100) / 100)
            st.write(
                f"**Status:** {job['status']} | **Stage:** {job['stage']} | "
                f"**Pages:** {job['pages_processed']}/{job['total_pages']} | "
                f"**Facts extracted:** {job['facts_extracted']} | "
                f"**Relationships found:** {job['relationships_found']}"
            )
            if job["error"]:
                st.error(job["error"])

    st.divider()
    st.subheader("Documents in the knowledge layer")
    docs = api_get("/documents")
    if not docs:
        st.info("No documents uploaded yet.")
    else:
        st.dataframe(
            [
                {
                    "Title": d["title"] or d["filename"], "Pages": d["page_count"],
                    "Status": d["latest_job_status"], "Facts": d["fact_count"],
                    "Relationships": d["relationship_count"], "Uploaded": d["created_at"][:19],
                }
                for d in docs
            ],
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------------------
# Facts tab
# ---------------------------------------------------------------------------

with tab_facts:
    st.subheader("Search facts")
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Search (subject, predicate, or retrieval text)", "")
    with col2:
        docs = api_get("/documents")
        doc_options = {"(all documents)": None, **{d["title"] or d["filename"]: d["id"] for d in docs}}
        doc_choice = st.selectbox("Document", list(doc_options.keys()))

    facts = api_get("/facts", search=search or None, document_id=doc_options[doc_choice])
    st.write(f"{len(facts)} fact(s)")

    for f in facts[:100]:
        with st.expander(f"**{f['subject']}** — {f['predicate']} = {f['raw_value']} {f['raw_unit'] or ''}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Normalized:** {f['normalized_value']} {f['normalized_unit'] or ''}")
                st.markdown(f"**Period:** {f['period_label'] or '—'}")
                st.markdown(f"**Scope:** {f['scope'] or '—'}")
                st.markdown(f"**Status:** {f['status'] or '—'}")
                if f["qualifiers"]:
                    st.markdown(f"**Qualifiers:** `{f['qualifiers']}`")
                st.markdown(f"**Extraction confidence:** {f['extraction_confidence']}")
            with c2:
                st.markdown(f"**Source:** {f['document_filename']}")
                for ev in f["evidence"]:
                    st.markdown(f"*Page {ev['page_number']}, {ev['evidence_type']}, method={ev['extraction_method']}, confidence={ev['confidence']}*")
                    st.text((ev["text"] or "")[:600])
                    if ev["artifact_path"]:
                        try:
                            img = requests.get(f"{API_BASE}/evidence/{ev['id']}/artifact", timeout=30).content
                            st.image(img, caption="Source page render", use_container_width=True)
                        except Exception:
                            pass

# ---------------------------------------------------------------------------
# Relationships tab
# ---------------------------------------------------------------------------

with tab_rels:
    st.subheader("Cross-document relationships")
    rel_filter = st.selectbox(
        "Filter", ["(all)", "CORROBORATES", "CONTRADICTS", "LIKELY_CONTRADICTION", "CONTEXT_RECONCILES"]
    )
    relationships = api_get("/relationships", relationship_type=None if rel_filter == "(all)" else rel_filter)
    st.write(f"{len(relationships)} relationship(s)")

    for r in relationships[:100]:
        fa, fb = r["fact_a"], r["fact_b"]
        st.markdown(f"### {relationship_badge(r['relationship_type'])}  (confidence: {r['confidence']:.2f})" if r["confidence"] else f"### {relationship_badge(r['relationship_type'])}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{fa['subject']} — {fa['predicate']}**")
            st.markdown(f"{fa['raw_value']} {fa['raw_unit'] or ''}  ·  {fa['period_label'] or ''}  ·  {fa['scope'] or ''}")
            st.caption(fa["document_filename"])
        with c2:
            st.markdown(f"**{fb['subject']} — {fb['predicate']}**")
            st.markdown(f"{fb['raw_value']} {fb['raw_unit'] or ''}  ·  {fb['period_label'] or ''}  ·  {fb['scope'] or ''}")
            st.caption(fb["document_filename"])
        st.info(r["reason"])
        st.divider()

# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------

with tab_ask:
    st.subheader("Ask a question")
    question = st.text_input("e.g. What was Delhivery's FY24 revenue?")
    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving and reasoning..."):
            answer = api_post("/query", json={"question": question})
        st.markdown(f"### Answer\n{answer['answer']}")

        if answer["facts"]:
            st.markdown("**Facts used:**")
            for f in answer["facts"]:
                st.markdown(f"- {f['subject']} — {f['predicate']}: **{f['value']}** ({f['period'] or '—'}, {f['scope'] or '—'})")

        if answer["evidence"]:
            st.markdown("**Source evidence:**")
            for e in answer["evidence"]:
                st.markdown(f"- *{e['document']}, page {e['page']}*")
                st.text((e["text"] or "")[:400])

        if answer["relationships"]:
            st.markdown("**Related relationships:**")
            for r in answer["relationships"]:
                st.markdown(f"- {relationship_badge(r['type'])}: {r['reason']}")

# ---------------------------------------------------------------------------
# Failures tab
# ---------------------------------------------------------------------------

with tab_failures:
    st.subheader("Extraction issues and low-confidence results")
    st.caption(
        "Every extraction failure the system noticed is logged here, not hidden. This includes low-confidence "
        "chart/figure reads, page processing errors, and fact-extraction errors."
    )
    issues = api_get("/extraction-issues")
    if not issues:
        st.info("No extraction issues logged yet.")
    else:
        for issue in issues[:100]:
            st.warning(f"**{issue['issue_type']}** (confidence: {issue['confidence']})\n\n{issue['description']}")

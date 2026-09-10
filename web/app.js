// Plain fetch-based frontend for the Fact Knowledge Layer API.
// Served from the same origin as the API, so all paths below are relative - no CORS involved.

const AUTO_REFRESH_MS = 4000;
const JOB_POLL_MS = 1200;

let activeTab = "documents";
let activeJobId = null;
let jobPollTimer = null;
let autoRefreshTimer = null;

async function apiGet(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v);
  });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json();
}

async function apiPostJson(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} -> ${res.status}`);
  return res.json();
}

async function apiPostForm(path, formData) {
  const res = await fetch(path, { method: "POST", body: formData });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `POST ${path} -> ${res.status}`);
  }
  return res.json();
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

const TAB_META = {
  documents: { title: "Documents", subtitle: "Upload a PDF and track how it's processed into evidence-grounded facts." },
  facts: { title: "Facts", subtitle: "Every atomic claim extracted so far, each grounded in its source evidence." },
  relationships: { title: "Relationships", subtitle: "Where facts across documents corroborate, contradict, or reconcile through context." },
  ask: { title: "Ask", subtitle: "Ask a question and get an answer grounded only in retrieved facts and evidence." },
  failures: { title: "Failures", subtitle: "Low-confidence reads, unavailable models, and extraction errors - nothing hidden." },
};

function initTabs() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });
}

function setActiveTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
  const meta = TAB_META[tab];
  if (meta) {
    document.getElementById("page-title").textContent = meta.title;
    document.getElementById("page-subtitle").textContent = meta.subtitle;
  }
  refreshActiveTab();
}

function refreshActiveTab() {
  if (activeTab === "documents") refreshDocuments();
  else if (activeTab === "facts") refreshFacts();
  else if (activeTab === "relationships") refreshRelationships();
  else if (activeTab === "failures") refreshIssues();
}

// ---------------------------------------------------------------------------
// API health
// ---------------------------------------------------------------------------

async function checkHealth() {
  const el = document.getElementById("api-status");
  try {
    await apiGet("/health");
    el.textContent = "API connected";
    el.className = "api-status ok";
    return true;
  } catch {
    el.textContent = "API unreachable - start it with: uvicorn app.main:app --reload";
    el.className = "api-status down";
    return false;
  }
}

// ---------------------------------------------------------------------------
// Documents tab
// ---------------------------------------------------------------------------

async function refreshDocuments() {
  let docs;
  try {
    docs = await apiGet("/documents");
  } catch {
    return;
  }
  const tbody = document.querySelector("#documents-table tbody");
  const empty = document.getElementById("documents-empty");
  tbody.innerHTML = "";
  empty.hidden = docs.length > 0;
  for (const d of docs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(d.title || d.filename)}</td>
      <td>${d.page_count ?? "-"}</td>
      <td>${escapeHtml(d.latest_job_status || "-")}</td>
      <td>${d.fact_count}</td>
      <td>${d.relationship_count}</td>
      <td>${escapeHtml((d.created_at || "").slice(0, 19).replace("T", " "))}</td>
    `;
    tbody.appendChild(tr);
  }
  populateDocumentFilter(docs);
}

function populateDocumentFilter(docs) {
  const select = document.getElementById("fact-document-filter");
  const current = select.value;
  select.innerHTML = '<option value="">(all documents)</option>';
  for (const d of docs) {
    const opt = document.createElement("option");
    opt.value = d.id;
    opt.textContent = d.title || d.filename;
    select.appendChild(opt);
  }
  select.value = current;
}

function setupUploadForm() {
  const form = document.getElementById("upload-form");
  const errorBox = document.getElementById("upload-error");
  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const filenameLabel = document.getElementById("dropzone-filename");
  const uploadBtn = document.getElementById("upload-btn");

  function setSelectedFile(file) {
    if (file && file.type === "application/pdf") {
      filenameLabel.textContent = file.name;
      uploadBtn.disabled = false;
    } else {
      filenameLabel.textContent = file ? "Please choose a PDF file." : "";
      uploadBtn.disabled = true;
    }
  }

  fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      setSelectedFile(file);
    }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.hidden = true;
    if (!fileInput.files.length) return;
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    uploadBtn.disabled = true;
    try {
      const result = await apiPostForm("/documents", formData);
      fileInput.value = "";
      setSelectedFile(null);
      startJobPolling(result.job_id);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
      uploadBtn.disabled = false;
    }
  });
}

function startJobPolling(jobId) {
  activeJobId = jobId;
  document.getElementById("active-job-card").hidden = false;
  if (jobPollTimer) clearInterval(jobPollTimer);
  pollJobOnce();
  jobPollTimer = setInterval(pollJobOnce, JOB_POLL_MS);
}

async function pollJobOnce() {
  if (!activeJobId) return;
  let job;
  try {
    job = await apiGet(`/jobs/${activeJobId}`);
  } catch {
    return;
  }
  const fill = document.getElementById("job-progress-fill");
  const statusLine = document.getElementById("job-status-line");
  const errorBox = document.getElementById("job-error");
  fill.style.width = `${Math.min(job.progress, 100)}%`;
  statusLine.textContent =
    `Status: ${job.status} | Stage: ${job.stage || "-"} | ` +
    `Pages: ${job.pages_processed}/${job.total_pages} | ` +
    `Facts extracted: ${job.facts_extracted} | Relationships found: ${job.relationships_found}`;
  if (job.error) {
    errorBox.textContent = job.error;
    errorBox.hidden = false;
  } else {
    errorBox.hidden = true;
  }
  if (job.status === "completed" || job.status === "failed") {
    clearInterval(jobPollTimer);
    jobPollTimer = null;
    activeJobId = null;
    refreshDocuments();
  }
}

// ---------------------------------------------------------------------------
// Facts tab
// ---------------------------------------------------------------------------

let factSearchDebounce = null;

function setupFactsControls() {
  document.getElementById("fact-search").addEventListener("input", () => {
    clearTimeout(factSearchDebounce);
    factSearchDebounce = setTimeout(refreshFacts, 300);
  });
  document.getElementById("fact-document-filter").addEventListener("change", refreshFacts);
}

let lastFactsSignature = null;

async function refreshFacts() {
  const search = document.getElementById("fact-search").value.trim();
  const documentId = document.getElementById("fact-document-filter").value;
  let facts;
  try {
    facts = await apiGet("/facts", { search: search || null, document_id: documentId || null });
  } catch {
    return;
  }
  document.getElementById("fact-count").textContent = facts.length
    ? `${facts.length} fact(s)`
    : "No facts match yet - try a different search, or upload a document first.";

  // Skip re-rendering entirely when nothing changed - the common case on an auto-refresh
  // tick - so an open <details> card (or scroll position) is never disturbed. When the data
  // really did change, re-render but restore which cards were open beforehand.
  const signature = JSON.stringify(facts.map((f) => f.id));
  if (signature === lastFactsSignature) return;
  lastFactsSignature = signature;

  const container = document.getElementById("facts-list");
  const openIds = new Set([...container.querySelectorAll("details[open]")].map((d) => d.dataset.factId));
  container.innerHTML = "";
  for (const f of facts.slice(0, 100)) {
    const card = renderFactCard(f);
    if (openIds.has(f.id)) card.open = true;
    container.appendChild(card);
  }
}

function confidenceLabel(confidence) {
  if (confidence === null || confidence === undefined) return "not reported";
  const pct = Math.round(confidence * 100);
  if (confidence >= 0.85) return `${pct}% (high)`;
  if (confidence >= 0.5) return `${pct}% (medium)`;
  return `${pct}% (low)`;
}

function sourcePageLinks(documentId, pageNumber) {
  const imgUrl = `/documents/${documentId}/pages/${pageNumber}/image`;
  const pdfUrl = `/documents/${documentId}/file#page=${pageNumber}`;
  return `<a href="${imgUrl}" target="_blank" rel="noopener">View page image</a> ·
          <a href="${pdfUrl}" target="_blank" rel="noopener">Open PDF at page ${pageNumber}</a>`;
}

function renderFactCard(f) {
  const details = document.createElement("details");
  details.className = "fact-card";
  details.dataset.factId = f.id;
  const value = `${f.raw_value || ""} ${f.raw_unit || ""}`.trim();
  const summary = document.createElement("summary");
  summary.textContent = `${f.subject} - ${f.predicate} = ${value}`;
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "fact-detail";

  const left = document.createElement("div");
  left.innerHTML = `
    <div><strong>Normalized:</strong> ${escapeHtml(f.normalized_value)} ${escapeHtml(f.normalized_unit || "")}</div>
    <div><strong>Period:</strong> ${escapeHtml(f.period_label || "-")}</div>
    <div><strong>Scope:</strong> ${escapeHtml(f.scope || "-")}</div>
    <div><strong>Status:</strong> ${escapeHtml(f.status || "-")}</div>
    <div><strong>Extraction confidence:</strong> ${escapeHtml(confidenceLabel(f.extraction_confidence))}</div>
    ${Object.keys(f.qualifiers || {}).length ? `<div><strong>Qualifiers:</strong> ${escapeHtml(JSON.stringify(f.qualifiers))}</div>` : ""}
  `;

  const right = document.createElement("div");
  right.innerHTML = `<div><strong>Source:</strong> ${escapeHtml(f.document_filename)}</div>`;
  for (const ev of f.evidence || []) {
    const block = document.createElement("div");
    block.className = "evidence-block";
    block.innerHTML = `
      <div class="rel-source">Page ${ev.page_number}, ${escapeHtml(ev.evidence_type)}, method=${escapeHtml(ev.extraction_method)}, confidence=${escapeHtml(confidenceLabel(ev.confidence))}</div>
      <div>${escapeHtml((ev.text || "").slice(0, 500))}</div>
      <div class="evidence-links">${sourcePageLinks(ev.document_id, ev.page_number)}</div>
      ${ev.artifact_path ? `<img loading="lazy" src="/evidence/${ev.id}/artifact" alt="source page render" />` : ""}
    `;
    right.appendChild(block);
  }

  body.appendChild(left);
  body.appendChild(right);
  details.appendChild(body);
  return details;
}

// ---------------------------------------------------------------------------
// Relationships tab
// ---------------------------------------------------------------------------

function setupRelationshipControls() {
  document.getElementById("relationship-filter").addEventListener("change", refreshRelationships);
}

function badgeClass(type) {
  return `badge badge-${type.toLowerCase()}`;
}

async function refreshRelationships() {
  const type = document.getElementById("relationship-filter").value;
  let rels;
  try {
    rels = await apiGet("/relationships", { relationship_type: type || null });
  } catch {
    return;
  }
  document.getElementById("relationship-count").textContent = rels.length
    ? `${rels.length} relationship(s)`
    : "No relationships found yet - they appear once at least two related facts exist.";
  const container = document.getElementById("relationships-list");
  container.innerHTML = "";
  for (const r of rels.slice(0, 100)) {
    const card = document.createElement("div");
    card.className = "relationship-card";
    const confidenceStr = r.confidence !== null && r.confidence !== undefined ? ` (confidence: ${r.confidence.toFixed(2)})` : "";
    card.innerHTML = `
      <div><span class="${badgeClass(r.relationship_type)}">${escapeHtml(r.relationship_type)}</span>${confidenceStr}</div>
      <div class="rel-facts">
        <div>
          <strong>${escapeHtml(r.fact_a.subject)} - ${escapeHtml(r.fact_a.predicate)}</strong><br/>
          ${escapeHtml(r.fact_a.raw_value)} ${escapeHtml(r.fact_a.raw_unit || "")} · ${escapeHtml(r.fact_a.period_label || "")} · ${escapeHtml(r.fact_a.scope || "")}
          <div class="rel-source">${escapeHtml(r.fact_a.document_filename)}</div>
        </div>
        <div>
          <strong>${escapeHtml(r.fact_b.subject)} - ${escapeHtml(r.fact_b.predicate)}</strong><br/>
          ${escapeHtml(r.fact_b.raw_value)} ${escapeHtml(r.fact_b.raw_unit || "")} · ${escapeHtml(r.fact_b.period_label || "")} · ${escapeHtml(r.fact_b.scope || "")}
          <div class="rel-source">${escapeHtml(r.fact_b.document_filename)}</div>
        </div>
      </div>
      <div class="rel-reason">${escapeHtml(r.reason || "")}</div>
    `;
    container.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// Ask tab
// ---------------------------------------------------------------------------

function setupAskForm() {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("ask-input");

  async function ask(question) {
    if (!question) return;
    const resultBox = document.getElementById("ask-result");
    resultBox.innerHTML = "<p class=\"hint\">Retrieving and reasoning...</p>";
    try {
      const answer = await apiPostJson("/query", { question });
      renderAskResult(answer);
    } catch (err) {
      resultBox.innerHTML = `<div class="error-box">${escapeHtml(err.message)}</div>`;
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    ask(input.value.trim());
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.q;
      ask(chip.dataset.q);
    });
  });
}

function renderAskResult(answer) {
  const resultBox = document.getElementById("ask-result");
  let html = `<h3>Answer</h3><div class="answer-text">${escapeHtml(answer.answer)}</div>`;

  if (answer.facts && answer.facts.length) {
    html += "<h3>Facts used</h3><ul>";
    for (const f of answer.facts) {
      html += `<li>${escapeHtml(f.subject)} - ${escapeHtml(f.predicate)}: <strong>${escapeHtml(f.value)}</strong> (${escapeHtml(f.period || "-")}, ${escapeHtml(f.scope || "-")})</li>`;
    }
    html += "</ul>";
  }

  if (answer.evidence && answer.evidence.length) {
    html += "<h3>Source evidence</h3><ul>";
    for (const e of answer.evidence) {
      html += `<li><em>${escapeHtml(e.document)}, page ${e.page}</em><br/>${escapeHtml((e.text || "").slice(0, 300))}</li>`;
    }
    html += "</ul>";
  }

  if (answer.relationships && answer.relationships.length) {
    html += "<h3>Related relationships</h3><ul>";
    for (const r of answer.relationships) {
      html += `<li><span class="${badgeClass(r.type)}">${escapeHtml(r.type)}</span> ${escapeHtml(r.reason || "")}</li>`;
    }
    html += "</ul>";
  }

  resultBox.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Failures tab
// ---------------------------------------------------------------------------

async function refreshIssues() {
  let issues;
  try {
    issues = await apiGet("/extraction-issues");
  } catch {
    return;
  }
  const container = document.getElementById("issues-list");
  const empty = document.getElementById("issues-empty");
  container.innerHTML = "";
  empty.hidden = issues.length > 0;
  for (const issue of issues.slice(0, 100)) {
    const card = document.createElement("div");
    card.className = "issue-card";
    card.innerHTML = `
      <div class="issue-type">${escapeHtml(issue.issue_type)} (confidence: ${escapeHtml(issue.confidence)})</div>
      <div class="issue-desc">${escapeHtml(issue.description)}</div>
    `;
    container.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  setupUploadForm();
  setupFactsControls();
  setupRelationshipControls();
  setupAskForm();

  await checkHealth();
  refreshActiveTab();

  // Skip network calls while the tab isn't visible - no point polling a page nobody's
  // looking at, and it stops piling up requests in a background browser tab.
  autoRefreshTimer = setInterval(() => {
    if (document.visibilityState !== "visible") return;
    checkHealth();
    refreshActiveTab();
  }, AUTO_REFRESH_MS);

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      checkHealth();
      refreshActiveTab();
    }
  });
});

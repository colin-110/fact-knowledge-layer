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
// Source PDF side panel - opens the actual uploaded PDF (whole document, or
// jumped straight to a specific page) inline rather than a new tab, so a fact,
// a relationship's evidence, or a row in the Documents list all lead to the
// same place: the real source page, not just a rendered crop.
// ---------------------------------------------------------------------------

function openPdfPanel(documentId, filename, pageNumber) {
  const url = `/documents/${documentId}/file` + (pageNumber ? `#page=${pageNumber}` : "");
  document.getElementById("pdf-panel-frame").src = url;
  document.getElementById("pdf-panel-filename").textContent = filename || "Source PDF";
  document.getElementById("pdf-panel-page").textContent = pageNumber ? `Page ${pageNumber}` : "Full document";
  document.getElementById("pdf-panel-open-tab").href = url;
  document.getElementById("pdf-panel").hidden = false;
  document.getElementById("pdf-panel-backdrop").hidden = false;
}

function closePdfPanel() {
  document.getElementById("pdf-panel").hidden = true;
  document.getElementById("pdf-panel-backdrop").hidden = true;
  document.getElementById("pdf-panel-frame").src = "about:blank";
}

function setupPdfPanel() {
  document.getElementById("pdf-panel-close").addEventListener("click", closePdfPanel);
  document.getElementById("pdf-panel-backdrop").addEventListener("click", closePdfPanel);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePdfPanel();
  });
  // Delegated so it keeps working for cards re-rendered/reconciled after this listener is
  // attached once - no need to re-bind per card.
  document.addEventListener("click", (e) => {
    const trigger = e.target.closest("[data-open-pdf]");
    if (!trigger) return;
    e.preventDefault();
    const page = trigger.dataset.page ? Number(trigger.dataset.page) : null;
    openPdfPanel(trigger.dataset.documentId, trigger.dataset.filename, page);
  });
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
    const title = escapeHtml(d.title || d.filename);
    tr.innerHTML = `
      <td><button type="button" class="doc-title-link" data-open-pdf data-document-id="${d.id}" data-filename="${title}">${title}</button></td>
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

  const visible = facts.slice(0, 100);
  // Skip re-rendering entirely when nothing changed - the common case on an auto-refresh tick.
  const signature = JSON.stringify(visible.map((f) => f.id));
  if (signature === lastFactsSignature) return;
  lastFactsSignature = signature;

  const container = document.getElementById("facts-list");
  // Reconcile in place rather than wiping and rebuilding: while a document is still being
  // ingested, new facts keep shifting everyone else's position, so the fact list changes on
  // nearly every poll. Recreating every <details> node each time destroyed an open card's
  // `open` state the instant it happened to land mid-poll - closing whatever the user had just
  // expanded a moment before. Reusing existing nodes (and only moving/adding/removing what
  // actually changed) means an open card is never torn down just because something else in
  // the list changed.
  const existingById = new Map([...container.children].map((el) => [el.dataset.factId, el]));
  const keep = new Set();
  let cursor = container.firstChild;
  for (const f of visible) {
    keep.add(f.id);
    const card = existingById.get(f.id) || renderFactCard(f);
    if (card === cursor) {
      cursor = cursor.nextSibling;
    } else {
      container.insertBefore(card, cursor);
    }
  }
  for (const [id, el] of existingById) {
    if (!keep.has(id)) el.remove();
  }
}

function confidenceLabel(confidence) {
  if (confidence === null || confidence === undefined) return "not reported";
  const pct = Math.round(confidence * 100);
  if (confidence >= 0.85) return `${pct}% (high)`;
  if (confidence >= 0.5) return `${pct}% (medium)`;
  return `${pct}% (low)`;
}

function sourcePageLinks(documentId, filename, pageNumber) {
  const imgUrl = `/documents/${documentId}/pages/${pageNumber}/image`;
  const safeFilename = escapeHtml(filename || "");
  return `<a href="${imgUrl}" target="_blank" rel="noopener">View page image</a> ·
          <a href="#" data-open-pdf data-document-id="${documentId}" data-filename="${safeFilename}" data-page="${pageNumber}">Open PDF at page ${pageNumber}</a>`;
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
      <div class="evidence-links">${sourcePageLinks(ev.document_id, ev.document_filename, ev.page_number)}</div>
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

function factSourceLine(f) {
  // The first evidence unit carries the page this fact actually came from - link straight to
  // it (opens in the side PDF panel) rather than just naming the document.
  const ev = (f.evidence || [])[0];
  const filename = escapeHtml(f.document_filename || "");
  if (!ev) return filename;
  return `<a href="#" data-open-pdf data-document-id="${f.document_id}" data-filename="${filename}" data-page="${ev.page_number}">${filename}, page ${ev.page_number}</a>`;
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
          <div class="rel-source">${factSourceLine(r.fact_a)}</div>
        </div>
        <div>
          <strong>${escapeHtml(r.fact_b.subject)} - ${escapeHtml(r.fact_b.predicate)}</strong><br/>
          ${escapeHtml(r.fact_b.raw_value)} ${escapeHtml(r.fact_b.raw_unit || "")} · ${escapeHtml(r.fact_b.period_label || "")} · ${escapeHtml(r.fact_b.scope || "")}
          <div class="rel-source">${factSourceLine(r.fact_b)}</div>
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
  setupPdfPanel();

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

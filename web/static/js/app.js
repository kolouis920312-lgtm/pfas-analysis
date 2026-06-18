/* PFAS 資料分析工具 — 前端邏輯
   對應原 Tkinter 程式：方法切換、依 spec 自動產生參數、驗證、執行、顯示結果。 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const api = (path, opts) => fetch(path, opts).then(async (r) => {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
});

const state = {
  meta: null,
  methods: [],
  current: null,     // 目前選的 spec
  dataset: null,     // 資料 token
  columns: [],       // 目前資料的欄名
};

// ───────────────────────────────── 初始化
document.addEventListener("DOMContentLoaded", init);

async function init() {
  try {
    state.meta = await api("/api/meta");
  } catch (e) {
    setStatus("無法載入方法清單：" + e.message, "err");
    return;
  }
  state.methods = state.meta.methods;
  buildMethodList();
  buildThemeControls();
  buildOutputControls();
  restorePrefs();
  bindEvents();

  if (state.methods.length) selectMethod(state.methods[0].key);
}

function bindEvents() {
  $("#file-input").addEventListener("change", onUpload);
  $("#demo-btn").addEventListener("click", onDemo);
  $("#run-btn").addEventListener("click", onRun);
  $("#reset-theme-btn").addEventListener("click", resetTheme);
  // 說明書小視窗
  $("#manual-btn").addEventListener("click", openManual);
  $("#manual-modal").addEventListener("click", (e) => {
    if (e.target.dataset && e.target.dataset.close !== undefined) closeManual();
  });
  document.querySelectorAll(".modal-tab").forEach((t) =>
    t.addEventListener("click", () => switchManualTab(t.dataset.tab)));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeManual();
  });
  // 主題/輸出變更 → 存到 localStorage
  ["font", "primary", "accent", "cmap_sequential", "cmap_diverging",
   "cmap_categorical", "image_format", "dpi"].forEach((id) => {
    $("#" + id).addEventListener("change", savePrefs);
  });
}

// ───────────────────────────────── 方法清單
function buildMethodList() {
  const ul = $("#method-list");
  ul.innerHTML = "";
  state.methods.forEach((m) => {
    const li = document.createElement("li");
    li.textContent = m.name;
    li.dataset.key = m.key;
    li.addEventListener("click", () => selectMethod(m.key));
    ul.appendChild(li);
  });
}

function selectMethod(key) {
  const spec = state.methods.find((m) => m.key === key);
  if (!spec) return;
  state.current = spec;
  document.querySelectorAll("#method-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.key === key));
  $("#method-summary").textContent = spec.summary;

  // 說明書按鈕（有內容才顯示）
  const hasManual = spec.manual && (spec.manual.beginner || spec.manual.pro);
  $("#manual-btn").classList.toggle("hidden", !hasManual);

  // 範本下載連結
  const tl = $("#template-link");
  if (spec.has_template) {
    tl.classList.remove("hidden");
    tl.href = `/api/template/${spec.key}`;
  } else {
    tl.classList.add("hidden");
  }

  buildParams();
  refreshValidation();
  savePrefs();
}

// ───────────────────────────────── 參數面板（依 spec 自動產生）
function buildParams() {
  const wrap = $("#params");
  wrap.innerHTML = "";
  const spec = state.current;
  if (!spec.params.length) {
    wrap.innerHTML = `<p class="hint">此方法沒有可調參數。</p>`;
    return;
  }
  spec.params.forEach((p) => {
    const row = document.createElement("div");
    row.className = "param-row";

    const label = document.createElement("div");
    label.className = "param-label";
    label.textContent = p.label;
    row.appendChild(label);

    const control = document.createElement("div");
    control.className = "param-control";

    let input;
    if (p.kind === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!p.default;
    } else if (p.kind === "choice") {
      input = makeSelect(p.choices, p.default == null ? "" : String(p.default));
    } else if (p.kind === "column") {
      input = makeSelect(columnOptions(p), columnDefault(p));
      input.addEventListener("change", refreshValidation);
    } else {
      input = document.createElement("input");
      input.type = (p.kind === "int" || p.kind === "float") ? "number" : "text";
      if (p.kind === "int") input.step = "1";
      if (p.kind === "float") input.step = "any";
      if (p.minimum != null) input.min = p.minimum;
      if (p.maximum != null) input.max = p.maximum;
      input.value = p.default == null ? "" : p.default;
    }
    input.classList.add("param-input");
    input.dataset.key = p.key;
    input.dataset.kind = p.kind;
    // 改參數就重新驗證（debounce）
    input.addEventListener("input", debouncedValidate);
    control.appendChild(input);

    if (p.help) {
      const help = document.createElement("div");
      help.className = "param-help";
      help.textContent = p.help;
      control.appendChild(help);
    }
    row.appendChild(control);
    wrap.appendChild(row);
  });
}

function columnOptions(p) {
  return (p.optional ? ["(無)"] : []).concat(state.columns);
}

function columnDefault(p) {
  const opts = columnOptions(p);
  if (p.default != null && opts.includes(p.default)) return p.default;
  return p.optional ? "(無)" : (opts[0] || "");
}

function makeSelect(options, value) {
  const sel = document.createElement("select");
  options.forEach((o) => {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o === "" ? "（空）" : o;
    sel.appendChild(opt);
  });
  if (value !== undefined && options.includes(value)) sel.value = value;
  return sel;
}

// 上傳/示範資料後，重設欄位型參數的選項（對應原 _refresh_columns）
function refreshColumnParams() {
  if (!state.current) return;
  state.current.params.forEach((p) => {
    if (p.kind !== "column") return;
    const sel = document.querySelector(`#params [data-key="${cssEscape(p.key)}"]`);
    if (!sel) return;
    const prev = sel.value;
    const opts = columnOptions(p);
    sel.innerHTML = "";
    opts.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o;
      opt.textContent = o === "" ? "（空）" : o;
      sel.appendChild(opt);
    });
    sel.value = opts.includes(prev) ? prev : columnDefault(p);
  });
}

function cssEscape(s) {
  return (window.CSS && CSS.escape) ? CSS.escape(s) : s.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

// ───────────────────────────────── 蒐集設定
function collectParams() {
  const out = {};
  document.querySelectorAll("#params .param-input").forEach((inp) => {
    out[inp.dataset.key] = (inp.dataset.kind === "bool") ? inp.checked : inp.value;
  });
  return out;
}

function collectTheme() {
  return {
    font_family: $("#font").value,
    primary: $("#primary").value,
    accent: $("#accent").value,
    cmap_sequential: $("#cmap_sequential").value,
    cmap_diverging: $("#cmap_diverging").value,
    cmap_categorical: $("#cmap_categorical").value,
  };
}

// ───────────────────────────────── 資料來源
async function onUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  setStatus("上傳中…", "busy");
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    applyDataset(res);
    setStatus("已載入資料", "ok");
  } catch (e) {
    setStatus("讀取失敗：" + e.message, "err");
    alert("無法讀取 CSV：\n" + e.message);
  } finally {
    ev.target.value = "";
  }
}

async function onDemo() {
  if (!state.current) return;
  setStatus("產生示範資料…", "busy");
  try {
    const res = await api("/api/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method: state.current.key }),
    });
    applyDataset(res);
    setStatus("已載入示範資料", "ok");
  } catch (e) {
    setStatus("失敗：" + e.message, "err");
  }
}

function applyDataset(res) {
  state.dataset = res.dataset;
  state.columns = res.preview.columns;
  $("#data-label").textContent =
    `${res.label}　(${res.preview.nrows} 列 × ${res.preview.ncols} 欄)`;
  renderPreview(res.preview);
  refreshColumnParams();
  refreshValidation();
}

function renderPreview(preview) {
  const box = $("#data-preview");
  let html = "<table><thead><tr>";
  preview.columns.forEach((c) => (html += `<th>${escapeHtml(c)}</th>`));
  html += "</tr></thead><tbody>";
  preview.rows.forEach((row) => {
    html += "<tr>";
    row.forEach((v) => (html += `<td>${v == null ? "" : escapeHtml(String(v))}</td>`));
    html += "</tr>";
  });
  html += "</tbody></table>";
  if (preview.nrows > preview.rows.length) {
    html += `<div class="more">… 共 ${preview.nrows} 列，僅顯示前 ${preview.rows.length} 列</div>`;
  }
  box.innerHTML = html;
  box.classList.remove("hidden");
}

// ───────────────────────────────── 驗證
let validateTimer = null;
function debouncedValidate() {
  clearTimeout(validateTimer);
  validateTimer = setTimeout(refreshValidation, 350);
}

async function refreshValidation() {
  if (!state.current) return;
  const box = $("#validation");
  if (!state.dataset) {
    box.textContent = "請先選擇 CSV 或按「用示範資料」。";
    box.className = "validation";
    return;
  }
  try {
    const res = await api("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        method: state.current.key,
        dataset: state.dataset,
        params: collectParams(),
      }),
    });
    const rep = res.report;
    box.textContent = rep.text;
    box.className = "validation " + (rep.ok ? "ok" : (rep.errors.length ? "err" : ""));
  } catch (e) {
    box.textContent = "驗證失敗：" + e.message;
    box.className = "validation err";
  }
}

// ───────────────────────────────── 執行
async function onRun() {
  if (!state.current) return;
  if (!state.dataset) {
    alert("請先選擇 CSV 或按「用示範資料」。");
    return;
  }
  showOverlay(true, "執行中…（複雜方法可能需數十秒）");
  $("#run-btn").disabled = true;
  setStatus("執行中…", "busy");
  try {
    const res = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        method: state.current.key,
        dataset: state.dataset,
        params: collectParams(),
        theme: collectTheme(),
        image_format: $("#image_format").value,
        dpi: $("#dpi").value,
      }),
    });
    renderResults(res);
    setStatus(`完成 ✓  圖 ${res.previews.length}、表 ${res.tables.length}`, "ok");
  } catch (e) {
    setStatus("執行失敗：" + e.message, "err");
    if (state.current) refreshValidation();
    alert("執行失敗：\n" + e.message);
  } finally {
    showOverlay(false);
    $("#run-btn").disabled = false;
  }
}

function renderResults(res) {
  $("#results").classList.remove("hidden");
  $("#results-meta").textContent =
    `${state.current.name}　·　圖 ${res.previews.length} 張、表 ${res.tables.length} 個`;

  // 圖像預覽
  const prev = $("#previews");
  prev.innerHTML = "";
  if (!res.previews.length) {
    prev.innerHTML = `<p class="hint">此方法沒有圖像輸出。</p>`;
  }
  res.previews.forEach((f) => {
    const item = document.createElement("div");
    item.className = "preview-item";
    item.innerHTML =
      `<div class="cap">${escapeHtml(f.name)}</div>` +
      `<a href="${f.url}" target="_blank"><img src="${f.url}" alt="${escapeHtml(f.name)}"></a>`;
    prev.appendChild(item);
  });

  // 下載（圖檔 + 表格 + 其他）
  const dl = $("#downloads");
  dl.innerHTML = "";
  const all = [].concat(res.figures, res.tables, res.extras);
  if (!all.length) {
    dl.innerHTML = `<p class="hint">沒有可下載的檔案。</p>`;
  }
  all.forEach((f) => {
    const a = document.createElement("a");
    a.href = f.url + "?dl=1";
    a.textContent = "⬇ " + f.name;
    dl.appendChild(a);
  });

  // 紀錄 + 摘要
  $("#log").textContent =
    (res.log || "") + "\n\n【摘要】\n" + (res.summary || "");

  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ───────────────────────────────── 使用說明書（小視窗）
function openManual() {
  const spec = state.current;
  if (!spec || !spec.manual) return;
  $("#manual-title").textContent = spec.name + " · 使用說明書";
  $("#manual-beginner").innerHTML = renderMarkdown(spec.manual.beginner) + exampleFigureHtml(spec);
  $("#manual-pro").innerHTML = renderMarkdown(spec.manual.pro);
  $("#manual-glossary").innerHTML = renderMarkdown((state.meta && state.meta.glossary) || "");
  switchManualTab("beginner");
  $("#manual-modal").classList.remove("hidden");
}

function closeManual() {
  $("#manual-modal").classList.add("hidden");
}

function switchManualTab(tab) {
  document.querySelectorAll(".modal-tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === tab));
  $("#manual-beginner").classList.toggle("hidden", tab !== "beginner");
  $("#manual-pro").classList.toggle("hidden", tab !== "pro");
  $("#manual-glossary").classList.toggle("hidden", tab !== "glossary");
}

// 「快速上手」底部附一張範例輸出圖（若該方法有縮圖）
function exampleFigureHtml(spec) {
  const url = state.meta && state.meta.examples && state.meta.examples[spec.key];
  if (!url) return "";
  return '<figure class="manual-example">'
    + '<figcaption>▼ 範例輸出（用「示範資料」跑出來的樣子，點圖可放大）</figcaption>'
    + '<a href="' + url + '" target="_blank" rel="noopener">'
    + '<img src="' + url + '" alt="' + escapeHtml(spec.name) + ' 範例輸出"></a>'
    + '</figure>';
}

// 輕量 Markdown 渲染（## 標題 / - 清單 / 1. 編號 / **粗體** / `碼` / > 重點），無外部相依
function renderMarkdown(md) {
  if (!md || !md.trim()) return "<p class='hint'>（此方法尚無說明內容）</p>";
  const escAll = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const inline = (s) => escAll(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  let html = "", para = [], inUl = false, inOl = false;
  const flushPara = () => { if (para.length) { html += "<p>" + inline(para.join(" ")) + "</p>"; para = []; } };
  const closeLists = () => {
    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) { flushPara(); closeLists(); continue; }
    let m;
    if ((m = line.match(/^###\s+(.*)/))) { flushPara(); closeLists(); html += "<h5>" + inline(m[1]) + "</h5>"; }
    else if ((m = line.match(/^##\s+(.*)/))) { flushPara(); closeLists(); html += "<h4>" + inline(m[1]) + "</h4>"; }
    else if ((m = line.match(/^>\s?(.*)/))) { flushPara(); closeLists(); html += "<blockquote>" + inline(m[1]) + "</blockquote>"; }
    else if ((m = line.match(/^\s*[-•]\s+(.*)/))) {
      flushPara(); if (inOl) { html += "</ol>"; inOl = false; }
      if (!inUl) { html += "<ul>"; inUl = true; } html += "<li>" + inline(m[1]) + "</li>";
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      flushPara(); if (inUl) { html += "</ul>"; inUl = false; }
      if (!inOl) { html += "<ol>"; inOl = true; } html += "<li>" + inline(m[1]) + "</li>";
    } else { para.push(line.trim()); }
  }
  flushPara(); closeLists();
  return html;
}

// ───────────────────────────────── 主題 / 輸出控制
function buildThemeControls() {
  fillDatalist("#font-list", state.meta.fonts);
  fillSelect("#cmap_sequential", state.meta.cmap_sequential);
  fillSelect("#cmap_diverging", state.meta.cmap_diverging);
  fillSelect("#cmap_categorical", state.meta.cmap_categorical);
  applyThemeDefaults();
}

function applyThemeDefaults() {
  const t = state.meta.theme_default;
  $("#font").value = t.font_family;
  $("#primary").value = t.primary;
  $("#accent").value = t.accent;
  $("#cmap_sequential").value = t.cmap_sequential;
  $("#cmap_diverging").value = t.cmap_diverging;
  $("#cmap_categorical").value = t.cmap_categorical;
}

function resetTheme() {
  applyThemeDefaults();
  savePrefs();
}

function buildOutputControls() {
  fillSelect("#image_format", state.meta.image_formats);
  $("#image_format").value = "png";
}

// ───────────────────────────────── localStorage 記住設定
function savePrefs() {
  const prefs = {
    method: state.current ? state.current.key : null,
    theme: collectTheme(),
    image_format: $("#image_format").value,
    dpi: $("#dpi").value,
  };
  try { localStorage.setItem("pfas_prefs", JSON.stringify(prefs)); } catch (e) {}
}

function restorePrefs() {
  let prefs;
  try { prefs = JSON.parse(localStorage.getItem("pfas_prefs") || "null"); } catch (e) {}
  if (!prefs) return;
  const t = prefs.theme || {};
  if (t.font_family) $("#font").value = t.font_family;
  if (t.primary) $("#primary").value = t.primary;
  if (t.accent) $("#accent").value = t.accent;
  if (t.cmap_sequential) $("#cmap_sequential").value = t.cmap_sequential;
  if (t.cmap_diverging) $("#cmap_diverging").value = t.cmap_diverging;
  if (t.cmap_categorical) $("#cmap_categorical").value = t.cmap_categorical;
  if (prefs.image_format) $("#image_format").value = prefs.image_format;
  if (prefs.dpi) $("#dpi").value = prefs.dpi;
  if (prefs.method && state.methods.some((m) => m.key === prefs.method)) {
    // 在 init 末端 selectMethod 前先記住；這裡直接切過去
    setTimeout(() => selectMethod(prefs.method), 0);
  }
}

// ───────────────────────────────── 小工具
function fillSelect(sel, items) {
  const el = $(sel);
  el.innerHTML = "";
  items.forEach((it) => {
    const opt = document.createElement("option");
    opt.value = it; opt.textContent = it;
    el.appendChild(opt);
  });
}
function fillDatalist(sel, items) {
  const el = $(sel);
  el.innerHTML = "";
  items.forEach((it) => {
    const opt = document.createElement("option");
    opt.value = it;
    el.appendChild(opt);
  });
}
function setStatus(msg, kind) {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status" + (kind ? " " + kind : "");
}
function showOverlay(show, text) {
  $("#overlay").classList.toggle("hidden", !show);
  if (text) $("#overlay-text").textContent = text;
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

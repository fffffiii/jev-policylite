const $ = (id) => document.getElementById(id);
const state = { latency: [], modelLoaded: false };

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function formatMs(value) {
  if (value == null) return "--";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(0)} ms`;
}
function formatPercent(value, digits = 1) { return value == null ? "--" : `${(value * 100).toFixed(digits)}%`; }
function showToast(message, error = false) {
  const toast = $("toast"); toast.textContent = message; toast.className = `toast show${error ? " error" : ""}`;
  clearTimeout(showToast.timer); showToast.timer = setTimeout(() => toast.className = "toast", 3500);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `请求失败 (${response.status})`);
  }
  return response.json();
}

function setPreview(file) {
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) { showToast("图片不能超过 10 MB", true); return; }
  const preview = $("preview");
  preview.src = URL.createObjectURL(file);
  $("dropzone").classList.add("has-image");
}
function clearImage(event) {
  event?.preventDefault();
  $("imageInput").value = ""; $("preview").removeAttribute("src"); $("dropzone").classList.remove("has-image");
}

function drawChart() {
  const canvas = $("latencyChart");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#292e29"; ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(f => { ctx.beginPath(); ctx.moveTo(0, height * f); ctx.lineTo(width, height * f); ctx.stroke(); });
  if (state.latency.length < 2) return;
  const max = Math.max(...state.latency, 1), min = Math.min(...state.latency, 0), range = Math.max(max - min, 1);
  ctx.beginPath();
  state.latency.forEach((v, i) => {
    const x = (i / (state.latency.length - 1)) * width;
    const y = height - 10 - ((v - min) / range) * (height - 20);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = "#c7ff4a"; ctx.lineWidth = 2; ctx.stroke();
  ctx.lineTo(width, height); ctx.lineTo(0, height); ctx.closePath();
  const gradient = ctx.createLinearGradient(0, 0, 0, height); gradient.addColorStop(0, "rgba(199,255,74,.18)"); gradient.addColorStop(1, "rgba(199,255,74,0)");
  ctx.fillStyle = gradient; ctx.fill();
}

function renderRecent(records) {
  const tbody = $("recentRows");
  if (!records.length) return;
  tbody.innerHTML = records.map(row => `<tr><td>${new Date(row.timestamp).toLocaleTimeString()}</td><td>${escapeHtml(row.policy_id.replace("pilot-", "").replace("-v1", ""))}</td><td class="decision-${row.decision}">${row.decision === "pass" ? "通过" : "违规"}</td><td>${(row.score * 100).toFixed(1)}%</td><td>${formatMs(row.latency_ms)}</td><td>${formatMs(row.inference_ms)}</td></tr>`).join("");
}

function renderStatus(data) {
  $("statusDot").classList.toggle("live", data.model_ready); $("serviceText").textContent = data.model_ready ? "服务在线" : "模型加载中";
  $("heroReady").textContent = data.model_ready ? "READY" : "LOADING";
  const uptime = Math.floor(data.uptime_seconds); $("uptime").textContent = `UP ${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m`;
  $("heroLatency").textContent = formatMs(data.latency_ms.p50);
  $("latencySummary").textContent = `P50 ${formatMs(data.latency_ms.p50)} · P95 ${formatMs(data.latency_ms.p95)}`;
  if (data.latency_ms.latest != null) { state.latency.push(data.latency_ms.latest); state.latency = state.latency.slice(-40); drawChart(); }
  const gpu = data.gpus.find(item => item.index === 0) || data.gpus[0];
  if (gpu) {
    const used = gpu.memory_used_mb / 1024, total = gpu.memory_total_mb / 1024;
    $("gpuMemory").textContent = `${used.toFixed(1)} GB`; $("gpuMemoryDetail").textContent = `${used.toFixed(1)} / ${total.toFixed(1)} GB`;
    $("gpuMemoryBar").style.width = `${(used / total) * 100}%`; $("gpuUtil").textContent = `${gpu.utilization_percent.toFixed(0)}%`;
    $("gpuThermal").textContent = `${gpu.temperature_c.toFixed(0)} °C · ${gpu.power_w.toFixed(0)} W`;
  }
  $("processMemory").textContent = `${(data.process.rss_mb / 1024).toFixed(1)} GB`; $("cudaMemory").textContent = `CUDA 已分配 ${data.cuda.allocated_mb.toFixed(0)} MB`;
  $("requestRate").textContent = data.requests_per_minute; $("requestTotal").textContent = `累计 ${data.total_requests} · 失败 ${data.failed_requests}`;
  $("gpuList").innerHTML = data.gpus.map(gpu => `<div class="gpu-item"><div class="gpu-item-head"><span>GPU ${gpu.index} · ${gpu.name.replace("NVIDIA GeForce ", "")}</span><span>${gpu.utilization_percent.toFixed(0)}%</span></div><div class="gpu-item-meta"><span>${(gpu.memory_used_mb/1024).toFixed(1)} / ${(gpu.memory_total_mb/1024).toFixed(1)} GB</span><span>${gpu.temperature_c.toFixed(0)} °C · ${gpu.power_w.toFixed(0)} W</span></div></div>`).join("");
  renderRecent(data.recent_requests);
}

async function refreshStatus() {
  try { renderStatus(await api("/api/v1/status")); }
  catch (_) { $("statusDot").classList.remove("live"); $("serviceText").textContent = "连接中断"; }
}

async function loadModelInfo() {
  try {
    const data = await api("/api/v1/model"), metrics = data.test_metrics.global;
    $("heroPrauc").textContent = metrics.pr_auc.toFixed(3); $("metricPrauc").textContent = metrics.pr_auc.toFixed(3);
    $("metricAccuracy").textContent = formatPercent(metrics.accuracy); $("metricRecall").textContent = formatPercent(metrics.recall);
    $("metricFlip").textContent = formatPercent(data.test_metrics.policy_flip.both_correct_rate);
    $("limitationsList").innerHTML = data.limitations.map(item => `<p>— ${item}</p>`).join("");
  } catch (error) { showToast(error.message, true); }
}

function renderHeadCard(head) {
  const rows = (head.labels || []).map(label => {
    const width = label.probability == null ? 0 : Math.max(0, Math.min(1, label.probability)) * 100;
    const value = label.probability == null ? "—" : `${(label.probability * 100).toFixed(1)}%`;
    const selected = head.action && head.action === label.id ? " selected" : "";
    return `<div class="head-row${selected}"><span>${escapeHtml(label.name)}</span><div class="head-track"><i style="width:${width}%"></i></div><b>${value}</b></div>`;
  }).join("");
  const caption = head.note || head.question || "";
  return `<section class="head-card ${escapeHtml(head.id)}"><h3>${escapeHtml(head.title)}</h3><p>${escapeHtml(caption)}</p>${rows}</section>`;
}

function renderHeads(heads) {
  const root = $("headGrid");
  if (!root) return;
  root.innerHTML = heads ? [heads.attribute, heads.decision].filter(Boolean).map(renderHeadCard).join("") : "";
}

function renderResult(result) {
  $("resultEmpty").classList.add("hidden"); const content = $("resultContent"); content.classList.remove("hidden");
  const violation = result.decision === "violation"; content.classList.toggle("violation", violation);
  $("decisionMark").textContent = violation ? "!" : "✓"; $("decisionText").textContent = violation ? "违规" : "通过";
  $("scoreText").textContent = `${(result.violation_score * 100).toFixed(1)}%`; $("scoreBar").style.width = `${result.violation_score * 100}%`;
  $("thresholdMark").style.left = `${result.threshold * 100}%`; $("thresholdLabel").textContent = `阈值 ${result.threshold.toFixed(3)}`;
  renderHeads(result.heads);
  $("totalTime").textContent = formatMs(result.total_ms); $("inferenceTime").textContent = formatMs(result.inference_ms);
  $("preprocessTime").textContent = formatMs(result.preprocessing_ms); $("tokenCount").textContent = `${result.input_tokens} tokens`;
  const experimental = result.policy_support === "experimental";
  $("supportBadge").textContent = experimental ? "实验性规则" : "训练内规则";
  $("resultNote").textContent = experimental ? "自定义规则超出当前试训验证范围，请把结果当作探索性信号。" : result.threshold_mode === "low_fpr" ? "低误报模式会提高阈值，并降低部分边界样本的召回率。" : "均衡模式使用 0.5 阈值，适合观察模型的原始区分能力。";
}

async function submit(event) {
  event.preventDefault(); const input = $("imageInput");
  if (!input.files[0]) { showToast("请先选择图片", true); return; }
  const button = $("submitButton"); button.disabled = true; button.querySelector("span").textContent = "审核中…";
  const form = new FormData(); form.append("image", input.files[0]); form.append("text", $("contentText").value);
  form.append("policy_id", $("policySelect").value); form.append("policy_text", $("policyText").value); form.append("threshold_mode", $("thresholdMode").value);
  try { const result = await api("/api/v1/moderations", { method: "POST", body: form }); renderResult(result); refreshStatus(); }
  catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; button.querySelector("span").textContent = "运行审核"; }
}

$("imageInput").addEventListener("change", event => setPreview(event.target.files[0]));
$("clearImage").addEventListener("click", clearImage);
$("policySelect").addEventListener("change", event => $("customPolicyField").classList.toggle("hidden", event.target.value !== "custom"));
$("moderationForm").addEventListener("submit", submit);
const dropzone = $("dropzone");
["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", event => { const file = event.dataTransfer.files[0]; if (file) { const transfer = new DataTransfer(); transfer.items.add(file); $("imageInput").files = transfer.files; setPreview(file); } });
window.addEventListener("resize", drawChart);
loadModelInfo(); refreshStatus(); setInterval(refreshStatus, 2000);

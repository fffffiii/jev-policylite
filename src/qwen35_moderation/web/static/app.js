"use strict";

const $ = (id) => document.getElementById(id);
const state = { latency: [], maxUploadBytes: 10 * 1024 * 1024, previewUrl: null, lastRequestCount: null };
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const clampPercent = (value) => finite(value) ? Math.max(0, Math.min(1, value)) * 100 : 0;

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}
function formatMs(value) {
  if (!finite(value)) return "--";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(0)} ms`;
}
function formatPercent(value, digits = 1) { return finite(value) ? `${(value * 100).toFixed(digits)}%` : "--"; }
function fixed(value, digits = 1) { return finite(value) ? value.toFixed(digits) : "--"; }
function gib(value) { return finite(value) ? `${(value / 1024).toFixed(1)} GiB` : "--"; }
function showToast(message, error = false) {
  const toast = $("toast"); toast.textContent = message; toast.className = `toast show${error ? " error" : ""}`;
  clearTimeout(showToast.timer); showToast.timer = setTimeout(() => toast.className = "toast", 3500);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `请求失败 (${response.status})`);
  }
  return response.json();
}
function clearImage(event) {
  event?.preventDefault();
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $("imageInput").value = ""; $("preview").removeAttribute("src"); $("dropzone").classList.remove("has-image");
}
function validFile(file) {
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showToast("请选择 JPEG、PNG 或 WebP 图片。", true); return false;
  }
  if (file.size > state.maxUploadBytes) {
    showToast(`图片不能超过 ${state.maxUploadBytes / 2**20} MiB。`, true); return false;
  }
  return true;
}
function setPreview(file) {
  if (!file) return;
  if (!validFile(file)) { clearImage(); return; }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $("preview").src = state.previewUrl; $("dropzone").classList.add("has-image");
}
function drawChart() {
  const canvas = $("latencyChart"), ratio = window.devicePixelRatio || 1;
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
  $("recentRows").innerHTML = records.length ? records.map(row => {
    const decision = row.decision === "pass" ? "pass" : row.decision === "violation" ? "violation" : "unknown";
    const name = { pass: "未判违规", violation: "判为违规", unknown: "未知" }[decision];
    return `<tr><td>${escapeHtml(new Date(row.timestamp).toLocaleTimeString())}</td><td>${escapeHtml(row.policy_id)}</td><td class="decision-${decision}">${name}</td><td>${formatPercent(row.score)}</td><td>${formatMs(row.latency_ms)}</td><td>${formatMs(row.inference_ms)}</td></tr>`;
  }).join("") : '<tr><td colspan="6" class="empty-row">还没有审核请求</td></tr>';
}
function renderStatus(data) {
  $("statusDot").classList.toggle("live", data.model_ready); $("serviceText").textContent = data.model_ready ? "服务在线" : "模型未就绪";
  $("heroReady").textContent = data.model_ready ? "已就绪" : "未就绪";
  const uptime = Math.floor(data.uptime_seconds || 0); $("uptime").textContent = `运行 ${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m`;
  const latency = data.latency_ms || {};
  $("heroLatency").textContent = formatMs(latency.p50);
  $("latencySummary").textContent = `P50 ${formatMs(latency.p50)} · P95 ${formatMs(latency.p95)}`;
  // 状态轮询不代表产生了一次新的推理测量。
  if (finite(latency.latest) && data.total_requests !== state.lastRequestCount) {
    state.latency.push(latency.latest); state.latency = state.latency.slice(-40); drawChart();
  }
  state.lastRequestCount = data.total_requests;
  const gpus = Array.isArray(data.gpus) ? data.gpus : [], gpu = gpus[0];
  $("gpuMemoryTitle").textContent = gpu ? `GPU ${gpu.index} 显存（监控）` : "GPU 显存";
  $("gpuMemory").textContent = gpu ? gib(gpu.memory_used_mb) : "--";
  $("gpuMemoryDetail").textContent = gpu ? `${gib(gpu.memory_used_mb)} / ${gib(gpu.memory_total_mb)}` : "未获得 NVIDIA GPU 监控数据";
  $("gpuMemoryBar").style.width = `${gpu && gpu.memory_total_mb > 0 ? clampPercent(gpu.memory_used_mb / gpu.memory_total_mb) : 0}%`;
  $("gpuUtil").textContent = gpu ? `${fixed(gpu.utilization_percent, 0)}%` : "--";
  $("gpuThermal").textContent = gpu ? `${fixed(gpu.temperature_c, 0)} °C · ${fixed(gpu.power_w, 0)} W` : "--";
  $("processMemory").textContent = gib(data.process?.rss_mb);
  $("cudaMemory").textContent = `CUDA 已分配 ${fixed(data.cuda?.allocated_mb, 0)} MiB`;
  $("requestRate").textContent = data.requests_per_minute ?? "--"; $("requestTotal").textContent = `累计 ${data.total_requests ?? 0} · 失败 ${data.failed_requests ?? 0}`;
  $("gpuList").innerHTML = gpus.length ? gpus.map(item => `<div class="gpu-item"><div class="gpu-item-head"><span>GPU ${escapeHtml(item.index)} · ${escapeHtml(String(item.name || "").replace("NVIDIA GeForce ", ""))}</span><span>${fixed(item.utilization_percent, 0)}%</span></div><div class="gpu-item-meta"><span>${gib(item.memory_used_mb)} / ${gib(item.memory_total_mb)}</span><span>${fixed(item.temperature_c, 0)} °C · ${fixed(item.power_w, 0)} W</span></div></div>`).join("") : '<p>未获得 GPU 监控数据；不代表模型一定运行在 CPU 上。</p>';
  renderRecent(Array.isArray(data.recent_requests) ? data.recent_requests : []);
}
async function refreshStatus() {
  try { renderStatus(await api("/api/v1/status")); }
  catch (_) { $("statusDot").classList.remove("live"); $("serviceText").textContent = "连接中断"; $("heroReady").textContent = "连接中断"; }
}
function renderModelInfo(data) {
  const metrics = data.test_metrics?.global || {};
  $("heroPrauc").textContent = fixed(metrics.pr_auc, 3); $("metricPrauc").textContent = fixed(metrics.pr_auc, 3);
  $("metricAccuracy").textContent = formatPercent(metrics.accuracy); $("metricRecall").textContent = formatPercent(metrics.recall);
  $("metricFlip").textContent = formatPercent(data.test_metrics?.policy_flip?.both_correct_rate);
  $("evidenceCount").textContent = finite(metrics.count) ? `${metrics.count.toLocaleString()} 条评测记录` : "未提供有效评测记录数";
  $("evidenceNote").textContent = data.test_metrics_note || "指标来自导入文件，请核对检查点、数据与阈值。";
  $("limitationsList").replaceChildren(...(Array.isArray(data.limitations) ? data.limitations : []).map(item => {
    const p = document.createElement("p"); p.textContent = String(item); return p;
  }));
  if (finite(data.max_upload_bytes) && data.max_upload_bytes > 0) state.maxUploadBytes = data.max_upload_bytes;
  $("uploadLimit").textContent = `最大 ${state.maxUploadBytes / 2**20} MiB`;
}
async function loadModelInfo() {
  try { renderModelInfo(await api("/api/v1/model")); }
  catch (error) { $("evidenceNote").textContent = "无法读取评测信息；未显示任何替代结果。"; showToast(error.message, true); }
}
function renderHeadCard(head) {
  const rows = (head.labels || []).map(label => {
    const selected = head.action && head.action === label.id ? " selected" : "";
    return `<div class="head-row${selected}"><span>${escapeHtml(label.name)}</span><div class="head-track"><i style="width:${clampPercent(label.probability)}%"></i></div><b>${formatPercent(label.probability)}</b></div>`;
  }).join("");
  return `<section class="head-card ${escapeHtml(head.id)}"><h3>${escapeHtml(head.title)}</h3><p>${escapeHtml(head.note || head.question || "")}</p>${rows}</section>`;
}
function renderHeads(heads) {
  $("headGrid").innerHTML = heads ? [heads.attribute, heads.decision].filter(Boolean).map(renderHeadCard).join("") : "";
}
function renderResult(result) {
  $("resultEmpty").classList.add("hidden"); const content = $("resultContent"); content.classList.remove("hidden");
  const violation = result.decision === "violation"; content.classList.toggle("violation", violation);
  $("decisionMark").textContent = violation ? "!" : "✓"; $("decisionText").textContent = violation ? "判为违规" : "未判违规";
  $("scoreText").textContent = formatPercent(result.violation_score); $("scoreBar").style.width = `${clampPercent(result.violation_score)}%`;
  $("thresholdMark").style.left = `${clampPercent(result.threshold)}%`; $("thresholdLabel").textContent = `阈值 ${fixed(result.threshold, 3)}`;
  renderHeads(result.heads);
  $("totalTime").textContent = formatMs(result.total_ms); $("inferenceTime").textContent = formatMs(result.inference_ms);
  $("preprocessTime").textContent = formatMs(result.preprocessing_ms); $("tokenCount").textContent = `${result.input_tokens} tokens`;
  const experimental = result.policy_support === "experimental";
  $("supportBadge").textContent = experimental ? "自定义规则 · 未验证" : "预设试验规则";
  const modeNote = experimental ? "自定义规则尚未验证，结果仅供测试。" : result.threshold_mode === "low_fpr" ? "使用温度校准分数及校准文件阈值；不保证线上误报率。" : "使用原始分数与 0.5 阈值；“未判违规”不等于内容一定安全。";
  $("resultNote").textContent = `${modeNote} 策略头建议与二元判断独立。服务处理时间不含上传和图片解码。`;
}
async function submit(event) {
  event.preventDefault(); const file = $("imageInput").files[0];
  if (!file) { showToast("请先选择图片", true); return; }
  if (!validFile(file)) { clearImage(); return; }
  const button = $("submitButton"); button.disabled = true; button.querySelector("span").textContent = "审核中…";
  const form = new FormData(); form.append("image", file); form.append("text", $("contentText").value);
  const policy = $("policySelect").value;
  form.append("policy_id", policy);
  // 切回预设规则时，不再提交之前填写的自定义规则。
  form.append("policy_text", policy === "custom" ? $("policyText").value : "");
  form.append("threshold_mode", $("thresholdMode").value);
  try { renderResult(await api("/api/v1/moderations", { method: "POST", body: form })); refreshStatus(); }
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
dropzone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];
  if (file) { const transfer = new DataTransfer(); transfer.items.add(file); $("imageInput").files = transfer.files; setPreview(file); }
});
window.addEventListener("resize", drawChart);
window.addEventListener("pagehide", () => { if (state.previewUrl) URL.revokeObjectURL(state.previewUrl); });
loadModelInfo(); refreshStatus(); setInterval(refreshStatus, 2000);

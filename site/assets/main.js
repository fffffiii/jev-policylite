"use strict";

const stages = [
  { label: "SUPERVISED ADAPTATION", title: "先用标注数据训练任务头和 LoRA。", description: "冻结视觉编码器，先预热分类头，再联合更新语言 LoRA 与有监督信号的任务头。每张原图的多规则样本放在同一数据划分中。", detail: "更新：LoRA + 任务头　｜　输入：图片、正文、规则与标签" },
  { label: "HUMAN PREFERENCES", title: "记录人工动作与需要纠正的模型动作。", description: "在相同图片、正文和政策条件下，人工认可的动作成为 chosen，被纠正的原动作成为 rejected。保留原图组标识，核对训练与验证之间没有同图泄漏；相同动作会跳过。", detail: "输出：偏好 JSONL　｜　动作：block / review / allow" },
  { label: "DISCRETE DPO", title: "重用图文特征，只更新策略头。", description: "冻结主干和 LoRA，先把特征缓存在本次进程中。当前策略头与冻结参考头共享这些特征，通过偏好概率比计算 DPO 损失；只有当前策略头接收梯度。", detail: "更新：policy_head　｜　冻结：主干、LoRA、参考头、其余任务头" },
  { label: "EVALUATE & DELIVER", title: "在独立测试集上检查策略变化。", description: "记录偏好准确率、边际、KL 与资源占用，再在独立数据上评估误阻断和复审率。工具保存 adapter、processor 与全部已有任务头；加载时仍需可用的基础模型。", detail: "输出：检查点 + dpo_metrics.json　｜　端侧：单头 Q4 原型，多头仍需适配" }
];
const snippets = [
  "# 在项目根目录创建训练环境\npython -m venv .venv\nsource .venv/bin/activate\npip install -e '.[dev,web]'\n\n# 校验自有图文清单\npython scripts/validate_data.py \\\n  --manifest data/manifest.jsonl \\\n  --image-root .",
  "# 先准备真实反馈与已训练的多头检查点\npython scripts/build_preference_pairs.py \\\n  --feedback data/review_feedback.jsonl \\\n  --output outputs/preferences/review_pairs.jsonl \\\n  --strict\n\npython scripts/train_policy_dpo.py \\\n  --checkpoint outputs/public-pilot-multihead \\\n  --preferences outputs/preferences/review_pairs.jsonl \\\n  --output-dir outputs/policy-dpo \\\n  --epochs 3 --beta 0.5",
  "# 对新检查点做独立测试\npython scripts/evaluate.py \\\n  --checkpoint outputs/policy-dpo \\\n  --manifest data/manifest.jsonl \\\n  --image-root . \\\n  --split test \\\n  --output-dir outputs/policy-dpo-eval\n\n# 人工动作标签是策略头评测的前提\npython scripts/evaluate_heads.py \\\n  --predictions outputs/policy-dpo-eval/test_predictions.jsonl \\\n  --output outputs/policy-dpo-eval/head_metrics.json"
];

function bindTabs(selector, update) {
  const tabs = Array.from(document.querySelectorAll(selector));
  const select = (index) => {
    tabs.forEach((tab, position) => {
      tab.setAttribute("aria-selected", String(position === index));
      tab.tabIndex = position === index ? 0 : -1;
    });
    update(index, tabs[index].id);
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(index));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (index + 1) % tabs.length;
      if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); select(next); tabs[next].focus(); }
    });
  });
}

bindTabs("[data-stage]", (index, tabId) => {
  const stage = stages[index];
  ["label", "title", "description", "detail"].forEach((key) => {
    document.getElementById(`stage-${key}`).textContent = stage[key];
  });
  document.getElementById("stage-panel").setAttribute("aria-labelledby", tabId);
});
bindTabs("[data-code]", (index, tabId) => {
  document.getElementById("code-content").textContent = snippets[index];
  document.getElementById("code-panel").setAttribute("aria-labelledby", tabId);
  document.getElementById("copy-status").textContent = "示例命令 · 在项目根目录执行";
  document.getElementById("copy-code").textContent = "复制";
});

document.getElementById("copy-code").addEventListener("click", async () => {
  const button = document.getElementById("copy-code");
  try {
    await navigator.clipboard.writeText(document.getElementById("code-content").textContent);
    button.textContent = "已复制";
    document.getElementById("copy-status").textContent = "命令已复制到剪贴板。";
  } catch {
    button.textContent = "手动复制";
    document.getElementById("copy-status").textContent = "当前浏览器不允许剪贴板访问，请选中代码手动复制。";
  }
});

const dialog = document.getElementById("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach((button) => {
  button.addEventListener("click", () => {
    const image = document.getElementById("dialog-image");
    image.src = button.dataset.zoom;
    image.alt = button.dataset.caption;
    document.getElementById("dialog-caption").textContent = button.dataset.caption;
    document.getElementById("dialog-download").href = button.dataset.zoom;
    dialog.showModal();
  });
});
document.getElementById("close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

// 未发布时保留可用的本页入口，不构造虚假的仓库地址。
let repositoryUrl = window.MODERATION_SITE?.repositoryUrl || "";
if (!repositoryUrl && location.hostname.endsWith(".github.io")) {
  const owner = location.hostname.slice(0, -".github.io".length);
  const project = location.pathname.split("/").filter(Boolean)[0];
  const repository = !project || project === "index.html" ? `${owner}.github.io` : project;
  repositoryUrl = `https://github.com/${owner}/${repository}`;
}
if (/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\/?$/.test(repositoryUrl)) {
  document.querySelectorAll("[data-repo-link]").forEach((link) => {
    link.href = repositoryUrl;
    link.textContent = "GitHub ↗";
  });
  document.querySelectorAll("[data-doc-path]").forEach((link) => {
    link.href = repositoryUrl.replace(/\/$/, "") + "/blob/main/" + link.dataset.docPath;
  });
}

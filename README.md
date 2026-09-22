# Jev-PolicyLite · 面向内容审核的轻量多模态决策模型

<p align="center"><strong>图文与规则联合审核 · 共享表征多头决策 · 人工反馈驱动的快速后训练</strong><br>Adapting Jev-inspired decisions to content moderation with multi-head prediction and lightweight policy-head DPO.</p>

<p align="center">
  <img src="site/assets/architecture.png" alt="Jev-PolicyLite：图像、文本与规则经共享表征后分支到违规、属性与策略头；DPO 只更新策略头。" width="100%">
</p>

<p align="center">
  <a href="site/">Project page</a> ·
  <a href="docs/TRAINING_GUIDE.md">Training guide</a> ·
  <a href="docs/HUMAN_FEEDBACK_AND_RL.md">Preference &amp; DPO</a> ·
  <a href="docs/EXPERIMENTS.md">Experiments</a>
</p>

> **面向内容审核，从违规识别到处置决策，再到人工纠偏后的策略更新。**<br>
> Image + text + policy → shared representation → violation score, visual attributes and policy action.

**Jev-PolicyLite** 将 Jev / NanoJev 的直接评分与决策接口思想用于**内容审核任务**。我们基于 `Qwen/Qwen3.5-0.8B`，实现了图文与规则联合输入、共享表征多头预测、审核反馈偏好构建和策略头离散 DPO，并配套训练、评测与本地服务工具。当前实验围绕色情与敏感内容审核展开，项目名中的 **Jev** 表达思想来源，**PolicyLite** 表达轻量审核策略建模与更新；仓库名为 `jev-policylite`。

项目保留现成小型视觉语言模型的图文能力，通过领域 LoRA 与小型决策头适配任务；对固定审核标签，一次提取共享表征，同时输出违规分数、视觉属性和处置动作。人工反馈可进一步用于离散 DPO，只更新策略头。当前定位是**小型、可按规则判断的多模态审核研究工程**；规则泛化、细粒度识别和真实反馈收益需要独立评测验证。

## 我们为审核任务做了什么？

我们的工作集中在**把直接决策模型落实为可训练、可纠偏、可评测的审核流程**。下面列出已实现的任务适配和工程工作，以及对应的验证进展。

| 本项目的工作 | 解决的审核问题 | 实现与验证进展 |
| --- | --- | --- |
| **图文与规则联合输入** | 为模型提供图片、正文和政策语境，让同一内容可以在不同审核规则下接受判断。 | 已实现输入与领域 LoRA 训练路径；未见规则泛化仍需专项评测。 |
| **共享表征上的三个审核头** | 分别回答“是否违规”“有哪些视觉属性”“如何处置”，允许裸露与医学等属性同时存在。 | 已实现违规头、属性头与 `block / review / allow` 策略头；属性可靠性和复审能力取决于对应监督数据。 |
| **审核纠偏转为动作偏好** | 将人工认可的处置与被纠正的处置组织为同一审核条件下的 chosen / rejected 对。 | 已提供反馈转偏好工具、数据校验和按原图组隔离的训练/验证划分。 |
| **只更新策略头的离散 DPO** | 在冻结表征足够表达证据时，降低反复调整审核处置策略的训练开销。 | 已跑通 RTX 3090 上的 128 对 bootstrap 实验；编码 46.23 秒后，两轮头部训练及验证共 0.19 秒。真实人工偏好收益尚待验证。 |
| **四图拼接压力测试** | 检查目标缩小和多图干扰下，模型能否识别整张拼图是否包含违规内容。 | 已提供构建与评测工具；首轮 200 张拼图准确率 89.5%，单违规格场景 84.0%，不代表逐格定位能力。 |
| **本地审核服务与端侧探索** | 支持本地调用、网页测试和资源观测，并探索普通设备离线运行。 | 已实现 FastAPI 服务与网页；已有 Linux CPU 单头 Q4 原型，多头端侧适配与手机实测仍待完成。 |

```text
图片 + 正文 + 审核规则
          ↓
Qwen3.5-0.8B + 领域 LoRA → 共享图文表征
          ├─ 违规头：风险分数
          ├─ 属性头：可同时成立的视觉属性
          └─ 策略头：拦截 / 复审 / 放行
                              ↓
                        人工审核与纠偏
                              ↓
                  动作偏好对 → 策略头 DPO → 重新评测
```

训练方法见 [训练指南](docs/TRAINING_GUIDE.md)，反馈与后训练流程见 [偏好优化指南](docs/HUMAN_FEEDBACK_AND_RL.md)，实验协议见 [评测工具](docs/EXPERIMENTS.md)。这里的快速更新指冻结特征后的策略头优化，完整耗时还包括模型加载与图文编码。

## 从 Jev / NanoJev 借鉴了什么？

本项目借鉴的是**将隐藏表征用于直接评分，以结构化决策接口组织输出**的思路。按本项目的技术设计说明，NanoJev 的评分路径为这种设计提供了参考；Jev-PolicyLite 将其应用到固定审核任务，并选择共享表征加多个任务头的实现。

| 思想与设计 | Jev-PolicyLite 的实现与边界 |
| --- | --- |
| 直接输出判断 | 从主干最后一个有效 token 的隐藏表征读取特征，经小型 `LayerNorm + Linear` 头计算分数；审核路径无需生成解释文本。 |
| 按任务定义输出 | 违规判断使用 sigmoid；可同时成立的视觉属性分别使用 sigmoid；`block / review / allow` 互斥动作使用 softmax。 |
| 共享公共输入计算 | 图片、正文与规则共同编码一次，固定任务头复用同一表征；当前不实现动态候选集合注意力或候选重排接口。 |
| 用反馈调整策略 | 本项目增加冻结特征上的策略头离散 DPO，重用编码结果以减少后续优化计算；不将这一训练器归为 Jev 的原始算法。 |

**借鉴关系：Jev-PolicyLite 是独立实现，不是 Jev 全部内部结构或 NanoJev 的完整复现，也不表示官方关联。** 接入决策头本身不会增加视觉识别能力；效果仍依赖局部视觉信息、训练标签、规则覆盖与决策目标。直接评分也不是 Jev 独有的能力，速度与效果优势应通过相同输入和数据条件下的对照实验验证。

## One representation, three moderation heads

Jev-PolicyLite adapts `Qwen/Qwen3.5-0.8B` to policy-conditioned image-and-text moderation. It separates three questions:

| Head | Question | Current output |
| --- | --- | --- |
| Violation head | Is this content a violation under the supplied policy? | Binary probability |
| Attribute head | Which visual attributes are present? | `nudity`, `sexual_act`, `suggestive`, `medical` |
| Policy head | What should the system do? | `block`, `review`, `allow` |

The model pools one shared multimodal representation and attaches small `LayerNorm + Linear` heads. The architecture lets a human feedback loop target the policy head without retraining the visual backbone for every policy adjustment.

<p align="center">
  <img src="site/assets/post-training.png" alt="后训练流程：人工偏好对、一次冻结特征抽取、冻结参考策略头、可训练策略头和离散 DPO 损失。" width="100%">
</p>

## Why post-train a moderation policy head?

An action can disagree with the active policy even when the visual evidence is represented correctly. Policy-head post-training targets that decision layer; it cannot recover visual details missing from frozen features. The project turns a human correction into an auditable preference pair:

```text
(image, text, policy, chosen action, rejected action)
```

The discrete DPO trainer freezes the Qwen backbone, LoRA, binary head and attribute head. It encodes each preference record once, caches its pooled feature, and updates only the small three-way policy head against a frozen reference head. This is a preference-optimization implementation for the existing classifier architecture, not online PPO, GRPO or a reward-model rollout pipeline.

## Measured smoke result

The first end-to-end DPO smoke run used **128 binary-bootstrap preference pairs** on one RTX 3090. These pairs verify the training path; they do **not** demonstrate a gain from real human preference data or teach the `review` class.

| Metric | Validation before | Validation after |
| --- | ---: | ---: |
| DPO loss | 0.6931 | 0.4371 |
| Preference ranking accuracy | 100.0% | 100.0% |
| Mean chosen/rejected log-probability margin | 6.7104 | 7.9800 |
| KL to frozen reference policy | 0.00000 | 0.00119 |

Feature extraction for 128 records took **46.23 s**. Two policy-head training epochs, including per-epoch validation, took **0.19 s** with a **2.00 GiB** peak CUDA allocation. The head time excludes model loading and feature extraction. Full details: [DPO smoke result](docs/DPO_SMOKE_RESULT_V1.md).

## Quick start

Requirements: Python 3.10+, PyTorch with a compatible CUDA build for GPU training, and Transformers 5.x.

```bash
git clone <YOUR_REPOSITORY_URL> jev-policylite
cd jev-policylite

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev,web]'
```

Prepare a UTF-8 JSONL manifest, then validate group isolation before training:

```bash
python scripts/validate_data.py \
  --manifest data/manifest.jsonl \
  --image-root .
```

See [data/README.md](data/README.md) for the manifest schema and [Training guide](docs/TRAINING_GUIDE.md) for supervised adaptation, calibration and evaluation.

## Preference optimization

Start from a checkpoint with an already trained `policy_head.pt`, and use real review feedback whenever possible:

```bash
python scripts/build_preference_pairs.py \
  --feedback data/review_feedback.jsonl \
  --output outputs/preferences/review_pairs.jsonl \
  --strict

CUDA_VISIBLE_DEVICES=0 python scripts/train_policy_dpo.py \
  --checkpoint outputs/your-multihead-checkpoint \
  --preferences outputs/preferences/review_pairs.jsonl \
  --output-dir outputs/your-policy-dpo \
  --epochs 3 \
  --batch-size 4 \
  --head-batch-size 64 \
  --learning-rate 5e-4 \
  --beta 0.5
```

The output is a standalone checkpoint containing the source adapter, processor, unchanged binary and attribute heads, an updated `policy_head.pt`, and `dpo_metrics.json`. The trainer rejects output-directory overwrites and performs a group-isolated train/validation split.

The helper below only tests the data and training path. It maps old binary labels to `block` and `allow`; it does not create new supervision.

```bash
python scripts/build_bootstrap_preferences.py \
  --manifest data/public-pilot/manifest.jsonl \
  --split train \
  --max-records 128 \
  --output outputs/preferences/bootstrap.jsonl
```

Read the full [human feedback and discrete DPO guide](docs/HUMAN_FEEDBACK_AND_RL.md) before using either command.

## Evaluation and stress tests

The repository includes a 2×2 mosaic stress test that composes four original images and derives the label by an explicit OR rule. The first 200-mosaic result reached 89.5% accuracy, 95.1% precision, 90.7% recall and 14.0% FPR. The one-violation scenario was the hardest at 84.0% accuracy; a mosaic test is not evidence of per-tile detection.

```bash
python scripts/build_mosaic_manifest.py \
  --manifest data/manifest.jsonl \
  --image-root . \
  --split test \
  --output-dir outputs/mosaic-2x2 \
  --output-manifest outputs/mosaic-2x2/manifest.jsonl

python scripts/evaluate.py \
  --checkpoint outputs/your-multihead-checkpoint \
  --manifest outputs/mosaic-2x2/manifest.jsonl \
  --image-root outputs/mosaic-2x2 \
  --split test \
  --output-dir outputs/mosaic-2x2/evaluation
```

Read [experiment tools](docs/EXPERIMENTS.md) and the [first mosaic result](docs/MOSAIC_2X2_RESULT_V1.md) for the exact protocol and limitations.

## Local service and edge research

The FastAPI service provides a local webpage, moderation endpoint, status telemetry and OpenAPI docs from one model process:

```bash
MODEL_CHECKPOINT=outputs/your-checkpoint \
CALIBRATION_FILE=outputs/your-checkpoint/calibration.json \
python -m uvicorn qwen35_moderation.web.app:app \
  --host 0.0.0.0 --port 8089 --workers 1
```

For offline research, the project has a Linux x86_64 CPU Q4 prototype for the current binary head. Its current measured model directory is about 479 MB and peak RSS is about 859 MiB with four CPU threads. The policy and attribute heads require separate edge-runtime support; mobile, Windows, Android and iOS measurements are pending. See [edge status](docs/EDGE_STATUS.md).

## Published pilot adapter

The first public pilot checkpoint is included in [`models/pilot-multihead-v0.1`](models/pilot-multihead-v0.1). It contains the LoRA adapter, binary / attribute / policy heads, calibration file and metadata; Git LFS stores the adapter weights. It does **not** contain the 1.7 GB merged base model, training images, manifests, predictions or reviewer feedback.

Load it only with the stated `Qwen/Qwen3.5-0.8B` base model and this repository's code. The adapter was trained with source-level proxy labels for a public-pilot development check. Its validation metrics and DPO smoke result are not production safety, policy-generalization or fine-grained attribute claims; see the [model card](models/pilot-multihead-v0.1/README.md) before use.

## Scope and limitations

- Jev-inspired heads do not establish better visual recognition or faster end-to-end inference by themselves. Compare against a vision encoder with a classifier and the same VLM with single-step Yes/No scoring under matched conditions.
- Policy conditioning requires policy-dependent labels. Evaluate removed, replaced and unseen policy text to check whether the model uses the rules; those ablations are evaluation goals, not results established by the current smoke run.
- Sigmoid and softmax outputs require calibration checks before being interpreted as confidence. Report recall at a stated FPR on an independent test set, alongside results by content type.
- Personalized recommendation, dynamic candidate ranking and video temporal modeling are outside the current implementation. Content understanding alone does not establish user-preference prediction.
- The public-pilot data and binary-bootstrap DPO run are development checks, not production safety claims.
- `review` requires dedicated human decision labels and preference comparisons. It is not validated by mapping binary labels to block/allow.
- The public-pilot attribute labels are source-level proxies. They do not validate medical capability or fine-grained attribute reliability.
- Small validation sets cannot establish a real-world low false-positive rate. Use held-out, group-isolated evaluation and report errors by policy and content type.
- Do not commit real moderation images, annotations, reviewer identity, credentials, model weights or private run logs. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Project layout

```text
configs/                    reproducible training settings
data/                       schema examples only; local data is ignored
scripts/                    data, training, DPO, calibration and evaluation tools
src/qwen35_moderation/      model, heads, service and browser UI
site/                       static GitHub Pages project page
docs/                       experimental protocols and public documentation
tests/                      unit tests for data, heads, feedback and DPO loss
```

## Contributing and release

Run `python -m pytest -q` before a pull request. Release review must confirm that sensitive data, weights, internal addresses and private logs are ignored. See [CONTRIBUTING.md](CONTRIBUTING.md), [release checklist](RELEASE_CHECKLIST.md), and [GitHub Pages deployment](docs/GITHUB_PAGES.md).

## License

The project source code is released under the [MIT License](LICENSE). The included pilot adapter depends on `Qwen/Qwen3.5-0.8B`; base-model weights, datasets and third-party dependencies remain subject to their own licenses and terms.

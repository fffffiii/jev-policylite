# Jev-PolicyLite

**结合图片、正文和审核规则，预测违规分数与处置动作**

[中文](README.md) | [English](README.en.md)

[模型](models/pilot-multihead-v0.1) · [四图检测头](models/pilot-multi-photo-v0.1) · [训练指南](docs/TRAINING_GUIDE.md) · [实验记录](docs/EXPERIMENTS.md) · [项目页面源码](site/) · [MIT](LICENSE)

Jev-PolicyLite 基于 Qwen3.5-0.8B，将图片、正文和审核规则作为输入，直接预测违规分数、视觉属性和处置动作。项目参考 Jev / NanoJev 直接从隐藏表征评分的做法，提供多头训练、人工复核记录转换和策略头偏好优化工具。目前的试验主要使用性暗示、裸露与色情分级数据，不代表对所有敏感内容都有效。

项目测试小模型能否结合图文与规则判断内容，也测量调整处置动作所需的训练开销。三个审核头共享图文表征；偏好优化时冻结主干并缓存特征，只更新预测拦截、复审或放行的策略头。真实人工反馈是否带来收益，仍需单独验证。

<p align="center">
  <img src="site/assets/architecture-paper.png" alt="Jev-PolicyLite 架构：图文与规则编码为共享表征，连接违规、属性与策略三个审核头。" width="100%">
</p>

## 项目内容

- **多头审核模型**：保留 Qwen3.5 的图文编码能力，通过 LoRA 适配审核任务，分别训练违规、属性和处置策略。
- **策略头后训练**：将人工纠偏整理成动作偏好对，在冻结特征上执行离散 DPO，减少重复编码。
- **训练与评测工具**：提供数据校验、按原图分组划分、阈值校准、多头评测，以及四张照片一次输入的逐图定位与速度对照。
- **模型与本地服务**：提供试验版 LoRA 适配器、三个审核头，以及带资源监控的 FastAPI 网页服务。

仓库记录了监督训练、DPO 和本地部署的流程试验。以下结果来自已有实验记录，不是完整的生产环境评估。

## 方法

### 图文与规则联合审核

模型取最后一个有效 token 的隐藏表征，接三个 `LayerNorm + Linear` 头：

| 审核头 | 输出 | 用途 |
| --- | --- | --- |
| 违规头 | sigmoid 分数 | 判断内容在输入规则下是否违规 |
| 属性头 | 多个 sigmoid 分数 | 预测 `nudity`、`sexual_act`、`suggestive`、`medical` 等可共存属性 |
| 策略头 | 三分类 softmax | 预测 `block`、`review`、`allow` |

监督训练先预热任务头，再训练语言 LoRA 与任务头，视觉编码器保持冻结。属性与处置分开建模，例如“存在裸露”和“医学语境”可以同时成立，最终处置由策略头学习。

Jev / NanoJev 提供了直接评分与决策接口的设计参考。本项目针对固定审核标签采用共享表征多头结构，并增加策略头 DPO；没有实现动态候选集合注意力，也不构成 Jev 的完整复现。

### 策略头偏好后训练

人工复核记录保留同一图片、正文和规则下的两个动作：认可的动作 `chosen` 与被纠正的动作 `rejected`。

训练时冻结主干、LoRA、违规头和属性头，一次性提取图文特征。当前策略头与冻结参考头读取同一份特征，通过离散 DPO 更新动作概率。参考策略只需复制一个小策略头。

<p align="center">
  <img src="site/assets/post-training-paper.png" alt="后训练流程：构建动作偏好、缓存冻结特征、通过离散 DPO 更新策略头。" width="100%">
</p>

这里的计算节省来自特征复用。它适用于已有表征能够区分样本、但处置需要调整的情况；冻结特征丢失的视觉细节，无法靠更新策略头补回。当前实现是离散动作上的偏好优化，不包含在线 rollout、PPO 或 GRPO。

## 实验结果

### RTX 3090 上的 DPO 流程验证

使用二元标签自动构造 128 对 block/allow 偏好，按原图分组划分为 103 对训练、25 对验证，训练 2 轮。

| 验证指标 | 训练前 | 训练后 |
| --- | ---: | ---: |
| DPO loss | 0.6931 | 0.4371 |
| 偏好排序准确率 | 100.0% | 100.0% |
| chosen/rejected 平均对数概率差 | 6.7104 | 7.9800 |
| 相对参考策略的 KL | 0 | 0.00119 |

128 条图文特征抽取耗时 **46.23 秒**；两轮策略头训练及验证耗时 **0.19 秒**；CUDA 峰值分配量 **2.00 GiB**。计时不包含模型加载与保存，0.19 秒也不包含特征抽取；训练脚本的这段计时还包含训练结束后的最终评估。

这批偏好复述已有二元标签，验证集训练前已经全部排序正确。结果证明流程可运行，尚不能说明真实人工反馈带来的收益，也没有验证复审动作。完整设置见 [DPO 实验记录](docs/DPO_SMOKE_RESULT_V1.md)。

### 四图拼接与端侧试验

| 试验 | 已测结果 | 说明 |
| --- | --- | --- |
| 2×2 拼图，200 张 | 准确率 89.5%，召回率 90.7%，误报率 14.0% | 按“任一格违规则整图违规”评测，不评估逐格定位 |
| 仅一格违规的拼图 | 准确率 84.0% | 用于检查小目标与多图干扰 |
| Linux x86_64，单头 Q4，CPU 4 线程 | 模型目录约 479 MB，峰值 RSS 约 859 MiB | 8 条抽样决策一致，未完成全量回归 |

详细记录：[四图实验](docs/MOSAIC_2X2_RESULT_V1.md)、[端侧进展](docs/EDGE_STATUS.md)。手机端、多头量化和 Windows 端尚未完成实测。

### 四张照片一次输入的逐图判断

在冻结主干与 LoRA 后，项目训练了四个位置检测头：一次输入四张原图，每个头输出对应图片的违规分数。局部读出版本从每张图视觉片段末端取表征，四个头合计 12,292 个参数。下表在 RTX 3090 上用同一批 **200 组四图、800 个逐图标签**测得，平均耗时包含读图、预处理与模型前向，不含模型加载和网络传输。

<!-- BEGIN MULTI_PHOTO_TABLE -->

| 方法 | 逐图准确率 | F1 | 召回率 | 四张全对 | 均值/组 | P95/组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 逐张串行 | 90.12% | 88.67% | 88.29% | 65.0% | 537.47 ms | 544.34 ms |
| 单图 batch=4 | 89.75% | 88.22% | 87.71% | 64.0% | 163.79 ms | 169.85 ms |
| 四图，共享末 token | 84.88% | 82.18% | 79.71% | 55.0% | 153.08 ms | 157.76 ms |
| 四图，局部位置头 | 87.50% | 85.34% | 83.14% | 60.0% | 154.64 ms | 157.47 ms |
| 四图，共享局部头 | 87.12% | 84.70% | 81.43% | 58.5% | 153.97 ms | 157.48 ms |
| 实时 2×2 拼图 | 80.75% | 78.90% | 82.29% | 50.0% | 182.59 ms | 195.44 ms |

<!-- END MULTI_PHOTO_TABLE -->

一次输入四图比逐张串行快约 3.47 倍，但比 batch=4 只快约 5.6%，且逐图准确率低 2.25 个百分点。当前不能把它当作准确率不变的加速方案；需要更稳妥的逐图结果时，优先批量处理单图。标签来自源图，不是人工重审四图组合。完整设置、P50/P95 与其他读出方式见 [四图定位与耗时报告](docs/MULTI_PHOTO_RESULT_V1.md)。

核查确认，这 800 个位置标签来自 **426 张不同原图**；原图不跨训练、验证和测试划分。局部读出仍能受到前面图片的影响。新生成器修复了跨划分组合 ID 重复的问题；旧实验按目录读取，未发现因此造成的数据泄漏。

![四图方案的速度与准确率](docs/figures/multi-photo-tradeoff.png)

### 与公开鉴黄系统对比

2026-09-26，在同一张 RTX 3090 上顺序测试三种公开系统和 Jev 的两个分辨率配置。测试为 **600 张原图 × 两条规则 = 1,200 条判断**；阈值只在另外 300 张原图组成的校准集上选择，各政策分别最大化 F1。“准确率 @0.5”用于查看校准前的表现，其余质量指标使用校准阈值。

<!-- BEGIN NSFW_QUALITY_TABLE -->

| 模型 | 准确率 @0.5 | 校准后准确率 | F1 | 召回率 | 误报率 | 准确率 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Falconsai ViT | 80.33% | 83.33% | 86.60% | 86.13% | 21.33% | 81.17%–85.42% |
| Giacomo ViT (5-class) | 74.92% | 76.42% | 82.46% | 88.67% | 44.00% | 74.00%–79.00% |
| NudeNet 320n | 75.83% | 80.17% | 85.58% | 94.13% | 43.11% | 78.00%–82.17% |
| Jev-PolicyLite 50k | 90.75% | 91.00% | 92.92% | 94.53% | 14.89% | 89.25%–92.58% |
| Jev-PolicyLite 201k | 94.08% | 93.50% | 94.72% | 93.33% | 6.22% | 91.83%–95.00% |

<!-- END NSFW_QUALITY_TABLE -->

速度使用相同的 120 张原图，先完整预热全部输入，再交错测试各模型，每种 batch 重复三轮。包含读图、预处理、前向和全部输出头或检测框后处理，不包含模型加载、网络和排队。B=4 一栏是**四张图整批完成**的时间；内存另用逐模型独立进程测量。

<!-- BEGIN NSFW_SPEED_TABLE -->

| 模型 | B=1 均值 / P95 (ms) | B=4 均值 / P95 (ms) | B=4 图片/秒 | 峰值 RSS (MiB) | Torch 分配显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Falconsai ViT | 21.50 / 24.36 | 54.21 / 57.46 | 73.8 | 1131 | 539 |
| Giacomo ViT (5-class) | 20.58 / 22.69 | 48.17 / 51.46 | 83.0 | 1144 | 539 |
| NudeNet 320n | 27.05 / 33.03 | 88.80 / 105.77 | 45.0 | 1170 | N/A (ONNX) |
| Jev-PolicyLite 50k | 164.96 / 167.33 | 198.02 / 201.80 | 20.2 | 2703 | 1749 |
| Jev-PolicyLite 201k | 169.95 / 172.36 | 211.72 / 225.72 | 18.9 | 2705 | 2012 |

<!-- END NSFW_SPEED_TABLE -->

50k / 201k 表示图像面积上限 50,176 / 200,704 像素，保持长宽比。两个 ViT 为 224×224，NudeNet 为 320×320。RSS 是 CPU 进程峰值；Torch 分配显存不等于完整进程显存，ONNX 不适用该计数器。

在这份同来源训练的双政策测试中，Jev 的识别质量较高，专用视觉系统更快。201k 版本校准后准确率 **93.50%**，batch=4 平均 **211.72 ms**；Falconsai 分别为 **83.33% / 54.21 ms**。50k 相比 201k 只减少约 **6.5%** 的整批耗时，准确率下降 **2.5 个百分点**，单靠降低分辨率的收益有限。当前证据支持图文与规则联合判断、轻量策略头后训练这两个方向，不能宣称其速度超过专用视觉鉴黄模型。

![鉴黄系统的速度与质量对比](docs/figures/nsfw-speed-quality.png)

| 系统 | 分类粒度 | 区域框 | 规则如何改变判断 |
| --- | --- | --- | --- |
| [Falconsai](https://huggingface.co/Falconsai/nsfw_image_detection) | normal / nsfw，2 类 | 无 | 程序调整阈值 |
| [Giacomo ViT](https://huggingface.co/giacomoarienti/nsfw-classifier) | drawings / hentai / neutral / porn / sexy，5 类 | 无 | 程序选择类别及阈值 |
| [NudeNet 320n](https://github.com/notAI-tech/NudeNet) | 18 种人体或覆盖状态标签，包含脸、脚等非违规类别 | 有 | 程序筛选框类别及阈值 |
| Jev-PolicyLite | 二元违规、4 个属性分数、3 个处置分数 | 无 | 将自然语言规则一起编码；目前只验证两条已见规则 |

标签数量不能当成细分类准确率。我们的属性头只用等级映射的代理标签验证了裸露、性行为和性暗示，医学没有正例，review 没有独立标注。外部五分类和检测框也缺少对应真值，不能报告五分类准确率或区域 mAP。Jev 在同来源训练划分上训练过，外部模型没有重新训练，因此本表是现成系统的任务适配对比，不能据此推断架构优劣。

在这些代理标签上，高分辨率 Jev 的裸露、性行为、性暗示 F1 分别为 **93.55% / 89.90% / 82.74%**。NudeNet 的性暗示政策校准阈值退化为 0，全部判违规；合并后的高召回掩盖了这一点。下图按政策与来源等级展示每种系统的正确率。

![不同政策和来源等级的错误分布](docs/figures/nsfw-source-levels.png)

本机 Qwen 运行仍使用 `causal_conv1d` 的 PyTorch 参考实现，未测优化内核的速度。初轮共享主机上的其他训练结束后，已对全部模型重新计时；以上也不是手机或 CPU 的速度。按政策拆分的召回、误报、AP、低误报工作点、属性诊断及复现步骤见 [完整对比与核查报告](docs/NSFW_COMPARISON_V1.md)，原始预测与计时见 [结果文件](docs/results/nsfw-comparison-v1/)。数据图提供 [PNG / SVG / PDF](docs/figures/)，可由 [绘图脚本](scripts/report_nsfw_benchmark.py) 重建。

## 快速开始

需要 Python 3.10+、Git LFS、Transformers 5.x。GPU 训练需要匹配 CUDA 的 PyTorch；以下命令使用 Bash。

```bash
git lfs install
git clone https://github.com/fffffiii/jev-policylite.git
cd jev-policylite
git lfs pull

python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,web]'
```

### 加载试验版模型

[试验模型目录](models/pilot-multihead-v0.1) 包含约 43 MB 的 LoRA 权重、三个审核头、校准文件和元数据。Qwen 基座需另行下载，模型包也未包含 processor；首次使用前，将基座 processor 保存到检查点目录：

```bash
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('models/pilot-multihead-v0.1/processor')"
```

首次加载基座需要联网或已有本地缓存。准备齐基座、适配器和 processor 后才具备离线加载条件。

```bash
python scripts/predict.py \
  --checkpoint models/pilot-multihead-v0.1 \
  --image path/to/image.jpg \
  --text "image caption" \
  --policy-id strict-v1 \
  --policy-file policies/strict.txt
```

此命令输出二元违规结果。该脚本目前不返回属性和策略头结果；三头输出与服务接口见 [训练指南](docs/TRAINING_GUIDE.md) 和 [模型说明](models/pilot-multihead-v0.1/README.md)。

### 监督训练

按 [数据格式](data/README.md) 准备自己的 JSONL 清单，在 `configs/train.yaml` 中设置数据路径与训练参数：

```bash
python scripts/validate_data.py --manifest data/manifest.jsonl --image-root .
python scripts/train.py --config configs/train.yaml
```

多头配置参考 `configs/public-pilot-multihead.yaml`。属性头和策略头需要对应标签；仅将二元标签映射成 block/allow 不会教会模型复审。完整步骤见 [训练指南](docs/TRAINING_GUIDE.md)。

### 用人工反馈更新策略

完成上述模型准备后，将复核记录写入 `data/review_feedback.jsonl`：

```bash
python scripts/build_preference_pairs.py \
  --feedback data/review_feedback.jsonl \
  --output outputs/preferences/review_pairs.jsonl \
  --strict

python scripts/train_policy_dpo.py \
  --checkpoint models/pilot-multihead-v0.1 \
  --preferences outputs/preferences/review_pairs.jsonl \
  --output-dir outputs/policy-dpo \
  --epochs 3 --beta 0.5
```

复核记录应提供原图组 `group_id`；同图和近重复图必须使用同组。缺少该字段时，转换器仅按图片路径分组，不能识别近重复图。训练器按组隔离训练与验证数据，拒绝覆盖已有输出目录。输出包含适配器、processor、各任务头和 `dpo_metrics.json`，加载时仍需要基座模型。偏好格式与参数见 [后训练指南](docs/HUMAN_FEEDBACK_AND_RL.md)。

### 本地服务

```bash
MODEL_CHECKPOINT=models/pilot-multihead-v0.1 \
CALIBRATION_FILE=models/pilot-multihead-v0.1/calibration.json \
python -m uvicorn qwen35_moderation.web.app:app \
  --host 127.0.0.1 --port 8089 --workers 1
```

浏览器访问 `http://localhost:8089`，可提交图片与正文、查看审核结果和运行状态。上方主结论来自二元违规头；属性和策略头在独立卡片显示，策略建议不等于二元阈值判断。

离线评测文件是可选的：通过 `TEST_METRICS_FILE` 指定与检查点对应的 `test_metrics.json`；未提供时页面显示“未提供”，不会填入其他试验的分数。上传图片会在预处理期间写入临时目录并在预处理后删除，服务不建立图片或正文历史库。服务没有登录鉴权，默认仅监听本机；局域网部署前请配置访问控制。

## 已知限制

- 试训属性标签来自内容等级映射，缺少充分的医学语境和真实复审标注；目前结果不能证明这些细分类别可靠。
- 输入支持审核规则，但还需要去除规则、替换规则和未见规则测试，确认模型是否真正使用政策语义。
- 小样本实验不能建立低误报保证。正式评估应在独立测试集上报告固定误报率下的召回，并按内容类型分组。
- 已完成与三种公开视觉鉴黄系统的任务对照；同基座 Yes/No 评分和统一训练数据的消融仍未完成，不能把质量差异单独归因于决策头。

## 文档与贡献

| 内容 | 入口 |
| --- | --- |
| 数据格式与标注 | [data/README.md](data/README.md) |
| 监督训练、校准与评测 | [训练指南](docs/TRAINING_GUIDE.md) |
| 人工反馈与离散 DPO | [后训练指南](docs/HUMAN_FEEDBACK_AND_RL.md) |
| 多头与四图压力测试 | [实验工具](docs/EXPERIMENTS.md) |
| 静态项目页部署 | [GitHub Pages](docs/GITHUB_PAGES.md) |

欢迎提交代码、复现结果和错误分析。提交前运行 `python -m pytest -q`；审核相关改动请说明使用的政策、标签与数据划分。详细要求见 [贡献说明](CONTRIBUTING.md)。真实审核图片、人工反馈和私有日志不随仓库发布。

## 许可与致谢

项目代码采用 [MIT](LICENSE) 协议。基座模型、数据和第三方依赖遵循各自许可。

感谢 Qwen 提供图文基座，以及 Jev / NanoJev 的直接决策设计思路。Jev-PolicyLite 为独立项目。

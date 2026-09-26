# 原始对照结果

本目录不包含图片或第三方模型权重。

- `audit.json`：数据划分、原图计数、组合标签及发布检查点散列。
- 每个模型的 `.json`：校准阈值、质量指标、逐批计时和独立进程内存。`quality_environment` 保留质量评测环境，`isolated_memory_environment` 保留内存测量环境。
- `paired_timing_protocol.json`：最终交错计时的预热范围、完整随机调度与设备快照。
- `before-full-warmup/`：只预热首批的上一轮原始记录，包括首见尺寸的长调用；不作为最终速度表。
- `*-predictions.jsonl`：校准和测试各条判断的原图组、政策、来源等级、标签及分数。Jev 额外保存属性/动作分数；NudeNet 保存原生检测标签和框。
- `summary.json` 与 `*-table.md`：由绘图脚本生成，不手工改数。

Jev 的属性数组顺序为 `nudity, sexual_act, suggestive, medical`，动作数组顺序为 `block, review, allow`。源图组 ID 是来源像素散列的前缀，同一原图在两个政策下出现。

两种 ViT 使用 [下载脚本](../../../scripts/download_nsfw_baselines.py) 固定的 revision。NudeNet 版本及 ONNX 文件散列记录在 `nudenet.json`。评测协议、标签映射和限制见 [完整报告](../../NSFW_COMPARISON_V1.md)。

在仓库根目录运行 `python scripts/report_nsfw_benchmark.py` 可以校验模型间样本集合与计时轮次，复算延迟汇总，并重建图表和中英文 README 的生成区块。

# 实验工具

## 2×2 四图拼接压力测试

`build_mosaic_manifest.py` 从同一个 validation、calibration 或 test 划分构造四图拼接样本。每张拼图包含 4 个互不重复的原图组，使用等比缩放与留边，不裁剪图片主体。

标签使用明确的 OR 规则：同一审核规则下，只要任意一个格子违规，整张拼图就标为违规。工具覆盖四种场景：`all_safe`、`one_violation`、`two_violations`、`all_violation`。

```bash
python scripts/build_mosaic_manifest.py \
  --manifest data/manifest.jsonl \
  --image-root . \
  --split test \
  --output-dir outputs/mosaic-2x2 \
  --output-manifest outputs/mosaic-2x2/manifest.jsonl \
  --cases-per-scenario 20 \
  --tile-size 224

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
  --checkpoint outputs/qwen35-08b-moderation-multihead \
  --manifest outputs/mosaic-2x2/manifest.jsonl \
  --image-root outputs/mosaic-2x2 \
  --split test \
  --calibration outputs/qwen35-08b-moderation-multihead/calibration.json \
  --output-dir outputs/mosaic-2x2/evaluation
```

工具会生成 `layouts.jsonl`，记录每张拼图的四个源 group、位置、标签和场景，便于定位漏检。`summary.json` 固化随机种子、图块尺寸和派生规则。

这是小目标、遮挡和多目标干扰压力测试，不等于“模型可以逐格审核”。若业务需要逐图结论，应先拆图再批量调用模型，或单独训练检测/区域模型。

公开试训的首轮结果见 [MOSAIC_2X2_RESULT_V1.md](MOSAIC_2X2_RESULT_V1.md)。

## 多头独立指标

先使用 `evaluate.py` 生成含 head 概率的预测文件，再分别汇总属性头和策略头：

```bash
python scripts/evaluate_heads.py \
  --predictions outputs/evaluation/test_predictions.jsonl \
  --output outputs/evaluation/head_metrics.json
```

默认只承认人工属性标签。公开试训数据中的 `source_L1` 到 `source_L4` 是代理等级；若仅作开发诊断，显式加 `--allow-source-level-proxy`。决策头评估需要预测文件中的人工 `decision_label`，没有真实 `review` 标注时，不能声称三分类已经验证。

对于四图拼接，还可以基于 `layouts.jsonl` 将每个 tile 的属性按 OR 规则聚合，单独观察小目标下的属性头：

```bash
python scripts/evaluate_mosaic_heads.py \
  --predictions outputs/mosaic-2x2/evaluation/test_predictions.jsonl \
  --layouts outputs/mosaic-2x2/layouts.jsonl \
  --allow-source-level-proxy \
  --output outputs/mosaic-2x2/evaluation/mosaic_head_metrics.json
```

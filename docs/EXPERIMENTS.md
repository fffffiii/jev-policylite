# 实验工具

## 2×2 四图拼接压力测试

`build_mosaic_manifest.py` 从指定的 train、validation、calibration 或 test 划分构造四图拼接样本。每张拼图包含 4 个互不重复的原图组，使用等比缩放与留边，不裁剪图片主体。

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

这是目标缩小和多图干扰压力测试；脚本没有额外模拟遮挡，不等于“模型可以逐格审核”。若业务需要逐图结论，应先拆图再批量调用模型，或单独训练检测/区域模型。

公开试训的首轮结果见 [MOSAIC_2X2_RESULT_V1.md](MOSAIC_2X2_RESULT_V1.md)。

## 四张照片一次输入：定位头与速度对照

新增的四位置输出预测左上、右上、左下、右下四张图的违规分数；只训练小检测头，不重新训练 Qwen 主干或 LoRA。`native` 将四张原图作为四个 image 项放进一次 prefill，并让四个头读取共同的末 token 表征；`native_local` 也只做一次 prefill，但让每个头读取对应图片视觉片段末端的表征；`native_shared` 复用相同的局部表征，用一个共享头分别预测四张图；`mosaic` 则先拼成 2×2 再送入一次 prefill。各模式的头权重不可互换。逐张串行和单图 batch=4 使用已有二元头，作为准确率与耗时基线。

下面的实验需要公开试训清单、原图和完整检查点；这些数据不包含在仓库里。每张拼图的四个位置按左上、右上、左下、右下排序。训练、验证、测试各自只取对应的源划分，防止原图跨划分泄漏。

```bash
python scripts/build_mosaic_manifest.py --manifest data/public-pilot/manifest.jsonl --image-root . --split train --output-dir outputs/multi-photo-data-v1/train --output-manifest outputs/multi-photo-data-v1/train/manifest.jsonl --cases-per-scenario 60 --seed 20260923
python scripts/build_mosaic_manifest.py --manifest data/public-pilot/manifest.jsonl --image-root . --split validation --output-dir outputs/multi-photo-data-v1/validation --output-manifest outputs/multi-photo-data-v1/validation/manifest.jsonl --cases-per-scenario 15 --seed 20260924
python scripts/build_mosaic_manifest.py --manifest data/public-pilot/manifest.jsonl --image-root . --split test --output-dir outputs/multi-photo-data-v1/test --output-manifest outputs/multi-photo-data-v1/test/manifest.jsonl --cases-per-scenario 25 --seed 20260922

python scripts/experiment_multi_photo.py --stage encode --mode native --checkpoint outputs/public-pilot-multihead --dataset-root outputs/multi-photo-data-v1 --source-image-root . --output-root outputs/multi-photo-v1 --device cuda:1
python scripts/experiment_multi_photo.py --stage encode --mode native_local --checkpoint outputs/public-pilot-multihead --dataset-root outputs/multi-photo-data-v1 --source-image-root . --output-root outputs/multi-photo-v1 --device cuda:1
python scripts/experiment_multi_photo.py --stage encode --mode mosaic --checkpoint outputs/public-pilot-multihead --dataset-root outputs/multi-photo-data-v1 --source-image-root . --output-root outputs/multi-photo-v1 --device cuda:1
python scripts/experiment_multi_photo.py --stage fit --mode native --output-root outputs/multi-photo-v1 --device cpu
python scripts/experiment_multi_photo.py --stage fit --mode native_local --output-root outputs/multi-photo-v1 --device cpu
python scripts/experiment_multi_photo.py --stage fit --mode native_shared --output-root outputs/multi-photo-v1 --device cpu
python scripts/experiment_multi_photo.py --stage fit --mode mosaic --output-root outputs/multi-photo-v1 --device cpu
python scripts/experiment_multi_photo.py --stage benchmark --checkpoint outputs/public-pilot-multihead --dataset-root outputs/multi-photo-data-v1 --source-image-root . --output-root outputs/multi-photo-v1 --device cuda:1 --benchmark-cases 200
```

`encode` 逐组缓存冻结特征；`fit` 在缓存上训练检测头并输出逐位置指标；`native_shared` 的 `fit` 直接复用 `native_local` 特征，无需再次编码。`benchmark` 在两项政策与四种违规数量场景之间均衡抽取同一批样本，记录逐张串行、单图 batch=4、三种原生四图读出和实时拼图的耗时。计时预热各输入形态，包含读图、预处理和前向，不包含模型加载与网络传输。实时拼图还计入拼接和 JPEG 编码。原生四图与逐张、batch=4 基线都使用每张图 50,176 像素上限；拼图整体使用 200,704 像素上限。

训练好的位置头和试验结果见 [模型目录](../models/pilot-multi-photo-v0.1) 与 [结果记录](MULTI_PHOTO_RESULT_V1.md)。单次四图推理可用 `scripts/predict_multi_photo.py`，四张输入的顺序就是四个输出的位置；其输出是未校准的 sigmoid 分数。

## 多头独立指标

先使用 `evaluate.py` 生成含 head 概率的预测文件，再分别汇总属性头和策略头：

```bash
python scripts/evaluate_heads.py \
  --predictions outputs/evaluation/test_predictions.jsonl \
  --output outputs/evaluation/head_metrics.json
```

默认只承认人工属性标签。公开试训数据中的 `source_L1` 到 `source_L4` 是代理等级；若仅作开发诊断，显式加 `--allow-source-level-proxy`。策略头评估需要预测文件中的人工 `decision_label`，没有真实 `review` 标注时，不能声称三分类已经验证。

对于四图拼接，还可以基于 `layouts.jsonl` 将每个 tile 的属性按 OR 规则聚合，单独观察小目标下的属性头：

```bash
python scripts/evaluate_mosaic_heads.py \
  --predictions outputs/mosaic-2x2/evaluation/test_predictions.jsonl \
  --layouts outputs/mosaic-2x2/layouts.jsonl \
  --allow-source-level-proxy \
  --output outputs/mosaic-2x2/evaluation/mosaic_head_metrics.json
```

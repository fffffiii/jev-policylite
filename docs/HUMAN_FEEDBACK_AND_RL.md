# 人工反馈与离散 DPO

工程支持两条策略头训练路径：清单中的 `decision` 字段可直接监督训练 `block / review / allow`；人工复核产生的动作偏好可继续对同一策略头执行离散 DPO。

## 1. 构建人工偏好对

```bash
python scripts/build_preference_pairs.py \
  --feedback data/review_feedback.jsonl \
  --output outputs/preferences/review_pairs.jsonl \
  --strict
```

输入格式见 [review_feedback.example.jsonl](../data/review_feedback.example.jsonl)。每行包含原图、审核规则、人工动作和原模型动作。人工动作成为 `chosen`，原模型动作成为 `rejected`；二者相同时不会产生偏好样本。

## 2. 训练策略头

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_policy_dpo.py \
  --checkpoint outputs/your-multihead-checkpoint \
  --preferences outputs/preferences/review_pairs.jsonl \
  --output-dir outputs/public-pilot-policy-dpo \
  --epochs 3 \
  --batch-size 4 \
  --head-batch-size 64 \
  --learning-rate 5e-4 \
  --beta 0.5
```

训练器先冻结视觉骨干、LoRA、二元违规头和属性头，只抽取一次每条偏好记录的池化图文特征。随后仅更新 `policy_head`。参考策略是起始检查点中策略头的冻结副本，因此不需要同时在显存中保存第二套 Qwen。

对偏好对 `(chosen, rejected)`，目标为：

```text
-log sigmoid(beta * ((log π(chosen)-log π(rejected))
                   -(log π_ref(chosen)-log π_ref(rejected))))
```

输出目录是可直接被现有服务加载的完整检查点，包含原 adapter、processor、未修改的二元头和属性头、优化后的 `policy_head.pt`、更新后的 `metadata.json` 以及 `dpo_metrics.json`。工具拒绝覆盖已有目录，并按 `group_id` 拆分训练与验证数据。

## 3. 仅验证链路的自动偏好

没有人工反馈时，可以从二元标签生成 Block/Allow 冒烟数据：

```bash
python scripts/build_bootstrap_preferences.py \
  --manifest data/public-pilot/manifest.jsonl \
  --split train \
  --max-records 128 \
  --output outputs/preferences/bootstrap.jsonl
```

这批数据只验证数据读取、视觉特征抽取、DPO 更新和检查点保存是否贯通。它复述已有二元标签，不提供新知识，也没有 `review` 偏好，不能作为 RL 效果结论。

## 4. 应报告的指标

`dpo_metrics.json` 保存训练前后数据：

- `preference_accuracy`：策略头给 chosen 的概率是否高于 rejected；
- `mean_policy_margin`：`log P(chosen) - log P(rejected)` 的均值；
- `mean_advantage`：相对冻结参考策略增加的偏好边际；
- `mean_kl`：当前策略到参考策略的 KL 散度；
- 特征抽取时间、策略头训练时间和 CUDA 峰值显存。

正式比较还应使用固定、组隔离的测试集，同时报告二元违规指标、三分类混淆矩阵、人工复审率和误阻断率。要让 `review` 成为可靠动作，偏好数据必须包含足够的人工 review 选择以及与 block、allow 的成对比较。

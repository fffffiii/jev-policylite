# 人工反馈与离散 DPO

策略头有两种训练方式：清单中的 `decision` 字段可直接监督训练 `block / review / allow`；人工复核产生的动作偏好可继续对同一策略头执行离散 DPO。

## 1. 构建人工偏好对

```bash
python scripts/build_preference_pairs.py \
  --feedback data/review_feedback.jsonl \
  --output outputs/preferences/review_pairs.jsonl \
  --strict
```

输入格式见 [review_feedback.example.jsonl](../data/review_feedback.example.jsonl)。每行包含图片路径、正文、审核规则、人工动作和原模型动作。建议显式提供 `group_id`：同一原图、裁剪版本和近重复图必须使用同一组。转换器会保留该字段；旧记录未提供时仅按 `image` 路径分组，不按反馈记录 ID 分组。图片路径不同但内容重复的情况，需要人工统一分组。人工动作成为 `chosen`，原模型动作成为 `rejected`；二者相同时不会产生偏好样本。

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

训练器先冻结视觉骨干、LoRA、二元违规头和属性头，为每条偏好记录抽取一次最后一个有效 token 的隐藏表征。随后仅更新 `policy_head`。参考策略是起始检查点中策略头的冻结副本，因此不需要同时在显存中保存第二套 Qwen。

对偏好对 `(chosen, rejected)`，目标为：

```text
-log sigmoid(beta * ((log π(chosen)-log π(rejected))
                   -(log π_ref(chosen)-log π_ref(rejected))))
```

输出目录包含现有加载器所需的检查点文件，但不包含基座模型；服务还需要二元校准文件。输出文件包括原 adapter、processor、未修改的二元头和属性头、优化后的 `policy_head.pt`、更新后的 `metadata.json` 以及 `dpo_metrics.json`。工具拒绝覆盖已有目录，并按 `group_id` 拆分训练与验证数据。

## 3. 用自动偏好检查流程

没有人工反馈时，可以从二元标签生成 block/allow 流程测试数据：

```bash
python scripts/build_bootstrap_preferences.py \
  --manifest data/public-pilot/manifest.jsonl \
  --split train \
  --max-records 128 \
  --output outputs/preferences/bootstrap.jsonl
```

这批数据只验证数据读取、视觉特征抽取、DPO 更新和检查点保存是否正常运行。它复述已有二元标签，不提供新知识，也没有 `review` 偏好，不能用来证明真实人工偏好训练有效。

## 4. 应报告的指标

`dpo_metrics.json` 保存训练前后数据：

- `preference_accuracy`：策略头给 chosen 的概率是否高于 rejected；
- `mean_policy_margin`：`log P(chosen) - log P(rejected)` 的均值；
- `mean_advantage`：相对冻结参考策略增加的偏好边际；
- `mean_kl`：当前策略到参考策略的 KL 散度；
- 特征抽取时间、策略头训练与评估时间、CUDA 峰值分配量。训练计时包括每轮验证及最终评估，不包括特征抽取、加载或保存。

`reward` 在本实现中作为偏好样本权重使用：训练器取其绝对值，要求是有限正数；它不是在线环境返回的奖励。转换器保存的 `prompt` 便于检查记录，训练器实际使用 `image`、`text` 和 `policy_text` 重新构造输入。

正式比较还应使用固定、组隔离的测试集，同时报告二元违规指标、三分类混淆矩阵、人工复审率和误阻断率。要让 `review` 成为可靠动作，偏好数据必须包含足够的人工 review 选择以及与 block、allow 的成对比较。

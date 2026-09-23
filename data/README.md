# 数据清单

训练数据使用 UTF-8 JSONL，每行代表“原图 × 审核政策”的一个判断样本。

必填字段：

- `sample_id`：样本唯一 ID。
- `group_id`：原图或近重复图片组 ID；同组不得跨数据划分。
- `image`：相对 `image_root` 的图片路径。
- `text`：内容正文，没有正文时填空字符串。
- `policy_id`、`policy_text`：政策版本和完整规则。
- `label`：`0` 表示不违规，`1` 表示违规。争议样本不要放入二元训练集。
- `split`：`train`、`validation`、`calibration` 或 `test`。

推荐字段：`attributes`、`decision`、`source`、`annotation_source`、`notes`。`attributes` 使用 `nudity`、`sexual_act`、`suggestive`、`medical` 等人工标签；`decision` 使用 `block`、`review`、`allow`。同一张图在两套政策下应使用相同 `group_id`。数据来源许可和访问条件需要单独登记。

当前属性编码器把空 `attributes` 视为缺少监督，不能用空列表表达“人工确认所有属性均为阴性”。这类数据需要另行定义完整的标注格式，不能靠文档描述替代实现。

只有 `label` 时，项目仍可训练二元违规头。要训练多头，至少需要人工属性标签或人工 `decision` 标签；把内容等级映射成属性、把二元标签映射成 block/allow 只能用于开发试验，不能证明医学属性或 review 路由有效。

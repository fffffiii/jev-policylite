# 开源发布前检查

- [x] 添加 MIT 项目许可证；基础模型、数据和第三方依赖仍遵循各自许可证。
- [ ] 运行 `git status --ignored`，确认权重、数据、反馈和浏览器 profile 没有进入暂存区。
- [ ] 检查 README、runbook、日志和截图中没有内网地址、用户名、密钥、令牌或私有路径。
- [ ] 删除或忽略真实审核图片、原始标注和生成的拼图。
- [ ] 运行 `python -m pytest -q`。
- [ ] 至少执行一次 `scripts/build_mosaic_manifest.py` 与 `scripts/evaluate.py`，将脱敏后的结果摘要写入 release note。
- [ ] 分别说明二元头、属性头、策略头的训练数据与已验证边界。
- [ ] 对 `review`、医学和艺术等未充分标注能力保持明确限制说明。

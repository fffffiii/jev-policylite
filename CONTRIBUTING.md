# 贡献说明

提交代码前请运行：

```bash
python -m pytest -q
```

除已审核的 `models/pilot-multihead-v0.1/` 适配器包外，不要提交其他模型权重、原始审核图片、人工复核记录、浏览器 profile、服务器地址、令牌或私有运行日志。新数据必须说明来源许可、是否包含敏感内容、标注规范和划分方式。

涉及审核规则或模型行为的修改，请说明规则文本、标签定义、原图分组方式和主要错误类型，并分别报告违规头、属性头、策略头的指标。代理标签与人工标注结果应分开，未验证的复审或属性能力不要写成已实现。

文档与图片修改请同步检查中英文说明、项目页和本地服务。技术图的可编辑源文件位于 `site/assets/*.svg`；修改后运行 `python scripts/render_diagrams.py` 更新 PNG，并检查两种格式是否一致。提交 PR 时列出已执行的测试及未测试项，不要把历史实验结果当作本次测试结果。

浏览器交互检查可单独运行，不会加载模型或发送真实请求：

```bash
pip install -e '.[browser]'
playwright install chromium
python scripts/check_web_ui.py
```

这个脚本直接加载本地 HTML/CSS/JS，并为 API 提供明确的测试数据。真实 HTTP 服务、CUDA 推理和 GPU 性能仍需另外验证。

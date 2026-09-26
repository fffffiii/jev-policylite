# 鉴黄系统对比与实验核查 v1

运行日期：2026-09-26。本报告比较已发布的 Jev-PolicyLite 试训检查点与三个公开鉴黄系统。数值由本仓库脚本在同一张 RTX 3090 上测得，不引用各模型作者在其他数据上的分数。

实测结论：Jev 201k 在本任务的校准后准确率为 **93.50%**，高于这三种未重新训练的公开模型；其四图批次耗时 **211.72 ms**，明显慢于 Falconsai 的 **54.21 ms**。Jev 50k 的四图批次为 **198.02 ms**，仅省约 6.5% 时间，准确率下降 2.5 个百分点。当前实现中，缩小图片并没有带来同等比例的加速；图文规则适配和轻量后训练的用途需要与纯视觉分类速度分别看待。

## 数据和评测协议

使用项目现有公开试训清单。训练、验证、校准、测试按原图隔离；测试集为 **600 张不同原图 × 2 条规则 = 1,200 条判断**，校准集为另外 **300 张原图 × 2 条规则 = 600 条判断**。每个集合均按来源 L1–L4 等级平衡，测试集每级 150 张。`pilot-suggestive-v1` 将 L2–L4 作为正例，`pilot-explicit-v1` 将 L3–L4 作为正例。

这是一份政策映射的开发基准。标签来自发布方等级，未经过本项目独立人工复核；Jev-PolicyLite 在同来源的训练划分上训练过，外部模型保持公开权重，只允许校准阈值。这不是训练数据相同的架构消融，也不是来源完全独立的泛化测试。第三方预训练集与该数据源是否重叠，无法核实。

质量推理统一使用 batch=4。每个模型在两个政策上分别使用校准集选择最大 F1 阈值，并固定到测试集；F1 并列时取较高阈值。同时保留 0.5 阈值结果。准确率置信区间采用 1,000 次原图分组 bootstrap，每次一起抽取同图的两个政策判断。校准集误报率不超过 5% 的工作点也单独报告，并给出测试集实际误报率，不能把它理解为测试误报保证。

### 对照系统与标签映射

| 系统 | 原生输出与分类粒度 | 本次二元映射 | 文本/规则输入 | 本次不能验证的内容 |
| --- | --- | --- | --- | --- |
| [Falconsai](https://huggingface.co/Falconsai/nsfw_image_detection) | `normal / nsfw`，整图二分类 | 两条政策均用 `P(nsfw)`，各自校准阈值 | 无 | 没有性暗示、性行为和区域框输出 |
| [Giacomo ViT](https://huggingface.co/giacomoarienti/nsfw-classifier) | `drawings / hentai / neutral / porn / sexy`，整图五分类 | 明确色情规则：`P(porn)+P(hentai)`；性暗示规则再加 `P(sexy)` | 无；由程序选类别和阈值 | 本数据的 L1–L4 不等于这五类，不能报告五分类准确率 |
| [NudeNet 320n](https://github.com/notAI-tech/NudeNet) | 18 种人体/覆盖状态标签及检测框，包含脸、脚等非违规类别 | 对臀部、女性胸部、女性生殖器、男性生殖器、肛门的裸露框取最大置信度；无框记 0，两条政策各自校准 | 无 | 无检测框真值，不能报告区域 mAP；该映射不能充分表达性暗示 |
| Jev-PolicyLite | 二元违规 + 4 个属性分数 + 3 个处置分数，整图输出；另有四图位置头 | 直接使用输入政策下的违规头分数 | 图片 + 正文 + 自然语言规则 | 医学属性无正例；复审无独立标注；未验证未见政策泛化；不输出区域框 |

Falconsai 与 Giacomo 固定为下载脚本中的 Hugging Face revision；NudeNet 固定 `nudenet==3.4.2` 内置 `320n.onnx`，结果记录 SHA-256。NudeNet 使用原包的图像预处理、NMS 和框后处理，显式构造 ONNX Runtime CUDA session，并拒绝静默回退为纯 CPU。分类器使用各自原处理器的 PIL 路径，没有重新训练、裁改权重或改动类别。

## 质量结果

<!-- BEGIN QUALITY_TABLE -->

| 模型 | 准确率 @0.5 | 校准后准确率 | F1 | 召回率 | 误报率 | 准确率 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Falconsai ViT | 80.33% | 83.33% | 86.60% | 86.13% | 21.33% | 81.17%–85.42% |
| Giacomo ViT (5-class) | 74.92% | 76.42% | 82.46% | 88.67% | 44.00% | 74.00%–79.00% |
| NudeNet 320n | 75.83% | 80.17% | 85.58% | 94.13% | 43.11% | 78.00%–82.17% |
| Jev-PolicyLite 50k | 90.75% | 91.00% | 92.92% | 94.53% | 14.89% | 89.25%–92.58% |
| Jev-PolicyLite 201k | 94.08% | 93.50% | 94.72% | 93.33% | 6.22% | 91.83%–95.00% |

<!-- END QUALITY_TABLE -->

“校准”只调整阈值，不表示概率经过校准。合并结果中违规样本占 62.5%；准确率需要与召回、FPR、F1 一起看。

高分辨率 Jev 在 0.5 阈值下准确率为 **94.08%**，按统一校准协议后为 **93.50%**，说明在校准集上选得更好的阈值不保证改善测试集。NudeNet 在性暗示政策上的最大 F1 阈值退化为 **0**，把所有图片判为违规：该政策测试召回与误报均为 100%。因此不能用其合并后的高召回宣称适合性暗示审核；完整政策表保留这个失败结果。

### 按政策拆分

<!-- BEGIN POLICY_TABLE -->

| 模型 | 政策 | 阈值 | 准确率 | F1 | 召回率 | FPR | AP | 校准 FPR≤5% 工作点的测试召回 / FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Falconsai ViT | pilot-suggestive-v1 | 0.558688 | 88.00% | 91.72% | 88.67% | 14.00% | 98.17% | 75.33% / 4.67% |
| Falconsai ViT | pilot-explicit-v1 | 0.999208 | 78.67% | 79.42% | 82.33% | 25.00% | 89.22% | 51.67% / 2.33% |
| Giacomo ViT (5-class) | pilot-suggestive-v1 | 0.325995 | 87.17% | 91.49% | 92.00% | 27.33% | 94.49% | 45.78% / 6.67% |
| Giacomo ViT (5-class) | pilot-explicit-v1 | 0.013334 | 65.67% | 70.90% | 83.67% | 52.33% | 68.19% | 19.00% / 5.33% |
| NudeNet 320n | pilot-suggestive-v1 | 0.000000 | 75.00% | 85.71% | 100.00% | 100.00% | 91.10% | 48.67% / 2.00% |
| NudeNet 320n | pilot-explicit-v1 | 0.290507 | 85.33% | 85.33% | 85.33% | 14.67% | 90.10% | 67.33% / 4.33% |
| Jev-PolicyLite 50k | pilot-suggestive-v1 | 0.130285 | 91.17% | 94.25% | 96.44% | 24.67% | 98.91% | 81.56% / 2.67% |
| Jev-PolicyLite 50k | pilot-explicit-v1 | 0.100879 | 90.83% | 90.91% | 91.67% | 10.00% | 96.92% | 87.67% / 5.33% |
| Jev-PolicyLite 201k | pilot-suggestive-v1 | 0.728748 | 93.50% | 95.62% | 94.67% | 10.00% | 99.19% | 82.67% / 2.67% |
| Jev-PolicyLite 201k | pilot-explicit-v1 | 0.583678 | 93.50% | 93.36% | 91.33% | 4.33% | 98.61% | 94.33% / 6.67% |

<!-- END POLICY_TABLE -->

最后一列是以校准集 FPR≤5% 选择阈值后的**测试召回 / 测试 FPR**。AP 是 average precision，不是梯形积分的 PR 面积。来源各等级的正确率如下；它衡量二元政策判断，不是四级分类准确率。

![来源等级错误分布](figures/nsfw-source-levels.png)

## 速度和内存

<!-- BEGIN SPEED_TABLE -->

| 模型 | B=1 均值 / P95 (ms) | B=4 均值 / P95 (ms) | B=4 图片/秒 | 峰值 RSS (MiB) | Torch 分配显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Falconsai ViT | 21.50 / 24.36 | 54.21 / 57.46 | 73.8 | 1131 | 539 |
| Giacomo ViT (5-class) | 20.58 / 22.69 | 48.17 / 51.46 | 83.0 | 1144 | 539 |
| NudeNet 320n | 27.05 / 33.03 | 88.80 / 105.77 | 45.0 | 1170 | N/A (ONNX) |
| Jev-PolicyLite 50k | 164.96 / 167.33 | 198.02 / 201.80 | 20.2 | 2703 | 1749 |
| Jev-PolicyLite 201k | 169.95 / 172.36 | 211.72 / 225.72 | 18.9 | 2705 | 2012 |

<!-- END SPEED_TABLE -->

`50k / 201k` 分别表示 Jev 输入图像面积上限 50,176 / 200,704 像素，并非固定 224×224 / 448×448；处理器保持长宽比并按视觉 patch 网格取整。两个 ViT 的固定输入是 224×224，NudeNet 为 320×320。各自采用原生预处理，因此这是部署配置之间的比较，而不是强制像素完全相同的算子测试。

计时使用相同的 **120 张不同测试原图**，每个来源等级 30 张、两条政策均衡，每种 batch 重复 3 轮。B=1 共 360 次调用，B=4 共 90 次调用。正式计时前，每个模型先完整运行全部 120 次单图与 30 次四图批次，覆盖动态尺寸。正式阶段打乱批次顺序，对同一批输入随机轮换五个模型，逐个同步执行，没有并发推理。表中的 B=4 耗时是四张图一起完成的时间，吞吐为该模型处理图片数除以它自己的累计调用时间，不是共享服务的总体吞吐。

包含磁盘读图、解码、缩放与归一化、CPU→GPU、模型前向、全部原生输出头或检测框后处理，以及结果转回 CPU。Jev 的三个审核头全部启用。计时前后同步 GPU，不包括加载模型、HTTP 上传/返回、网络或排队；文件系统缓存已预热。固定 PyTorch/ONNX CPU 线程数为 4。计时阶段五个模型在同一进程常驻，只有一个模型在任一时刻执行；内存另取逐模型进程测量，不能把五模型常驻的总显存分摊为每个模型的使用量。

机器为共享双卡主机，本次使用 GPU 0，原审核服务常驻但在检查时空闲。最初质量评测期间 GPU 1 有另一个训练任务；该任务结束后重新计时，又发现只预热首批会遗漏新尺寸初始化：201k B=4 的第一轮出现 2,832.73 ms 调用，后两轮整轮均值约 168–169 ms。该轮原始记录保留在 [before-full-warmup](results/nsfw-comparison-v1/before-full-warmup/)。最终表格采用全输入预热后的交错计时，所有正式调用均保留，不剔除异常值。完整调度与设备快照见 [paired_timing_protocol.json](results/nsfw-comparison-v1/paired_timing_protocol.json)，质量预测保持不变。它仍不是独占服务器的延迟服务等级保证，也不是手机或 CPU 的结果。旧四图报告日期与负载不同，数值不能合并为本次运行。

PyTorch 模型使用 BF16 autocast；Jev 主干权重为 BF16，ViT 权重为 FP32。NudeNet 使用 FP32 ONNX。Qwen 环境未安装 `causal_conv1d`，日志确认走 PyTorch 参考实现；未使用 TensorRT、`torch.compile` 或定制融合内核。因此速度结论仅针对这份可复现的软件配置，不代表各架构能达到的最高速度。Giacomo 的旧处理器类名 `ViTFeatureExtractor` 在 Transformers 5 中不能自动加载，脚本用 `ViTImageProcessor` 读取其原始参数，未改变均值、方差、尺寸或插值。

内存来自前一轮逐模型独立进程，使用同样 120 张计时图片与两个 batch 配置，其环境单独记录在 `isolated_memory_environment`。峰值 RSS 是 Linux 整个进程生命周期的 CPU 常驻内存高水位，包含加载阶段和导入的公共 PyTorch 库；NudeNet 的 RSS 也包括公共评测进程的依赖。Torch allocated 是预热后重置计数器测得的 PyTorch 峰值显存分配，不是完整进程显存，也不是 `nvidia-smi`。ONNX 显存不能用这个计数器测量，明确记为 N/A。两列不能相加，也不应据此推算手机内存。

![速度与识别质量](figures/nsfw-speed-quality.png)

## 细分类头的实际支持情况

<!-- BEGIN ATTRIBUTE_TABLE -->

| 属性 | 测试原图正例 | 50k F1 @0.5 | 201k F1 @0.5 |
| --- | ---: | ---: | ---: |
| 裸露 | 300 | 90.45% | 93.55% |
| 性行为 | 150 | 81.61% | 89.90% |
| 性暗示 | 150 | 74.56% | 82.74% |
| 医学 | 0 | 不可评估 | 不可评估 |

<!-- END ATTRIBUTE_TABLE -->

F1 按 1,200 条图像—政策记录计算，正例数一栏列出去重后的原图数。这些属性结果全部基于等级代理标签：L2→性暗示，L3→裸露，L4→裸露+性行为，L1→全阴性。它们不是医学、艺术或具体性行为的人工多标签测试；代理映射甚至不为 L3/L4 同时标注性暗示。医学类别没有正例，不能用“全预测阴性”的高准确率宣传该能力。策略头的 review 类也缺少正例，不能报告有效的三分类处置准确率。

## 核查发现与修正

1. 源清单为 3,000 张原图、6,000 条双政策记录；原图分组和保存的像素散列均未跨训练、验证、校准和测试划分。远端实测的 LoRA 与三个头的 SHA-256 均与本仓库发布文件一致。
2. 四图训练/验证/测试为 480/120/200 组，覆盖 1,127/234/426 张原图；测试的 800 个位置标签不能视作 800 张独立图片。旧报告已补充准确计数。
3. 旧四图组合 ID 在不同 split 目录复用。原实验按目录加载，因此没有把测试图送进训练，但直接合并清单可能发生 ID 冲突。新生成器已把 split 纳入 ID，读取器也增加了重复 ID 和集合一致性校验。
4. 四图局部表征仍受因果前缀影响，后面的图可以看到前面的图；与单图 batch 的独立序列不同。文档已明确，避免将其表述为四张图各自完全独立的编码。
5. 旧六方法配对计时只有汇总，没有全部逐组原始计时，无法追溯估计置信区间。更新脚本会保存逐组概率与延迟；本次外部对照保存全部预测、逐批计时、版本和校准工作点。
6. 新增回归测试确保阈值不读取测试结果、同图重复政策不虚增 bootstrap 样本量、重复清单 ID 在编码前被拒绝。
7. 修正新对照工具只预热首批的计时缺口，改为覆盖全部待测输入。交错计时脚本减少模型顺序与主机负载漂移的关联；旧轮次原始结果保留，未用手工删除离群点来降低均值。

完整检查输出见 [audit.json](results/nsfw-comparison-v1/audit.json)。本轮为数据协议和评测工具核查，不等同于对仓库全部功能的形式化验证。

## 复现

Linux + CUDA 环境中，先按 README 准备项目及 Qwen 试训检查点、公开试训数据。第三方对照包作为可选实验依赖，不是审核服务的运行依赖：

```bash
pip install -e '.[dev,plots]'
pip install --no-deps nudenet==3.4.2 onnxruntime-gpu==1.20.2 \
  opencv-python-headless==4.11.0.86 coloredlogs flatbuffers humanfriendly
python scripts/download_nsfw_baselines.py
python scripts/audit_experiment_data.py

# 提供 CUDA 12 / cuDNN 9 的动态库搜索路径；路径随安装位置变化。
# 本次还将 nvidia/cuda_nvrtc/lib 加入 LD_LIBRARY_PATH，避免 ONNX CUDA provider 回退。
CUDA_VISIBLE_DEVICES=0 bash scripts/run_nsfw_comparison.sh

# 已有完整预测时可仅重新计时；脚本核对清单散列和样本集合。
CUDA_VISIBLE_DEVICES=0 TIMING_ONLY=1 bash scripts/run_nsfw_comparison.sh

# README 的最终速度表来自全输入预热后的交错运行，需容纳五个常驻模型：
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_nsfw_paired.py

# 将本轮 *.json 和 *-predictions.jsonl 放入此目录后生成全部图表：
python scripts/report_nsfw_benchmark.py --results docs/results/nsfw-comparison-v1
python -m pytest -q
```

本次使用 Python 3.10、PyTorch 2.5.1+cu121、Transformers 5.17.0、ONNX Runtime GPU 1.20.2。完整依赖版本和设备快照记录在每个模型的 JSON 中。脚本默认检查点是 `outputs/public-pilot-multihead`，使用下载的开源模型包时，请在单独运行 `benchmark_nsfw.py` 时传 `--checkpoint models/pilot-multihead-v0.1`。绘图脚本同时输出 PNG、SVG 和 PDF，保留原始汇总文件，不手工改数。

## 来源与许可

- [Falconsai 模型卡](https://huggingface.co/Falconsai/nsfw_image_detection)：Apache-2.0；固定 revision `96cb0d0342c7afb80cab76ecc58b265fa44da256`。
- [Giacomo 模型卡](https://huggingface.co/giacomoarienti/nsfw-classifier)：CC-BY-NC-ND-4.0；固定 revision `29f43cab33874e62db8a1973bd0f84b0c69ff057`。用于本次研究评测，不作为本项目 MIT 权重分发。
- [NudeNet 仓库](https://github.com/notAI-tech/NudeNet)、[标签和后处理实现](https://github.com/notAI-tech/NudeNet/blob/v3/nudenet/nudenet.py)、[AGPL-3.0 许可](https://github.com/notAI-tech/NudeNet/blob/v3/LICENSE)。本项目不复制第三方源码或权重，评测时加载另行安装的包。
- [ONNX Runtime CUDA provider 要求](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)：用于核对 CUDA/cuDNN 兼容与 provider 加载。
- [来源数据](https://huggingface.co/datasets/jiangchengchengNLP/cashbox-nsfw-image-level)：准备脚本固定 revision，并记录等级映射与去重方式。

本仓库 MIT 许可只覆盖本项目代码，不替代上述第三方模型和数据许可。公开结果文件不包含原图。

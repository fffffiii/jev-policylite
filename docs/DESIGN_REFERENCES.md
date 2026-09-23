# Project-page design and figures

The project page presents the method first, followed by experiment settings, results, limitations and runnable commands. Headings describe what the code does rather than making speed or accuracy promises.

The original page drew layout references from [Nerfies](https://nerfies.github.io/), [Mip-NeRF 360](https://mipnerf360.github.io/), [TRL](https://github.com/huggingface/trl) and [Unsloth](https://github.com/unslothai/unsloth). These are design references, not affiliated projects or performance baselines.

## Figure sources

| Figure | Editable source | Raster fallback |
| --- | --- | --- |
| Model architecture | [architecture.svg](../site/assets/architecture.svg) | [architecture.png](../site/assets/architecture.png) |
| Policy-head DPO | [post-training.svg](../site/assets/post-training.svg) | [post-training.png](../site/assets/post-training.png) |

The SVGs contain editable text, shapes and arrows, not embedded screenshots. They are checked against `model.py`, `heads.py`, `feedback.py` and `train_policy_dpo.py`. The README and project page use SVG; PNG files are provided for tools that do not support SVG. No experimental image or benchmark result is drawn into either figure.

The architecture selects the final hidden state of the **last valid token**, not an average of token states. All three heads read it independently. The DPO diagram separates image/text/policy inputs from the `chosen` and `rejected` **action labels** used by the loss. Blue denotes frozen modules; orange denotes the trainable head. The orange dashed arrow is the gradient path; the dark dashed line carries action labels.

To update the raster versions:

```bash
pip install -e '.[browser]'
python -m playwright install chromium
python scripts/render_diagrams.py
python scripts/render_diagrams.py --check
```

Rendering uses Chromium's SVG engine. The PNG includes the SHA-256 of its SVG source; `--check` verifies the source hash and image dimensions. On a machine with different fonts or browser versions, regenerate and visually inspect the PNGs.

Keep experiment timings in the result tables, with hardware, sample counts and timing boundaries. Do not put unverified PR statistics or benchmark numbers into explanatory figures.

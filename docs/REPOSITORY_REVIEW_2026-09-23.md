# Repository review — 2026-09-23

## Scope

Base revision: `97bd01ee3b28fa83073366b6cba7eb615108d49a` on `main`. The reviewer's two remote branches contain temporary transfer payloads only; no PR was submitted from them. The changes were reconstructed from that verified payload onto a clean local integration branch. Temporary transfer files are excluded from the integration branch.

The inventory contained **99 original files**: 42 Python files, 14 Markdown documents, both web interfaces, configuration/deployment files, two technical PNGs, an SVG favicon, four JPEG fixtures, and model metadata/LFS pointers. The review focused on wording, consistency between documentation and code, figure semantics, user-visible states and the small defects directly connected to those claims. It is not an exhaustive security audit or a model-performance certification.

The original review used a temporary Actions snapshot. Its first archive failed while resolving LFS data; a second source-only archive used `GIT_LFS_SKIP_SMUDGE=1`. The integration recovered the reviewed text patch from remote transfer payloads and regenerated PNGs from the reviewed SVGs. These checks did not rerun model inference.

## Findings addressed

| Area | Problem | Change |
| --- | --- | --- |
| Project description | Slogans implied general rule understanding, rapid end-to-end adaptation, or deployment readiness | Describe actual inputs/outputs and separate measured trials from unverified capabilities |
| Head names | Historical `decision_head.pt` and “Decision Head” could be mistaken for the three-class policy head | Keep filenames/API keys stable; label the binary head and policy suggestions separately |
| DPO explanation | Some text called it DPO/GRPO or “fast RL”; figures suggested preferred text responses | Use discrete action DPO consistently; show action labels entering the loss, not the encoder |
| DPO timing | 0.19 s could be read as the total workflow | State that extraction took 46.23 s separately; training timing also includes final evaluation |
| Architecture diagram | Generic pooling and deployment icons obscured what is implemented | Show the last valid token, three independent branches, and distinct supervised/DPO update scopes |
| Feedback grouping | `build_preference_pair` discarded `group_id`; missing groups could become individual feedback IDs | Preserve explicit groups; use image-path fallback, with documented near-duplicate limitations |
| Service startup | Model-package example omitted a required test-metrics file; processor was not checked early | Make display metrics optional and validate the processor path |
| Local UI | Fixed sample count, dual-GPU wording and fixed upload size could describe the wrong runtime | Display returned values, explicit missing-data states and MiB/GiB units |
| Rule switching | Hidden custom-rule text was still submitted after selecting a preset | Submit custom text only when the custom option is selected |
| Privacy and timing | “Images are not retained” omitted temporary disk writes; “total time” excluded upload/decoding | Explain temporary storage and label service-processing time precisely |
| Mosaic results | Wording inferred a decrease caused by shrinking without a paired single-image baseline | Report scenario-level results without that causal claim |
| Packaging | Web HTML/CSS/JS were not declared as wheel package data | Include the three static files and verify the built wheel |

The two figures are [architecture.svg](../site/assets/architecture.svg) and [post-training.svg](../site/assets/post-training.svg). PNG fallbacks are rendered from these files. The four photographic test fixtures are unchanged; see [their provenance note](../data/smoke/README.md).

## Local validation

Original review environment: Linux, Python **3.13.5**, PyTorch **2.10.0+cpu**, pytest **9.0.2**, Playwright **1.57.0**, CairoSVG **2.8.2**. Integration environment: Windows, Python **3.11**, with Chromium for SVG rendering. No CUDA inference was run during integration.

| Check | Result and scope |
| --- | --- |
| Baseline unit suite | 17 passed before changes |
| Updated unit suite | 34 passed; includes grouping, report handling, head-output compatibility, local links and SVG structure |
| Browser interactions | 108 checks passed across 1440/768/390/320 px project-page widths and 1440/390/320 px service-UI widths |
| Browser cases | Stage/code tabs, keyboard navigation, clipboard failure fallback, SVG dialogs, missing/null metrics, independent binary/policy outputs, custom-to-preset switching, invalid files, GPU display and no horizontal page overflow |
| PNG regeneration | Both SVGs render successfully in Chromium; SVG source hashes and image dimensions match PNG metadata |
| Package | Wheel builds without installing model dependencies; HTML/CSS/JS are present in the archive |
| Syntax | Python compileall, both JavaScript parsers, shell syntax and `git diff --check` pass |
| Data tools | Smoke manifest validates; human-feedback and bootstrap converters run on small fixtures |

**Browser boundary:** the checker loads local HTML/CSS/JS directly into Chromium and supplies explicit in-browser API fixtures. Its 108 checks verify DOM behavior and layout, not FastAPI routing, authentication, network latency or real model predictions. Screenshot filenames containing `service-fixture` identify simulated outputs. The checker writes screenshots and a JSON report under `outputs/web-review/` by default.

Reproduce the checks:

```bash
python -m pytest -q
python scripts/render_diagrams.py --check
python scripts/check_web_ui.py
python -m compileall -q src scripts tests
node --check site/assets/main.js
node --check src/qwen35_moderation/web/static/app.js
bash -n scripts/start_public_pilot_multihead.sh
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

Install the optional `browser` dependency and Chromium before generating PNGs or running the browser checker. The new Actions workflow runs CPU unit/documentation checks and verifies raster/wheel contents; its result should be read from Actions after the update.

## Still requiring separate work

- **Real checkpoints and hardware:** verify LFS downloads with authorized access, prepare the base model and processor, then test model loading, inference, supervised training, DPO and export. None was rerun here. Historical RTX 3090/Q4 numbers are retained as historical records, not reproduced measurements.
- **Serving behavior:** live HTTP behavior, GPU concurrency/queue handling, disconnects, authentication and production deployment remain outside this review. The service is still a local research interface, not a hardened public endpoint.
- **Data and labels:** real review-action and medical-context supervision remains insufficient. Empty attribute lists currently mean absent supervision, not “all attributes confirmed negative.” Near-duplicate grouping still depends on explicit group IDs.
- **Provenance:** per-image licensing records for the four smoke photographs are not present in this snapshot. Resolve them before redistribution. External references and their availability were not revalidated in this review.

No model-weight contents, training prompts, content-policy text, original JPEGs, original smoke labels or historical numerical results were replaced. The deliberate behavior change affecting data preparation is preservation of image groups. Display fixes and optional report loading do not change model tensor computations or threshold formulas.

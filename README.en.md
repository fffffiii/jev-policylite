# Jev-PolicyLite

**A lightweight multimodal decision model for content moderation**

[中文](README.md) | [English](README.en.md)

[Model](models/pilot-multihead-v0.1) · [Training](docs/TRAINING_GUIDE.md) · [Experiments](docs/EXPERIMENTS.md) · [Project page source](site/) · [MIT](LICENSE)

Jev-PolicyLite adapts Qwen3.5-0.8B to content moderation. Given an image, accompanying text, and a moderation policy, it predicts a violation score, visual attributes, and a handling action. Inspired by Jev / NanoJev's use of hidden representations for direct scoring, the project implements multi-head training, review-feedback processing, and policy-head preference optimization for moderation experiments. The pilot uses sexual-content severity labels, not a general sensitive-content benchmark.

We study two questions: whether a small model can moderate content using both multimodal evidence and policy text, and the training cost of adjusting its actions. Benefits from real human feedback still need separate evaluation. Three moderation heads share one representation. During preference optimization, the backbone is frozen, features are cached, and only the head responsible for blocking, review, or allowing is updated.

<p align="center">
  <img src="site/assets/architecture-paper.png" alt="Jev-PolicyLite: image, text, and policy share a representation feeding violation, attribute, and policy heads." width="100%">
</p>

## What is included

- **Multi-head moderation:** Qwen3.5's multimodal backbone with task LoRA and separate violation, attribute, and action heads.
- **Policy-head post-training:** tools that convert review corrections into action preferences and run discrete DPO on cached frozen features.
- **Training and evaluation tools:** data validation, image-group splits, threshold calibration, head-level evaluation, and 2×2 mosaic stress tests.
- **Model and local service:** a pilot LoRA adapter, three moderation heads, and a FastAPI web interface with resource monitoring.

The repository records training, DPO, and local-deployment trials. The results below come from those experiment notes; they are not a production evaluation.

## Method

### Image, text, and policy input

The model reads the last valid token's hidden representation and passes it to three `LayerNorm + Linear` heads:

| Head | Output | Purpose |
| --- | --- | --- |
| Violation | Sigmoid score | Predict whether content violates the supplied policy |
| Attribute | Independent sigmoid scores | Predict co-occurring attributes: `nudity`, `sexual_act`, `suggestive`, `medical` |
| Policy | Three-way softmax | Predict `block`, `review`, or `allow` |

Supervised adaptation warms up the task heads, then trains language LoRA and the heads while keeping the visual encoder frozen. Attributes and handling actions are modeled separately: nudity and a medical context may coexist, while the policy head learns the handling action.

Jev / NanoJev inspired the direct-scoring and decision-interface design. This implementation uses shared representations for fixed moderation labels and adds policy-head DPO. It does not implement dynamic candidate-set attention or reproduce the full Jev system.

### Policy-head preference optimization

A review record pairs the reviewer's preferred action, `chosen`, with the model's original action being corrected, `rejected`, under the same image, text, and policy.

The backbone, LoRA, violation head, and attribute head are frozen. Multimodal features are extracted once. The trainable policy head and frozen reference head read the same features, and discrete DPO updates the action probabilities. The reference requires only a copy of the small policy head.

<p align="center">
  <img src="site/assets/post-training-paper.png" alt="Post-training: construct action preferences, cache frozen features, and update the policy head with discrete DPO." width="100%">
</p>

Feature reuse reduces repeated encoding. This is useful when existing features distinguish the relevant evidence but the action needs adjustment. Updating the head cannot recover visual details absent from those features. The implementation optimizes discrete action preferences; it does not include online rollouts, PPO, or GRPO.

## Results

### DPO workflow validation on an RTX 3090

Binary labels were used to construct 128 block/allow preference pairs, split by image group into 103 training and 25 validation pairs. Training ran for two epochs.

| Validation metric | Before | After |
| --- | ---: | ---: |
| DPO loss | 0.6931 | 0.4371 |
| Preference ranking accuracy | 100.0% | 100.0% |
| Mean chosen/rejected log-probability margin | 6.7104 | 7.9800 |
| KL to the reference policy | 0 | 0.00119 |

Feature extraction for 128 records took **46.23 s**. Two policy-head epochs and evaluation took **0.19 s**. Peak CUDA allocation was **2.00 GiB**. Timings exclude model loading and saving; the 0.19 s figure also excludes feature extraction. The script includes the final training/validation evaluation in this interval.

These preferences repeat existing binary labels, and all validation pairs were ranked correctly before training. The run establishes that the workflow operates, not a benefit from real human feedback or a validated review action. See the [DPO experiment record](docs/DPO_SMOKE_RESULT_V1.md).

### Mosaic and edge experiments

| Experiment | Measured result | Interpretation |
| --- | --- | --- |
| 200 mosaics, 2×2 layout | 89.5% accuracy, 90.7% recall, 14.0% FPR | A mosaic is positive if any tile is positive; this does not evaluate tile localization |
| Mosaics with one violating tile | 84.0% accuracy | Tests small-target and multi-image interference |
| Linux x86_64, single-head Q4, four CPU threads | About 479 MB model directory; about 859 MiB peak RSS | Decisions matched on 8 sampled cases; full regression is pending |

Details: [mosaic results](docs/MOSAIC_2X2_RESULT_V1.md) and [edge status](docs/EDGE_STATUS.md). Mobile devices, quantized multi-head inference, and Windows deployment have not yet been measured.

## Getting started

Requirements: Python 3.10+, Git LFS, and Transformers 5.x. GPU training requires a compatible CUDA build of PyTorch. Commands below use Bash.

```bash
git lfs install
git clone https://github.com/fffffiii/jev-policylite.git
cd jev-policylite
git lfs pull

python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,web]'
```

### Load the pilot model

The [model directory](models/pilot-multihead-v0.1) contains approximately 43 MB of LoRA weights, three moderation heads, calibration, and metadata. Download the Qwen base model separately. The adapter package also omits the processor; save the base processor into the checkpoint directory before first use:

```bash
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('models/pilot-multihead-v0.1/processor')"
```

The base model's first load requires network access or an existing local cache. Offline loading requires the base model, adapter, and processor to be available locally.

```bash
python scripts/predict.py \
  --checkpoint models/pilot-multihead-v0.1 \
  --image path/to/image.jpg \
  --text "image caption" \
  --policy-id strict-v1 \
  --policy-file policies/strict.txt
```

This command returns the binary violation result, not the attribute or policy-head outputs. See the [training guide](docs/TRAINING_GUIDE.md) and [model card](models/pilot-multihead-v0.1/README.md) for multi-head outputs and service usage.

### Supervised training

Prepare a JSONL manifest using the [data schema](data/README.md), then configure paths and training parameters in `configs/train.yaml`:

```bash
python scripts/validate_data.py --manifest data/manifest.jsonl --image-root .
python scripts/train.py --config configs/train.yaml
```

For multi-head settings, see `configs/public-pilot-multihead.yaml`. Attribute and policy heads require their corresponding labels; mapping binary labels to block/allow does not teach review. Full instructions are in the [training guide](docs/TRAINING_GUIDE.md).

### Update the policy from review feedback

After preparing the model above, place review records in `data/review_feedback.jsonl`:

```bash
python scripts/build_preference_pairs.py \
  --feedback data/review_feedback.jsonl \
  --output outputs/preferences/review_pairs.jsonl \
  --strict

python scripts/train_policy_dpo.py \
  --checkpoint models/pilot-multihead-v0.1 \
  --preferences outputs/preferences/review_pairs.jsonl \
  --output-dir outputs/policy-dpo \
  --epochs 3 --beta 0.5
```

Provide a `group_id` shared by every version of an image and its near duplicates. Without it, the converter groups by image path only; it cannot identify near duplicates. The trainer splits training and validation data by group and refuses to overwrite an existing output directory. It saves the adapter, processor, task heads, and `dpo_metrics.json`; loading still requires the base model. See the [post-training guide](docs/HUMAN_FEEDBACK_AND_RL.md) for the preference schema and parameters.

### Local service

```bash
MODEL_CHECKPOINT=models/pilot-multihead-v0.1 \
CALIBRATION_FILE=models/pilot-multihead-v0.1/calibration.json \
python -m uvicorn qwen35_moderation.web.app:app \
  --host 127.0.0.1 --port 8089 --workers 1
```

Open `http://localhost:8089` to submit images and text and inspect results and runtime status. The main verdict comes from the binary violation head; separate cards show attribute scores and policy-head suggestions.

Offline metrics are optional. Set `TEST_METRICS_FILE` to the matching `test_metrics.json`; otherwise the page shows that no metrics were supplied. Images are written to a temporary directory during preprocessing and deleted afterward; the service does not keep an image or text history. There is no login or authentication layer. The example binds to localhost; configure access controls before exposing it to a network.

## Limitations

- Pilot attribute labels are derived from source content categories. Dedicated medical-context and review labels are insufficient, so reliability for these categories has not been established.
- Policy text is supported as input. Removed-policy, replaced-policy, and unseen-policy tests are still needed to establish whether the model uses policy semantics.
- Small development experiments cannot establish low false-positive guarantees. Independent evaluation should report recall at a stated FPR and results by content type.
- Matched comparisons against a vision-only classifier and same-backbone Yes/No scoring are pending. Current results do not establish an accuracy or end-to-end speed advantage from decision heads.

## Documentation and contributions

| Topic | Documentation |
| --- | --- |
| Data schema and annotation | [Data guide](data/README.md) |
| Supervised training, calibration, evaluation | [Training guide](docs/TRAINING_GUIDE.md) |
| Review feedback and discrete DPO | [Post-training guide](docs/HUMAN_FEEDBACK_AND_RL.md) |
| Multi-head and mosaic stress tests | [Experiment tools](docs/EXPERIMENTS.md) |
| Static project-page deployment | [GitHub Pages](docs/GITHUB_PAGES.md) |

Supporting guides include Chinese-language documentation. Contributions, reproduction reports, and error analyses are welcome. Run `python -m pytest -q` before submitting changes. Moderation changes should state the policies, labels, and data splits used; see [contribution guidelines](CONTRIBUTING.md). Real moderation images, review feedback, and private logs are excluded from the repository.

## License and acknowledgments

Project code is released under [MIT](LICENSE). Base models, datasets, and third-party dependencies retain their own licenses.

We thank Qwen for the multimodal backbone and Jev / NanoJev for the direct-decision design inspiration. Jev-PolicyLite is an independent project.

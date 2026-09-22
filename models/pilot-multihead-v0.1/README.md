# Pilot Multi-Head Adapter v0.1

This package is the public development adapter for Jev-PolicyLite. It adapts `Qwen/Qwen3.5-0.8B` with LoRA and provides separate binary-violation, visual-attribute, and policy-action heads.

## Files

- `adapter/adapter_model.safetensors`: LoRA adapter weights, stored with Git LFS.
- `adapter/adapter_config.json`: PEFT configuration for the stated base model.
- `decision_head.pt`: binary violation head.
- `attribute_head.pt`: multi-label attribute head for `nudity`, `sexual_act`, `suggestive`, and `medical`.
- `policy_head.pt`: three-way action head for `block`, `review`, and `allow`.
- `calibration.json`: pilot binary-head calibration artifact.
- `metadata.json`: training configuration and development validation summary.

## Loading

Install this repository, download the Qwen base model under its own license, then supply this directory as the checkpoint to the project commands. The adapter package does not include the base model, the processor, training images, manifests, review feedback, predictions, or merged model weights.

```bash
pip install -e '.[web]'
git lfs pull
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('models/pilot-multihead-v0.1/processor')"
python scripts/predict.py \
  --checkpoint models/pilot-multihead-v0.1 \
  --image path/to/image.jpg \
  --text "content caption" \
  --policy-id strict-v1 \
  --policy-file policies/strict.txt
```

## Training and evaluation boundary

This is a public-pilot development checkpoint. It was trained with source-level proxy labels and evaluated on a small development split. The `review` action and medical attribute do not have sufficient dedicated labels. Do not use it as a production safety system or interpret the recorded metrics as a guarantee of low false-positive rates, policy generalization, fine-grained recognition, or human-preference improvement.

The project source code is MIT licensed. This adapter remains dependent on `Qwen/Qwen3.5-0.8B` and its associated terms; base-model and dataset licenses are not replaced by the project license.

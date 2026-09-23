# Four-photo position heads v0.1

The position-head checkpoints predict a binary violation score for each of four positions: top-left, top-right, bottom-left, bottom-right. They use the frozen [pilot multi-head checkpoint](../pilot-multihead-v0.1) as the backbone. The independent-head variants contain 12,292 parameters; the shared-head variant contains 3,073. No base-model or LoRA weights are duplicated here.

- `native_four_position_heads.pt`: four original images in one Qwen prefill, each capped at 50,176 pixels.
- `native_local_four_position_heads.pt`: the same four-image prefill, with each head reading its corresponding image's `vision_end` hidden state.
- `native_shared_position_head.pt`: the same local hidden states scored by one head shared across all four images.
- `mosaic_four_position_heads.pt`: four images composed into one 448×448 JPEG, capped at 200,704 pixels.

Use the matching input mode and the exact prompts/preprocessing in `scripts/predict_multi_photo.py`. The source pilot checkpoint still needs its processor and the Qwen base model. All position heads were trained only on cached frozen features from the public-pilot development split. They are not interchangeable with the original single-image binary head.

```bash
python scripts/predict_multi_photo.py \
  --checkpoint models/pilot-multihead-v0.1 \
  --heads models/pilot-multi-photo-v0.1/native_local_four_position_heads.pt \
  --mode native_local \
  --images first.jpg second.jpg third.jpg fourth.jpg \
  --policy-file policies/strict.txt
```

The four input paths map to top-left, top-right, bottom-left, and bottom-right output positions. Scores are raw sigmoid outputs at a 0.5 development threshold, not calibrated probabilities. A one-shot CLI process includes cold CUDA initialization; the [experiment](../../docs/MULTI_PHOTO_RESULT_V1.md) measures warm in-process latency.

The tile labels come from each original image's policy-specific binary label, not a separate human annotation of the combined image. Test measurements and limitations are in [the four-photo experiment](../../docs/MULTI_PHOTO_RESULT_V1.md). The project code is MIT licensed; Qwen and dataset terms still apply to their respective assets.

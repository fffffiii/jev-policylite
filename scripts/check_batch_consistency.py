"""用不同尺寸图片检查右侧 padding 是否改变单样本判断。"""
from pathlib import Path
import json
import torch
from qwen35_moderation.data import read_manifest, configure_max_pixels
from qwen35_moderation.runtime import load_checkpoint
from qwen35_moderation.model import model_inputs


def main():
    torch.set_num_threads(4)
    model, _, _, collator = load_checkpoint("outputs/smoke-train", torch.device("cuda"))
    collator.max_length = 512
    configure_max_pixels(collator.processor, 200704)
    rows = read_manifest("data/public-pilot/manifest.jsonl")
    chosen = [next(row for row in rows if row["source_label"] == label) for label in range(4)]
    encoded = [collator._encode_one(row) for row in chosen]
    lengths = [item["input_ids"].shape[1] for item in encoded]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        singles = torch.cat([model(**{key: value.cuda() for key, value in item.items()}) for item in encoded])
        batch = model(**{key: value.cuda() for key, value in model_inputs(collator(chosen)).items()})
    error = (singles - batch).abs().max().item()
    report = {"lengths": lengths, "single_logits": singles.float().tolist(), "batch_logits": batch.float().tolist(), "max_logit_difference": error}
    Path("logs/batch-consistency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if error > .05:
        raise AssertionError(f"不同 batch 下误差过大：{error}")


if __name__ == "__main__":
    main()

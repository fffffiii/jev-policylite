from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 MNN 端侧量化抽样对照用例")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    predictions = read_jsonl(args.predictions)
    selected: dict[tuple[str, str], dict] = {}
    for prediction in predictions:
        level = next(item for item in prediction["attributes"] if item.startswith("source_L"))
        key = (level, prediction["policy_id"])
        selected.setdefault(key, prediction)

    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, key in enumerate(sorted(selected)):
        prediction = selected[key]
        row = rows[prediction["sample_id"]]
        image = (args.project_root / row["image"]).resolve()
        prompt = (
            f"<img>{image}</img>你是内容审核模型。请依据给定规则判断内容是否违规。\n"
            f"审核规则：{row['policy_text']}\n内容正文：{row['text']}\n只需进行内部判断。"
        )
        prompt_path = args.output / f"case-{index:02d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        cases.append(
            {
                "case": index,
                "sample_id": row["sample_id"],
                "source_level": key[0],
                "policy_id": key[1],
                "label": prediction["label"],
                "bf16_logit": prediction["logit"],
                "bf16_probability": prediction["probability"],
                "bf16_prediction": prediction["prediction"],
                "prompt": str(prompt_path),
            }
        )
    (args.output / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(cases, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

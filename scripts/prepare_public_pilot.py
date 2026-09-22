"""下载公开分级数据，生成按图片隔离的双规则试验集。"""
from __future__ import annotations

import hashlib
import io
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

REPO = "jiangchengchengNLP/cashbox-nsfw-image-level"
REVISION = "8e51d9bd6c64e744562026f07195c75ab18f2b02"
POLICIES = {
    "pilot-suggestive-v1": "允许正常姿态且无性暗示的轻度性感穿着；禁止强烈性暗示、诱惑性姿势、私密部位裸露及明确性行为。",
    "pilot-explicit-v1": "允许有衣物遮挡的性感穿着和性暗示姿势；禁止私密部位裸露、自慰、明确性行为及性器官特写。",
}


def main():
    root = Path("data/public-pilot")
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    files = [f"data/train-{i:05d}-of-00007.parquet" for i in range(7)]

    def download(name):
        print(f"download_start={name}", flush=True)
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, token=False)
        print(f"download_done={name}", flush=True)
        return path

    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(download, files))
    for name in ("README.md", "merge_stats.json"):
        path = download(name)
        (root / name).write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")

    rng = random.Random(42)
    counts = Counter()
    skipped = Counter()
    accepted = []
    hashes = []
    seen = set()
    # 候选顺序固定，跨所有标签统一去重后再划分，避免同图跨集合。
    for shard, path in enumerate(paths):
        table = pq.read_table(path)
        indices = list(range(table.num_rows))
        rng.shuffle(indices)
        for index in indices:
            label = int(table["label"][index].as_py())
            if counts[label] >= 750:
                continue
            try:
                raw = table["image"][index].as_py()["bytes"]
                picture = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
                pixel_hash = hashlib.sha256(str(picture.size).encode() + picture.tobytes()).hexdigest()
                thumb = np.asarray(picture.resize((9, 8)).convert("L"))
                dhash = int.from_bytes(np.packbits(thumb[:, 1:] > thumb[:, :-1]).tobytes(), "big")
                if pixel_hash in seen or any((dhash ^ old).bit_count() <= 4 for old in hashes):
                    skipped["duplicate_or_similar"] += 1
                    continue
                if min(picture.size) < 64:
                    skipped["too_small"] += 1
                    continue
                seen.add(pixel_hash)
                hashes.append(dhash)
                group = pixel_hash[:24]
                image_path = root / "images" / f"{group}.jpg"
                picture.save(image_path, quality=95)
                accepted.append({"group_id": group, "image": image_path.as_posix(), "source_label": label,
                                 "source_shard": files[shard], "source_row": index, "pixel_sha256": pixel_hash,
                                 "dhash": f"{dhash:016x}"})
                counts[label] += 1
            except (OSError, ValueError) as exc:
                skipped[type(exc).__name__] += 1
        print(f"shard={shard} accepted={dict(counts)} skipped={dict(skipped)}", flush=True)
        del table
        if all(counts[label] == 750 for label in range(4)):
            break
    if min(counts.get(label, 0) for label in range(4)) < 500:
        raise RuntimeError(f"类别样本不足：{counts}")

    rows = []
    for label in range(4):
        items = [item for item in accepted if item["source_label"] == label]
        rng.shuffle(items)
        n = len(items)
        for i, item in enumerate(items):
            split = "train" if i < int(n * .6) else "validation" if i < int(n * .7) else "calibration" if i < int(n * .8) else "test"
            for policy, text in POLICIES.items():
                threshold = 1 if policy == "pilot-suggestive-v1" else 2
                rows.append({**item, "sample_id": f"{item['group_id']}:{policy}", "text": "", "policy_id": policy,
                             "policy_text": text, "label": int(label >= threshold), "split": split,
                             "attributes": [f"source_L{label + 1}"], "label_provenance": "publisher_level_mapping_not_independently_reviewed"})
    (root / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"dataset": REPO, "revision": REVISION, "seed": 42, "unique_images": len(accepted),
              "source_class_counts": dict(counts), "row_counts": dict(Counter(row["split"] for row in rows)),
              "skipped": dict(skipped), "policies": POLICIES, "license_metadata": "apache-2.0; dataset card restricts to lawful research/moderation",
              "limitations": ["来源标签未经本项目独立人工复核", "不代表日常图片分布，无医疗/艺术豁免标注", "dHash 可过滤部分近重复，不能保证相同人物/来源隔离", "仅对原始等级做规则映射，不能代表任意新规则泛化"]}
    (root / "provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

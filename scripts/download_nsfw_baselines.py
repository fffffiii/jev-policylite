"""下载对照模型并记录不可变版本；第三方权重不随本仓库分发。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "falconsai": ("Falconsai/nsfw_image_detection", "96cb0d0342c7afb80cab76ecc58b265fa44da256"),
    "five_class": ("giacomoarienti/nsfw-classifier", "29f43cab33874e62db8a1973bd0f84b0c69ff057"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/nsfw-baselines"))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = args.output / "revisions.json"
    manifest = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    for name in args.models:
        repo, revision = MODELS[name]
        folder = args.output / name
        snapshot_download(repo, revision=revision, local_dir=folder, token=False,
                          allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md"])
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        manifest[name] = {"repo": repo, "revision": revision, "id2label": config["id2label"]}
        (args.output / "revisions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest[name]), flush=True)


if __name__ == "__main__":
    main()

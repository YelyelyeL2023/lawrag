#!/usr/bin/env python3
"""Download the production embedder + reranker for lawrag.

Run this on the CONNECTED side (it needs internet). It places two ready-to-load
model directories under MODEL_DIR, matching the paths in config/default.yaml:

    MODEL_DIR/bge-m3                 -> embedding.model_path
    MODEL_DIR/bge-reranker-v2-m3     -> rerank.model_path

Those directories are self-contained and are what you carry across the air gap.
The query path never downloads; it loads these local folders.

Run it through the container, so the environment that downloads the weights is
the one that will load them:

    scripts/download_models.sh        (Windows: the .ps1 wrapper beside it)
    docker compose --profile download run --rm download-models

MODEL_DIR is the host directory mounted at /models; set it on the host, not
here. The default below is the in-container mount point, which is what the
compose service passes in.
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))

# repo id -> (local subfolder, patterns to skip). The ignored files are ONNX /
# Keras exports and repo images that FlagEmbedding never loads — skipping them
# saves ~1-2 GB and a good chunk of time. The PyTorch weights (.bin / .pt /
# .safetensors) and tokenizer are kept.
TARGETS = {
    "BAAI/bge-m3": ("bge-m3", ["onnx/*", "*.onnx", "imgs/*", "*.h5", "*.tflite", "*.msgpack"]),
    "BAAI/bge-reranker-v2-m3": ("bge-reranker-v2-m3", ["onnx/*", "*.onnx", "*.h5", "*.tflite", "*.msgpack"]),
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for repo, (sub, ignore) in TARGETS.items():
        dest = MODEL_DIR / sub
        print(f"==> downloading {repo}  ->  {dest}")
        snapshot_download(repo_id=repo, local_dir=str(dest), ignore_patterns=ignore)
        print(f"    done: {dest}")
    print(f"\nAll models downloaded under {MODEL_DIR}")
    print("Next: nothing to install — the lawrag `models` image already carries")
    print("torch + FlagEmbedding. Start with: docker compose up -d")


if __name__ == "__main__":
    main()

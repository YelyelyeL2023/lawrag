#!/usr/bin/env python3
"""Download the production embedder + reranker for lawrag.

Run this on the CONNECTED side (it needs internet). It places two ready-to-load
model directories under MODEL_DIR, matching the paths in config/default.yaml:

    MODEL_DIR/bge-m3                 -> embedding.model_path
    MODEL_DIR/bge-reranker-v2-m3     -> rerank.model_path

Those directories are self-contained and are what you carry across the air gap.
The query path never downloads; it loads these local folders.

    python scripts/download_models.py                 # default MODEL_DIR=D:/models
    set MODEL_DIR=E:/weights & python scripts/download_models.py   # custom (cmd)
    $env:MODEL_DIR="E:/weights"; python scripts/download_models.py  # custom (PS)
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "D:/models"))

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
    print("Next: pip install -e \".[models]\"  (torch + FlagEmbedding), then launch.")


if __name__ == "__main__":
    main()

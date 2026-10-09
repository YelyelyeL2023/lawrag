"""Contrastive fine-tuning of multilingual-e5-small on the train pairs.

Loss: (Cached)MultipleNegativesRankingLoss — for each query the other positives in the
batch are negatives, plus one mined *hard* negative: the best-ranked chunk of
another article under the pretrained model. Hard negatives are only taken from
articles outside validation/test, so no held-out text enters training.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, TrainConfig
from . import retrieval as R

TRAIN_META = "midterm_train.json"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name_or_path: str | Path, cfg: TrainConfig, device: str):
    from sentence_transformers import SentenceTransformer

    # The fused SDPA kernel on MPS has no dropout; eager attention is the path that trains there.
    kwargs = {"attn_implementation": "eager"} if device == "mps" else None
    model = SentenceTransformer(str(name_or_path), device=device, model_kwargs=kwargs)
    model.max_seq_length = cfg.max_seq_length
    return model


def mine_hard_negatives(model, train: pd.DataFrame, chunks: pd.DataFrame, doc_emb: np.ndarray,
                        held_out_articles: set[str], settings: Settings) -> list[str]:
    cfg, ev = settings.train, settings.eval
    allowed = ~chunks["article"].isin(held_out_articles).to_numpy()
    articles = chunks["article"].to_numpy()
    texts = chunks["embed_text"].to_numpy()
    rng = random.Random(cfg.seed)
    pool = np.flatnonzero(allowed)
    out = []
    for start in range(0, len(train), ev.query_block):
        block = train.iloc[start:start + ev.query_block]
        scores = R.encode(model, block["query"].tolist(), cfg.query_prefix, ev.encode_batch_size) @ doc_emb.T
        scores[:, ~allowed] = -np.inf
        top = np.argsort(-scores, axis=1)[:, :cfg.hard_negative_top_k]
        for row, target in zip(top, block["target_article"]):
            neg = next((j for j in row if articles[j] != target), None)
            out.append(texts[neg] if neg is not None else texts[rng.choice(pool)])
    return out


def build_dataset(train: pd.DataFrame, chunks: pd.DataFrame, negatives: list[str] | None, cfg: TrainConfig):
    from datasets import Dataset

    text_of = dict(zip(chunks["chunk_id"], chunks["embed_text"]))
    rows = {"anchor": [cfg.query_prefix + q for q in train["query"]],
            "positive": [cfg.passage_prefix + text_of[c] for c in train["target_chunk"]]}
    if negatives is not None:
        rows["negative"] = [cfg.passage_prefix + t for t in negatives]
    return Dataset.from_dict(rows)


def finetune(model, dataset, settings: Settings, device: str) -> dict:
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    cfg, out = settings.train, settings.paths.model_out
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out.parent / f".{out.name}.work"), num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio, weight_decay=cfg.weight_decay, fp16=device == "cuda",
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=20, save_strategy="no",
        report_to=[], seed=cfg.seed, dataloader_num_workers=0)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset,
                                         loss=losses.CachedMultipleNegativesRankingLoss(
                                             model, mini_batch_size=cfg.mini_batch_size))
    started = time.perf_counter()
    trainer.train()
    model.save(str(out))
    log = [r for r in trainer.state.log_history if "loss" in r]
    meta = {"config": asdict(cfg), "device": device, "train_pairs": len(dataset),
            "train_seconds": round(time.perf_counter() - started), "loss_log": log,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / TRAIN_META).write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def read_meta(model_dir: Path) -> dict | None:
    path = model_dir / TRAIN_META
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

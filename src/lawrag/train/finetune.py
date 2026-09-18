"""Fine-tune the dense embedder on mined pairs.

Contrastive training with ``MultipleNegativesRankingLoss``: for each (query,
positive) the other positives in the batch are negatives, plus one *hard*
negative per query — the highest-ranked wrong article under the base model.
Hard negatives are what teach the model the difference between article 277
and article 278, which in-batch negatives never present.

Sized for an 8 GB card:

* the cached variant of the loss encodes the batch in mini-batches, so the
  effective batch (and the number of in-batch negatives) is not bound by
  memory;
* only the top ``train_layers`` transformer layers are updated. The lower
  layers hold the multilingual lexical knowledge the model came with; the
  top ones are where "what does this sentence ask" is decided. Freezing the
  rest cuts optimiser memory by two thirds and guards against forgetting on
  a corpus this small;
* gradient checkpointing.

The result is a sentence-transformers directory that both embedding backends
load: ``sentence-transformers`` directly, and ``bge-m3`` through FlagEmbedding,
for which the base model's sparse and ColBERT heads are copied alongside.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from .evaluate import encode, encode_corpus, load_model
from .pairs import Pair

log = get_logger(__name__)

_SIDE_FILES = ("colbert_linear.pt", "sparse_linear.pt")


@dataclass(slots=True)
class TrainSettings:
    base: str
    out: str
    epochs: float = 1.0
    batch_size: int = 32
    mini_batch_size: int = 8
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    train_layers: int = 8
    hard_negatives: bool = True
    hard_negative_top_k: int = 20
    max_pairs: int | None = None
    seed: int = 13
    device: str | None = None


def mine_hard_negatives(base: str, pairs: list[Pair], corpus: list[dict], *,
                        cache_dir: Path | None, device: str | None,
                        top_k: int = 20, batch_size: int = 32) -> list[str | None]:
    """For each pair, the text of the best-ranked chunk of a *different* article
    under the base model, or None when nothing wrong ranks in the top k."""
    model = load_model(base, device)
    texts = [c["text"] for c in corpus]
    articles = [c["article"] for c in corpus]
    doc_emb = encode_corpus(model, base, texts, cache_dir, batch_size)
    out: list[str | None] = []
    step = 256
    for start in range(0, len(pairs), step):
        batch = pairs[start:start + step]
        q_emb = encode(model, [p.query for p in batch], batch_size=batch_size)
        scores = q_emb @ doc_emb.T
        top = np.argpartition(-scores, min(top_k, scores.shape[1] - 1), axis=1)[:, :top_k]
        for i, p in enumerate(batch):
            order = top[i][np.argsort(-scores[i, top[i]])]
            neg = next((int(j) for j in order if articles[j] != p.article), None)
            out.append(texts[neg] if neg is not None else None)
    del model
    return out


def _freeze_lower_layers(auto_model, train_layers: int) -> tuple[int, int]:
    """Keep only the top ``train_layers`` encoder layers trainable."""
    layers = None
    for attr in ("encoder",):
        enc = getattr(auto_model, attr, None)
        if enc is not None and hasattr(enc, "layer"):
            layers = enc.layer
    if layers is None:
        log.warning("could not find encoder layers; training every parameter")
        return sum(p.numel() for p in auto_model.parameters()), 0
    n_layers = len(layers)
    keep_from = max(0, n_layers - train_layers)
    for p in auto_model.parameters():
        p.requires_grad = False
    for layer in layers[keep_from:]:
        for p in layer.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in auto_model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in auto_model.parameters() if not p.requires_grad)
    return trainable, frozen


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return ""


def finetune(settings: TrainSettings, pairs: list[Pair], corpus: list[dict], *,
             cache_dir: Path | None = None) -> Path:
    import random

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    rng = random.Random(settings.seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    if settings.max_pairs is not None:
        pairs = pairs[:settings.max_pairs]
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
    chunk_text = {c["id"]: c["text"] for c in corpus}
    pairs = [p for p in pairs if p.pos in chunk_text]
    log.info("training on %d pairs, device %s", len(pairs), device)

    negatives: list[str | None] = [None] * len(pairs)
    if settings.hard_negatives:
        started = time.perf_counter()
        negatives = mine_hard_negatives(settings.base, pairs, corpus, cache_dir=cache_dir,
                                        device=device, top_k=settings.hard_negative_top_k)
        found = sum(1 for n in negatives if n)
        log.info("hard negatives for %d/%d pairs in %ds", found, len(pairs),
                 int(time.perf_counter() - started))
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # A row without a mined negative gets a random other chunk: the loss wants
    # a rectangular dataset, and a random negative is merely uninformative.
    all_texts = list(chunk_text.values())
    rows = {"anchor": [], "positive": [], "negative": []}
    for p, neg in zip(pairs, negatives):
        rows["anchor"].append(p.query)
        rows["positive"].append(chunk_text[p.pos])
        rows["negative"].append(neg or rng.choice(all_texts))
    dataset = Dataset.from_dict(rows)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.base, device=device)
    model.max_seq_length = settings.max_seq_length
    auto_model = model[0].auto_model
    trainable, frozen = _freeze_lower_layers(auto_model, settings.train_layers)
    log.info("trainable parameters: %.1fM (frozen %.1fM)", trainable / 1e6, frozen / 1e6)
    auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # With the embeddings frozen, checkpointed segments see no input that
    # requires grad and would silently skip the backward pass.
    auto_model.enable_input_require_grads()

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=settings.mini_batch_size)
    out = Path(settings.out)
    work = out.parent / f".{out.name}.train"
    args = SentenceTransformerTrainingArguments(
        output_dir=str(work),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        warmup_ratio=0.1,
        fp16=device.startswith("cuda"),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=settings.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=device.startswith("cuda"),
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    started = time.perf_counter()
    trainer.train()
    elapsed = int(time.perf_counter() - started)
    log.info("trained in %ds", elapsed)

    for p in auto_model.parameters():
        p.requires_grad = True
    auto_model.gradient_checkpointing_disable()
    if out.exists():
        shutil.rmtree(out)
    model.save(str(out))
    for name in _SIDE_FILES:
        src = Path(settings.base) / name
        if src.exists():
            shutil.copy2(src, out / name)
    meta = {
        "base": settings.base,
        "pairs": len(pairs),
        "kinds": {k: sum(1 for p in pairs if p.kind == k) for k in sorted({p.kind for p in pairs})},
        "settings": asdict(settings),
        "trainable_parameters": trainable,
        "train_seconds": elapsed,
        "git_commit": _git_commit(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "lawrag_train.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    log.info("saved fine-tuned model to %s", out)
    return out

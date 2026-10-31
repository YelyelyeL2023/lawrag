"""An LLM as the second-stage reranker, adapted with LoRA.

The bi-encoder ranks 43,857 chunks with one matrix product, but sees query
and article separately. A reranker reads them *together* and decides on the
top 20 only. Here the reranker is a decoder LLM, **Qwen3-0.6B**, asked the
question the Qwen3-Reranker models are trained on — "does this Document
answer the Query? yes / no" — and the relevance score is the log-odds of the
next token: ``logit("yes") - logit("no")``.

**LoRA.** The 0.6B base weights are frozen (bf16). Trainable low-rank
adapters ``W + (alpha / r) * B @ A`` are added to every linear layer of every
block (attention q/k/v/o and MLP gate/up/down). Only A and B — about 1.7% of
the parameters — get gradients and optimiser state (kept in fp32).

**QLoRA** (``load_in_4bit=True``, ``--qlora``) additionally stores the frozen
base in 4-bit NF4 with double quantisation. It is off by default: on the
reference machine (RTX 5070 Laptop, driver 610.62, bitsandbytes 0.50.2,
CUDA 13.0) the 4-bit inference kernel hit an illegal memory access and the
driver failed to recover (bugcheck 0x116, VIDEO_TDR_FAILURE). At 0.6B the
model fits in bf16 with room to spare, so quantising buys nothing here; it
is what makes 7B+ models trainable on the same card.

**Training signal.** One example is a *group*: a training query, the chunk
of its own article and 7 chunks of other articles that the fine-tuned
bi-encoder ranks highest for it (4 hardest + 3 sampled from the rest of its
top 30) — the exact confusions the reranker will be asked to resolve. The
loss is listwise softmax cross-entropy over the group's 8 scores, i.e. the
same InfoNCE objective the bi-encoder is trained with, applied to the LLM's
yes/no log-odds.

Validation and test articles are excluded from the groups entirely — as
queries, as positives and as negatives.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..logging_setup import get_logger
from .pairs import Pair

log = get_logger(__name__)

SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the '
          'Instruct provided. Note that the answer can only be "yes" or "no".')
INSTRUCTION = ("Given a question, a title or a sentence in Russian or Kazakh about the law of "
               "Kazakhstan, judge whether the Document is the statutory provision it asks "
               "about, names or refers to")
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
# Qwen3 is a hybrid thinking model: an empty think block asks for the answer
# token straight away.
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(slots=True)
class RerankSettings:
    base: str
    out: str
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = TARGET_MODULES
    # QLoRA quantisation of the frozen base (off: see the module docstring)
    load_in_4bit: bool = False
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    # optimisation
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    epochs: int = 1
    group_size: int = 8                 # 1 positive + 7 negatives
    grad_accum: int = 4                 # groups per optimiser step
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    # data
    n_queries: int = 4000
    hard_negatives: int = 4
    negative_pool: int = 30
    max_len: int = 512
    max_query_tokens: int = 128
    seed: int = 13
    device: str | None = None
    loss: str = field(default="listwise softmax cross-entropy over the group's yes/no log-odds")


# ------------------------------------------------------------- prompting --

def _content(query: str) -> str:
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: "


def build_ids(tok, query: str, doc: str, max_len: int = 512,
              max_query_tokens: int = 128) -> list[int]:
    """Token ids of one (query, document) prompt. The query is cut to
    ``max_query_tokens`` and the document fills what is left of ``max_len``,
    so the prompt's frame and the answer slot are never truncated."""
    pre = tok.encode(PREFIX, add_special_tokens=False)
    suf = tok.encode(SUFFIX, add_special_tokens=False)
    q_ids = tok.encode(query, add_special_tokens=False)[:max_query_tokens]
    head = tok.encode(_content(tok.decode(q_ids)), add_special_tokens=False)
    room = max(16, max_len - len(pre) - len(head) - len(suf))
    d_ids = tok.encode(doc, add_special_tokens=False)[:room]
    return pre + head + d_ids + suf


def _pad_left(batch: list[list[int]], pad_id: int):
    import torch

    width = max(len(x) for x in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, x in enumerate(batch):
        ids[i, width - len(x):] = torch.tensor(x)
        mask[i, width - len(x):] = 1
    return ids, mask


# --------------------------------------------------------------- loading --

def load_reranker(base: str, adapter: str | None = None, *, load_in_4bit: bool = False,
                  device: str | None = None, quant_type: str = "nf4",
                  double_quant: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    kwargs: dict = {"dtype": torch.bfloat16}
    if load_in_4bit and device.startswith("cuda"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=double_quant, bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def _yes_no(tok) -> tuple[int, int]:
    return tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")


def _logodds(model, ids, mask, yes: int, no: int):
    out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
    last = out.logits[:, -1, :].float()
    return last[:, yes] - last[:, no]


def score_pairs(model, tok, pairs: Sequence[tuple[str, str]], *, batch_size: int = 16,
                max_len: int = 512, max_query_tokens: int = 128) -> np.ndarray:
    """yes/no log-odds per (query, document); length-sorted batches."""
    import torch

    yes, no = _yes_no(tok)
    device = next(model.parameters()).device
    encoded = [build_ids(tok, q, d, max_len, max_query_tokens) for q, d in pairs]
    order = np.argsort([-len(x) for x in encoded], kind="stable")
    out = np.zeros(len(pairs), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            ids, mask = _pad_left([encoded[i] for i in idx], tok.pad_token_id)
            s = _logodds(model, ids.to(device), mask.to(device), yes, no)
            out[idx] = s.cpu().numpy()
            if (start // batch_size) % 200 == 0:
                log.info("reranker scored %d/%d", min(start + batch_size, len(order)), len(order))
    return out


def rerank(queries: list[dict], candidates: list[list[int]], corpus: list[dict], model, tok,
           **kw) -> tuple[list[int], list[str], np.ndarray]:
    """Re-order each query's candidate rows by the LLM's score. Returns the
    article-level rank of the right article (its old rank when it was not
    among the candidates), the new top-1 article, and the raw scores."""
    flat = [(q["query"], corpus[j]["text"]) for q, cand in zip(queries, candidates) for j in cand]
    scores = score_pairs(model, tok, flat, **kw)
    ranks, top1, at = [], [], 0
    for q, cand in zip(queries, candidates):
        s = scores[at:at + len(cand)]
        at += len(cand)
        order = [cand[i] for i in np.argsort(-s, kind="stable")]
        hit = next((r for r, j in enumerate(order, 1)
                    if corpus[j]["article"] == q["article"]
                    and corpus[j]["lang"] == q["pos_lang"]), None)
        ranks.append(hit if hit is not None else q["rank"])
        top1.append(corpus[order[0]]["article"])
    return ranks, top1, scores


# --------------------------------------------------------------- training --

def make_groups(pairs: list[Pair], tops: list[list[int]], corpus: list[dict],
                settings: RerankSettings, held_out: frozenset[str]) -> list[dict]:
    """(query, positive text, negative texts) per training query."""
    rng = random.Random(settings.seed)
    chunk = {c["id"]: c for c in corpus}
    groups = []
    for p, top in zip(pairs, tops):
        pos = chunk.get(p.pos)
        if pos is None:
            continue
        wrong = [j for j in top[:settings.negative_pool]
                 if corpus[j]["article"] != p.article
                 and corpus[j]["article"] not in held_out
                 and corpus[j]["text"] != pos["text"]]
        n_neg = settings.group_size - 1
        if len(wrong) < n_neg:
            continue
        hard = wrong[:settings.hard_negatives]
        rest = rng.sample(wrong[settings.hard_negatives:], n_neg - len(hard))
        groups.append({"query": p.query, "kind": p.kind, "positive": pos["text"],
                       "negatives": [corpus[j]["text"] for j in hard + rest]})
    return groups


def count_parameters(model) -> tuple[int, int]:
    """(trainable, total) parameters; 4-bit weights count at their logical size."""
    trainable = total = 0
    for p in model.parameters():
        n = p.numel()
        # bitsandbytes packs two 4-bit values per byte.
        if p.__class__.__name__ == "Params4bit":
            n *= 2
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def train_reranker(settings: RerankSettings, groups: list[dict]) -> Path:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_scheduler

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    model, tok = load_reranker(settings.base, load_in_4bit=settings.load_in_4bit,
                               device=settings.device, quant_type=settings.quant_type,
                               double_quant=settings.double_quant)
    model.train()
    if settings.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=settings.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    elif settings.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # The frozen embeddings would leave checkpointed segments without an
        # input that requires grad, and the backward pass would skip them.
        model.enable_input_require_grads()
    lora = LoraConfig(r=settings.lora_r, lora_alpha=settings.lora_alpha,
                      lora_dropout=settings.lora_dropout,
                      target_modules=list(settings.target_modules),
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable, total = count_parameters(model)
    log.info("trainable %.2fM of %.1fM parameters (%.2f%%)", trainable / 1e6, total / 1e6,
             100 * trainable / total)

    yes, no = _yes_no(tok)
    device = next(model.parameters()).device
    rng = random.Random(settings.seed)
    order = [g for _ in range(settings.epochs) for g in rng.sample(groups, len(groups))]
    steps = max(1, len(order) // settings.grad_accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    sched = get_scheduler(settings.lr_scheduler, opt,
                          num_warmup_steps=int(settings.warmup_ratio * steps),
                          num_training_steps=steps)
    history: list[dict] = []
    running, n_run, correct = 0.0, 0, 0
    started = time.perf_counter()
    target = torch.zeros(1, dtype=torch.long, device=device)   # the positive is first
    for i, g in enumerate(order):
        docs = [g["positive"], *g["negatives"]]
        ids, mask = _pad_left([build_ids(tok, g["query"], d, settings.max_len,
                                         settings.max_query_tokens) for d in docs],
                              tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            s = _logodds(model, ids.to(device), mask.to(device), yes, no)
        loss = F.cross_entropy(s.unsqueeze(0), target)
        (loss / settings.grad_accum).backward()
        running += loss.item()
        n_run += 1
        correct += int(torch.argmax(s).item() == 0)
        if (i + 1) % settings.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, settings.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = (i + 1) // settings.grad_accum
            if step % 10 == 0 or step == steps:
                history.append({"step": step, "epoch": round((i + 1) / len(groups), 4),
                                "loss": running / n_run, "group_acc": correct / n_run,
                                "learning_rate": sched.get_last_lr()[0]})
                if step % 50 == 0:
                    log.info("step %d/%d loss %.4f acc@1 %.3f (%.0fs)", step, steps,
                             running / n_run, correct / n_run, time.perf_counter() - started)
                running, n_run, correct = 0.0, 0, 0
    elapsed = int(time.perf_counter() - started)

    out = Path(settings.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    meta = {"settings": asdict(settings), "groups": len(groups),
            "kinds": {k: sum(1 for g in groups if g["kind"] == k)
                      for k in sorted({g["kind"] for g in groups})},
            "optimizer_steps": steps, "trainable_parameters": trainable,
            "total_parameters": total, "train_seconds": elapsed,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if device.type == "cuda" else None,
            "log_history": history, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "lawrag_rerank.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info("saved adapter to %s (%ds)", out, elapsed)
    return out

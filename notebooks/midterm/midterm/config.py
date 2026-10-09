"""All settings of the midterm experiment in one place.

Paths are resolved from the repository root, so nothing depends on where the
notebook or the script is launched from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_MARKER = Path("src") / "lawrag" / "__init__.py"


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / REPO_MARKER).exists():
            return candidate
    raise FileNotFoundError("lawrag repository root not found: run inside a clone of the repository")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    seed: int = 13
    # Same split as Assignment 2: 10 article-grouped folds, 1 test, 1 validation, 8 train.
    n_folds: int = 10
    test_fold: int = 0
    val_fold: int = 1


@dataclass(frozen=True, slots=True)
class TfidfConfig:
    analyzer: str = "word"
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_features: int = 400_000


@dataclass(frozen=True, slots=True)
class TrainConfig:
    base_model: str = "intfloat/multilingual-e5-small"
    # e5 was pre-trained with these role prefixes; dropping them costs several points of recall.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    max_seq_length: int = 256
    epochs: int = 1
    batch_size: int = 64
    # CachedMNRL encodes the batch in chunks of this size: the 64-way contrast is kept while peak
    # memory is that of 16 triplets. Plain MNRL at batch 64 swapped a 32 GB M1 Pro to a standstill.
    mini_batch_size: int = 16
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    hard_negatives: bool = True
    hard_negative_top_k: int = 20
    seed: int = 13


@dataclass(frozen=True, slots=True)
class EvalConfig:
    top_k: int = 10
    encode_batch_size: int = 128
    query_block: int = 512


@dataclass(frozen=True, slots=True)
class Paths:
    root: Path

    @property
    def adilet(self) -> Path:
        return self.root / "data" / "adilet"

    @property
    def work(self) -> Path:
        return self.root / "data" / "train" / "midterm"

    @property
    def corpus(self) -> Path:
        return self.work / "corpus.jsonl"

    @property
    def pairs(self) -> Path:
        return self.work / "pairs_split.jsonl"

    @property
    def embeddings(self) -> Path:
        return self.work / "emb"

    @property
    def model_out(self) -> Path:
        return self.root / "models" / "e5-small-lawrag"

    @property
    def outputs(self) -> Path:
        return self.root / "notebooks" / "midterm" / "outputs"


@dataclass(frozen=True, slots=True)
class Settings:
    paths: Paths
    split: SplitConfig = field(default_factory=SplitConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def default_settings(start: Path | None = None) -> Settings:
    return Settings(paths=Paths(find_repo_root(start)))

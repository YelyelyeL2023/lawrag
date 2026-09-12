from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("LAWRAG_CONFIG", Path(__file__).resolve().parents[2] / "config" / "default.yaml")
)


def _load_dotenv(path: Path | None = None) -> None:
    """Read a .env file into the environment, if one exists.

    Deliberately hand-rolled rather than pulling in python-dotenv: this needs to
    install cleanly on an air-gapped machine from a wheel bundle, and one fewer
    dependency is one fewer thing to carry across. Real environment variables
    always win, so a shell export overrides the file rather than the other way
    round — which is the behaviour anyone debugging a connection string
    expects.
    """
    candidates = [path] if path else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


@dataclass
class DBConfig:
    dsn: str = "postgresql://postgres:2006@localhost:5432/postgres"
    statement_timeout_ms: int = 30_000


@dataclass
class EmbeddingConfig:
    # "bge-m3" for production; "hash" is a deterministic dependency-free stand-in
    # so the whole pipeline runs (and is testable) without downloading weights.
    backend: str = "hash"
    model_path: str = "/opt/models/bge-m3"
    dim: int = 1024
    batch_size: int = 8
    max_length: int = 4096
    device: str = "cuda"
    normalize: bool = True


@dataclass
class RerankConfig:
    backend: str = "none"          # none | bge-reranker | qwen3-reranker
    model_path: str = "/opt/models/bge-reranker-v2-m3"
    device: str = "cpu"            # CPU keeps the 8 GB card free for the generator
    top_k_in: int = 40
    top_k_out: int = 8


@dataclass
class LLMConfig:
    backend: str = "echo"          # echo | llamacpp
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3.5-9b-q4_k_m"
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 1200
    timeout_s: int = 180
    # A second, smaller model used for the offline batch job that writes one
    # context sentence per chunk. Runs overnight; quality bar is low.
    context_model: str = "qwen3.5-4b-q4_k_m"


@dataclass
class ChunkConfig:
    # Structural boundaries always win; these only decide *at which level* to cut.
    max_article_tokens: int = 1000
    min_split_tokens: int = 120
    hard_max_tokens: int = 3000     # above this, even a point gets split at sentence bounds
    generate_context_sentences: bool = False


@dataclass
class RetrievalConfig:
    dense_k: int = 50
    sparse_k: int = 50
    rrf_k: int = 60
    final_k: int = 8
    hydrate_refs: bool = True
    hydrate_definitions: bool = True
    max_hydration_hops: int = 1
    max_packet_tokens: int = 12_000
    cross_language: bool = True
    # Absolute relevance floor. Rank fusion always returns its best candidates
    # even when none are relevant, so an out-of-corpus question would otherwise
    # be answered from whatever happened to rank first.
    #
    # min_dense_score is a cosine on the embedding model's own scale and MUST be
    # re-tuned when embedding.backend changes — the hash placeholder and BGE-M3
    # do not produce comparable cosines.
    # min_lexical_score is a ts_rank_cd value; it depends on the corpus and the
    # text-search configuration, not on the embedding model.
    # min_trigram_score is a word_similarity value on 0..1.
    min_dense_score: float = 0.30
    min_lexical_score: float = 0.60
    min_trigram_score: float = 0.45


@dataclass
class GenerationConfig:
    abstain_when_no_hits: bool = True
    min_rerank_score: float = -8.0
    answer_lang_follows_query: bool = True
    # The model never composes legal Kazakh; it wraps verbatim corpus text.
    quote_verbatim_only: bool = True


@dataclass
class Config:
    db: DBConfig = field(default_factory=DBConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data_dir: str = "data"
    log_level: str = "INFO"
    # Filled in by load_config, not read from YAML: which settings the
    # environment overrode. The console reports it, because "the YAML says cuda
    # but it is running on the CPU" is otherwise a silent contradiction.
    env_overrides: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: Any, override: dict[str, Any]) -> Any:
    for key, value in override.items():
        if not hasattr(base, key):
            raise ValueError(f"unknown config key: {key}")
        current = getattr(base, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(base, key, value)
    return base


def _coerce(raw: str, current: Any) -> Any:
    """Environment values arrive as strings; the existing value names the type."""
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply_env(cfg: Config) -> list[str]:
    """Override any scalar setting from `LAWRAG_<SECTION>_<FIELD>`.

    Containers are configured by environment, not by editing a file baked into
    an image: the same image has to run against `db:5432` in compose and a
    remote Postgres in production without a second copy of default.yaml. The
    YAML stays the place where a setting is *reasoned about*; the environment is
    where a deployment says where things are.

    Section names match the dataclass attributes, so `embedding.model_path` is
    `LAWRAG_EMBEDDING_MODEL_PATH` and `db.dsn` is `LAWRAG_DB_DSN`. Top-level
    scalars drop the section: `LAWRAG_LOG_LEVEL`, `LAWRAG_DATA_DIR`.

    Returns the names that were applied. A setting silently overridden by an
    inherited environment variable is a long debugging session, so load_config
    keeps the list and the console shows it.
    """
    applied: list[str] = []
    for f in fields(cfg):
        if f.name == "env_overrides":
            continue
        section = getattr(cfg, f.name)
        if is_dataclass(section):
            for sub in fields(section):
                name = f"LAWRAG_{f.name.upper()}_{sub.name.upper()}"
                if (raw := os.environ.get(name)) is not None:
                    setattr(section, sub.name, _coerce(raw, getattr(section, sub.name)))
                    applied.append(name)
        else:
            name = f"LAWRAG_{f.name.upper()}"
            if (raw := os.environ.get(name)) is not None:
                setattr(cfg, f.name, _coerce(raw, section))
                applied.append(name)
    return applied


def load_config(path: str | Path | None = None) -> Config:
    """Config precedence, lowest to highest: defaults, YAML, .env, environment."""
    _load_dotenv()
    cfg = Config()
    p = Path(path) if path else Path(os.environ.get("LAWRAG_CONFIG") or DEFAULT_CONFIG_PATH)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        _merge(cfg, raw)
    # Alias for the pre-container spelling of LAWRAG_DB_DSN. Applied first, so
    # that a deployment setting both gets the explicit name.
    if dsn := os.environ.get("LAWRAG_DSN"):
        cfg.db.dsn = dsn
    cfg.env_overrides = _apply_env(cfg)
    return cfg

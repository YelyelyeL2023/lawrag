"""lawrag command line."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .logging_setup import setup_logging
from .models import Lang

def _utf8_stdio() -> None:
    """Make the CLI's output UTF-8 regardless of the console's code page.

    Everything this tool prints — headings, statutory text, the console's
    URL line — is Cyrillic. On a Windows terminal with a legacy code page
    (cp1252, cp866) Python's stdout cannot encode it and `lawrag serve`
    crashed before binding the port; `lawrag search` died on its first hit.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # closed or exotic stream
                pass


_utf8_stdio()

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
console = Console()

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config YAML.")


def _cfg(path: Optional[str]):
    cfg = load_config(path)
    setup_logging(cfg.log_level)
    return cfg


def _as_of(value: Optional[str]) -> Optional[date]:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


# --------------------------------------------------------------------- db --
@app.command("init-db")
def init_db(config: Optional[str] = ConfigOpt,
            reset: bool = typer.Option(False, help="Drop all lawrag tables first.")):
    """Create (or reset) the schema."""
    from .index.db import Database

    db = Database(_cfg(config))
    if reset:
        db.reset()
    db.apply_schema()
    console.print("[green]schema ready[/green]")


@app.command()
def stats(config: Optional[str] = ConfigOpt):
    """Row counts and corpus health."""
    from .index.db import Database

    db = Database(_cfg(config))
    table = Table(title="lawrag corpus")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, count in db.stats().items():
        table.add_row(name, f"{count:,}")
    console.print(table)

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT dst_act_code, n FROM xref_unresolved LIMIT 15")
        rows = cur.fetchall()
    if rows:
        missing = Table(title="unresolved references (acts not in the corpus)")
        missing.add_column("act")
        missing.add_column("references", justify="right")
        for row in rows:
            missing.add_row(str(row["dst_act_code"]), f"{row['n']:,}")
        console.print(missing)


# ----------------------------------------------------------- connected side --
@app.command()
def fetch(out: Path = typer.Argument(..., help="Directory to write the snapshot into."),
          codes: str = typer.Option("", help="Comma-separated act codes; default all known."),
          langs: str = typer.Option("ru,kk"),
          delay: float = typer.Option(2.0, help="Seconds between requests."),
          as_of: Optional[str] = typer.Option(
              None, "--as-of",
              help="Fetch the redaction in force on this date (YYYY-MM-DD) instead of the "
                   "current one. Use a separate output directory per date, and ingest each; "
                   "a page whose footnotes cite a later amendment is rejected."),
          ca_bundle: Optional[str] = typer.Option(
              None, "--ca-bundle",
              help="Path to a PEM CA bundle to verify TLS against. Use when adilet's "
                   "certificate chains to a root your OS trusts but certifi does not "
                   "ship. Defaults to $LAWRAG_CA_BUNDLE."),
          insecure: bool = typer.Option(
              False, "--insecure",
              help="Skip TLS certificate verification (last resort, trusted network "
                   "only). Defaults to $LAWRAG_INSECURE."),
          config: Optional[str] = ConfigOpt):
    """Download a corpus snapshot from adilet. Runs on the CONNECTED side only.

    TLS note: httpx verifies against certifi, so a "certificate verify failed /
    unable to get local issuer" error usually means the signing root is trusted
    by the machine but missing from certifi. The fix is to hand the root in
    explicitly with --ca-bundle (or LAWRAG_CA_BUNDLE), e.g.

        lawrag fetch data/snapshots/2026-09 --ca-bundle /path/root.pem
    """
    from .ingest.fetch_adilet import build_plan, fetch as do_fetch

    _cfg(config)
    if ca_bundle is None:
        ca_bundle = os.environ.get("LAWRAG_CA_BUNDLE") or None
    if not insecure:
        insecure = os.environ.get("LAWRAG_INSECURE", "").strip().lower() in {"1", "true", "yes", "on"}
    plan = build_plan([c for c in codes.split(",") if c] or None,
                      tuple(l for l in langs.split(",") if l),
                      as_of=_as_of(as_of))
    console.print(f"fetching {len(plan)} documents")
    do_fetch(plan, out, delay_s=delay, ca_bundle=ca_bundle, insecure=insecure)


@app.command()
def pack(src: Path, out: Path, source: str = typer.Option("adilet.zan.kz"),
         note: str = typer.Option(""), config: Optional[str] = ConfigOpt):
    """Assemble a signed-manifest bundle ready to carry across the air gap."""
    from .ingest.bundle import pack as do_pack

    _cfg(config)
    manifest = do_pack(src, out, source=source, note=note)
    console.print(f"[green]release {manifest.release_id}[/green] "
                  f"({len(manifest.entries)} files)")


@app.command("verify-bundle")
def verify_bundle_cmd(path: Path,
                      require_signature: bool = typer.Option(False),
                      config: Optional[str] = ConfigOpt):
    """Check a bundle's manifest before importing it."""
    from .ingest.bundle import BundleVerificationError, verify_bundle

    _cfg(config)
    try:
        manifest = verify_bundle(path, require_signature=require_signature)
    except BundleVerificationError as exc:
        # A failed bundle is an expected operational outcome, not a crash. It
        # gets a readable message and a non-zero exit so a wrapper script stops.
        console.print(f"[red]BUNDLE REJECTED[/red]\n{exc}")
        raise typer.Exit(2) from None
    console.print(f"[green]OK[/green] release {manifest.release_id}, "
                  f"{len(manifest.entries)} files")


# ------------------------------------------------------------------ ingest --
@app.command()
def ingest(path: Path,
           reset: bool = typer.Option(False, help="Drop and recreate the schema first."),
           verify: bool = typer.Option(True, help="Verify the bundle manifest."),
           build_index: bool = typer.Option(True, "--build-index/--no-build-index"),
           config: Optional[str] = ConfigOpt):
    """Parse, chunk, extract and index a corpus directory."""
    from .ingest.bundle import BundleVerificationError
    from .ingest.pipeline import ingest_directory

    cfg = _cfg(config)
    try:
        totals = ingest_directory(cfg, path, verify=verify, reset=reset,
                                  build_index=build_index)
    except BundleVerificationError as exc:
        console.print(f"[red]BUNDLE REJECTED — nothing was imported[/red]\n{exc}")
        raise typer.Exit(2) from None
    console.print(totals)


@app.command("context")
def context_cmd(cache: Path = typer.Option(Path("data/context_cache.json")),
                limit: int = typer.Option(0, help="Process at most N chunks (0 = all)."),
                config: Optional[str] = ConfigOpt):
    """Generate one context sentence per chunk (offline batch; needs a real LLM)."""
    from .chunking.context import generate_context_sentences
    from .index.db import Database
    from .index.embed import build_embedder
    from .generate.llm import build_llm
    from .models import Chunk

    cfg = _cfg(config)
    if cfg.llm.backend in ("echo", "none", ""):
        console.print("[red]llm.backend is 'echo' — nothing to generate with.[/red]")
        raise typer.Exit(1)

    db = Database(cfg)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, node_id, article_node_id, act_code, lang, valid_from, valid_to,"
            "       display_text, embed_text, hierarchy_path, heading, lead_in,"
            "       repealed, release_id"
            "  FROM chunk ORDER BY id" + (f" LIMIT {int(limit)}" if limit else "")
        )
        rows = cur.fetchall()

    chunks = [Chunk(id=r["id"], node_id=r["node_id"],
                    article_node_id=r["article_node_id"], act_code=r["act_code"],
                    lang=Lang(r["lang"]), valid_from=r["valid_from"],
                    valid_to=r["valid_to"], display_text=r["display_text"],
                    embed_text=r["embed_text"], hierarchy_path=r["hierarchy_path"],
                    heading=r["heading"], lead_in=r["lead_in"],
                    repealed=r["repealed"], release_id=r["release_id"] or "")
              for r in rows]

    llm = build_llm(cfg.llm)
    generate_context_sentences(chunks, llm, cache, model=cfg.llm.context_model)

    embedder = build_embedder(cfg.embedding)
    from .index.indexer import Indexer
    indexer = Indexer(db, embedder)
    with db.connect() as conn, conn.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                "UPDATE chunk SET context_sentence = %s, embed_text = %s WHERE id = %s",
                (chunk.context_sentence, chunk.embed_text, chunk.id),
            )
    indexer._embed_chunks(chunks)  # noqa: SLF001 — re-embedding is the point
    console.print(f"[green]context sentences applied to {len(chunks)} chunks[/green]")


# --------------------------------------------------------------- retrieval --
@app.command()
def search(question: str,
           as_of: Optional[str] = typer.Option(None, "--as-of", help="YYYY-MM-DD"),
           lang: Optional[str] = typer.Option(None, help="ru | kk"),
           show_packet: bool = typer.Option(False, "--packet"),
           config: Optional[str] = ConfigOpt):
    """Retrieval only — no model, no answer. The fastest way to debug recall."""
    from .retrieve.engine import RetrievalEngine

    cfg = _cfg(config)
    engine = RetrievalEngine(cfg)
    result = engine.retrieve(question, as_of=_as_of(as_of),
                             lang=Lang(lang) if lang else None)

    console.print(f"[bold]as of[/bold] {result.analysis.as_of}  "
                  f"[bold]lang[/bold] {result.analysis.lang.value}  "
                  f"[bold]acts[/bold] {result.analysis.act_hints or '—'}")
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("chunk")
    table.add_column("score", justify="right")
    table.add_column("via")
    table.add_column("in force")
    for i, hit in enumerate(result.hits, 1):
        table.add_row(str(i), hit.chunk_id,
                      f"{hit.rerank_score if hit.rerank_score is not None else hit.fused_score:.4f}",
                      ",".join(hit.provenance),
                      "yes" if hit.is_current else f"→{hit.valid_to}")
    console.print(table)
    if show_packet:
        console.print(result.packet.text)
    console.print(f"[dim]{result.timings_ms}[/dim]")


@app.command()
def ask(question: str,
        as_of: Optional[str] = typer.Option(None, "--as-of", help="YYYY-MM-DD"),
        lang: Optional[str] = typer.Option(None, help="ru | kk"),
        as_json: bool = typer.Option(False, "--json"),
        config: Optional[str] = ConfigOpt):
    """Ask a question and get a cited, verified answer."""
    from .generate.answer import AnswerService

    cfg = _cfg(config)
    service = AnswerService(cfg)
    answer = service.ask(question, as_of=_as_of(as_of),
                         lang=Lang(lang) if lang else None)

    if as_json:
        console.print_json(json.dumps(answer.to_dict(), ensure_ascii=False, default=str))
        return

    console.print(f"\n[bold]{answer.text}[/bold]\n")
    if answer.citations:
        table = Table(title="Citations")
        table.add_column("act")
        table.add_column("article")
        table.add_column("point")
        table.add_column("quote", max_width=70)
        for c in answer.citations:
            table.add_row(str(c.get("act", "")), str(c.get("article", "")),
                          str(c.get("point", "") or ""), str(c.get("quote", ""))[:200])
        console.print(table)
    if answer.sources:
        console.print("[dim]sources: " +
                      ", ".join(s["node_id"] for s in answer.sources) + "[/dim]")
    flags = []
    if answer.abstained:
        flags.append(f"abstained ({answer.abstain_reason})")
    if not answer.verified:
        flags.append("verification FAILED")
    if answer.uses_superseded_text:
        flags.append("includes superseded redactions")
    if answer.escalate:
        flags.append("escalate to counsel")
    if flags:
        console.print(f"[yellow]{' | '.join(flags)}[/yellow]")
    for problem in answer.verification_problems:
        console.print(f"[red]  - {problem}[/red]")
    console.print(f"[dim]{answer.verification} · as of {answer.as_of} · "
                  f"{answer.timings_ms.get('total', 0)} ms[/dim]")
    console.print(f"[dim italic]{answer.disclaimer}[/dim italic]")


@app.command("eval")
def eval_cmd(path: Path = typer.Argument(Path("src/lawrag/evalkit/questions.sample.yaml")),
             top_k: int = typer.Option(8),
             config: Optional[str] = ConfigOpt):
    """Score retrieval against a golden question set."""
    from .evalkit.harness import run_eval, print_report

    cfg = _cfg(config)
    report = run_eval(cfg, path, top_k=top_k)
    print_report(report, console)


@app.command()
def calibrate(path: Path = typer.Argument(Path("src/lawrag/evalkit/questions.sample.yaml")),
              config: Optional[str] = ConfigOpt):
    """Measure where the relevance floor should sit for this model and corpus.

    Runs the question set with the floor off and proposes min_dense_score,
    min_lexical_score and min_trigram_score in the gap between the weakest
    answerable question and the strongest out-of-corpus one. Run it after
    switching embedding.backend, and after a large corpus change.
    """
    from .evalkit.calibrate import calibrate as run, print_calibration

    cfg = _cfg(config)
    print_calibration(run(cfg, path), console)


@app.command()
def inspect(path: Path,
            code: str = typer.Option(..., help="Act code, e.g. koap."),
            lang: Optional[str] = typer.Option(None, help="ru | kk; auto-detected if omitted."),
            valid_from: str = typer.Option(..., "--valid-from", help="Redaction date, YYYY-MM-DD."),
            show: int = typer.Option(8, help="How many articles to print."),
            config: Optional[str] = ConfigOpt):
    """Parse ONE document and report what came out — no database, no indexing.

    This is the first thing to run against a new source format. A container
    selector that misses, or a Kazakh file parsed with the Russian heading
    dialect, both show up here as an article count of zero or a handful, long
    before anything reaches the index.
    """
    from .ingest.parser import parse_act_source, parse_adilet
    from .ingest import refs as R

    _cfg(config)
    raw = path.read_text(encoding="utf-8", errors="replace")
    when = datetime.strptime(valid_from, "%Y-%m-%d").date()

    if raw.lstrip().startswith("---"):
        act, root, doc = parse_act_source(raw)
    else:
        act, root, doc = parse_adilet(raw, {
            "code": code, "kind": "code", "lang": lang,
            "title": code, "valid_from": when,
        })

    articles = list(root.articles())
    console.print(f"[bold]{path.name}[/bold]  act={doc.act_code}  "
                  f"lang={doc.lang.value}  valid_from={doc.valid_from}")
    console.print(f"[bold]{len(articles)}[/bold] articles parsed")

    if not articles:
        console.print("[red]ZERO articles.[/red] Either the container selector in "
                      "adilet_html_to_lines() missed the document body, or the "
                      "heading dialect does not match the language "
                      "(Russian 'Статья 278' vs Kazakh '278-бап').")
        raise typer.Exit(1)

    empty = [a.id for a in articles if not (a.texts[doc.lang].body or "").strip()]
    n_points = sum(1 for a in articles for c in a.children if c.kind.value == "point")
    n_refs = sum(len(R.extract_refs(a.id, doc.act_code, a.texts[doc.lang].body, doc.lang))
                 for a in articles if a.texts.get(doc.lang))

    table = Table(title="first articles")
    table.add_column("node id")
    table.add_column("heading", max_width=52)
    table.add_column("chars", justify="right")
    table.add_column("points", justify="right")
    for a in articles[:show]:
        text = a.texts[doc.lang]
        table.add_row(a.id, text.heading or "[red]— none —[/red]",
                      str(len(text.body)),
                      str(sum(1 for c in a.children if c.kind.value == "point")))
    console.print(table)

    console.print(f"points: {n_points}   cross-references: {n_refs}   "
                  f"empty bodies: {len(empty)}")
    if empty:
        console.print(f"[yellow]{len(empty)} articles parsed with no body — "
                      f"block tags are probably being collapsed: "
                      f"{', '.join(empty[:6])}[/yellow]")
    if n_refs == 0:
        console.print("[yellow]No cross-references found. For a real code this "
                      "means the citation graph will be empty and hydration will "
                      "do nothing.[/yellow]")


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000),
          config: Optional[str] = ConfigOpt):
    """Run the HTTP API."""
    import uvicorn

    from .api.security import warn_if_exposed

    _cfg(config)
    # uvicorn imports lawrag.api.main by name, and that module builds its own
    # config from the environment. Without this, --config was validated here
    # and then silently ignored by the server it started.
    if config:
        os.environ["LAWRAG_CONFIG"] = str(Path(config).resolve())
    if warning := warn_if_exposed(host):
        console.print(f"[yellow]warning[/yellow] {warning}")
    console.print(f"[green]API[/green]     http://{host}:{port}/docs")
    console.print(f"[green]Консоль[/green] http://{host}:{port}/")
    uvicorn.run("lawrag.api.main:app", host=host, port=port, reload=False)


# ---------------------------------------------------------------- train --
@app.command("train-pairs")
def train_pairs_cmd(path: Path = typer.Argument(..., help="Corpus directory (e.g. data/adilet)."),
                    out: Path = typer.Option(Path("data/train"), help="Where to write the pairs."),
                    dev_fraction: float = typer.Option(0.05, help="Share of articles held out."),
                    doc2query: bool = typer.Option(False, help="Ask the local LLM for questions per chunk."),
                    query_log: bool = typer.Option(True, help="Add verified questions from query_log."),
                    config: Optional[str] = ConfigOpt):
    """Mine training pairs from the corpus itself — no labelled data needed.

    Writes pairs.train.jsonl, pairs.dev.jsonl and corpus.jsonl. The dev set
    holds out whole articles, so its queries point at text the model never
    saw as a positive.
    """
    from .train.pairs import KINDS, mine_directory, split_pairs, write_jsonl

    cfg = _cfg(config)
    pairs, corpus = mine_directory(cfg, path, doc2query=doc2query, query_log=query_log,
                                   cache_dir=out)
    train, dev = split_pairs(pairs, dev_fraction=dev_fraction)
    write_jsonl(out / "pairs.train.jsonl", (p.to_dict() for p in train))
    write_jsonl(out / "pairs.dev.jsonl", (p.to_dict() for p in dev))
    write_jsonl(out / "corpus.jsonl", corpus.to_records())

    table = Table(title=f"training pairs -> {out}")
    table.add_column("kind")
    table.add_column("train", justify="right")
    table.add_column("dev", justify="right")
    for kind in KINDS:
        nt = sum(1 for p in train if p.kind == kind)
        nd = sum(1 for p in dev if p.kind == kind)
        if nt or nd:
            table.add_row(kind, f"{nt:,}", f"{nd:,}")
    table.add_row("[bold]total[/bold]", f"[bold]{len(train):,}[/bold]", f"[bold]{len(dev):,}[/bold]")
    console.print(table)
    console.print(f"corpus: {len(corpus.chunks):,} chunks")


@app.command("train")
def train_cmd(pairs: Path = typer.Option(Path("data/train"), help="Directory written by train-pairs."),
              base: Path = typer.Option(Path("models/bge-m3"), help="Base sentence-transformers model."),
              out: Path = typer.Option(Path("models/bge-m3-lawrag"), help="Where to save the result."),
              epochs: float = typer.Option(1.0),
              batch_size: int = typer.Option(32, help="Effective batch (in-batch negatives)."),
              mini_batch_size: int = typer.Option(8, help="What is encoded at once; sets VRAM use."),
              learning_rate: float = typer.Option(2e-5),
              max_seq_length: int = typer.Option(512),
              train_layers: int = typer.Option(8, help="Top transformer layers to update; 0 = all."),
              hard_negatives: bool = typer.Option(True, help="Mine one hard negative per pair with the base model."),
              max_pairs: Optional[int] = typer.Option(None, help="Cap the training set (time budget)."),
              device: Optional[str] = typer.Option(None, help="cuda | cpu; auto if omitted."),
              config: Optional[str] = ConfigOpt):
    """Fine-tune the embedder on the mined pairs.

    Afterwards point embedding.model_path (LAWRAG_EMBEDDING_MODEL_PATH) at the
    output directory, re-ingest so every chunk is re-embedded, and re-run
    `calibrate`: min_dense_score is on the model's own scale and moves with it.
    """
    from .train.finetune import TrainSettings, finetune
    from .train.pairs import read_jsonl, read_pairs

    _cfg(config)
    train_pairs = read_pairs(pairs / "pairs.train.jsonl")
    corpus = read_jsonl(pairs / "corpus.jsonl")
    settings = TrainSettings(
        base=str(base), out=str(out), epochs=epochs, batch_size=batch_size,
        mini_batch_size=mini_batch_size, learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        train_layers=train_layers if train_layers > 0 else 10**6,
        hard_negatives=hard_negatives, max_pairs=max_pairs, device=device,
    )
    result = finetune(settings, train_pairs, corpus, cache_dir=pairs / "cache")
    console.print(f"[green]saved[/green] {result}")
    console.print("next: LAWRAG_EMBEDDING_MODEL_PATH=<that path>, re-ingest, then `lawrag calibrate`")


@app.command("train-eval")
def train_eval_cmd(models: list[Path] = typer.Argument(..., help="One or more model directories to compare."),
                   pairs: Path = typer.Option(Path("data/train"), help="Directory written by train-pairs."),
                   split: str = typer.Option("dev", help="dev | train"),
                   max_queries: Optional[int] = typer.Option(2000, help="Sample this many queries, balanced by kind."),
                   device: Optional[str] = typer.Option(None),
                   out: Optional[Path] = typer.Option(None, help="Write the numbers as JSON here too."),
                   config: Optional[str] = ConfigOpt):
    """Compare embedders on the held-out articles — dense retrieval, no database.

    Reports recall@1/5/10 and MRR per pair kind. `xref` (citation sentences)
    and `querylog` are the kinds that look most like real questions.
    """
    from .train.evaluate import evaluate, print_results, save_results
    from .train.pairs import read_jsonl, read_pairs

    _cfg(config)
    dev = read_pairs(pairs / f"pairs.{split}.jsonl")
    corpus = read_jsonl(pairs / "corpus.jsonl")
    results = [evaluate(m, dev, corpus, device=device, cache_dir=pairs / "cache",
                        max_queries=max_queries) for m in models]
    print_results(results, console)
    if out:
        save_results(out, results)


if __name__ == "__main__":
    app()

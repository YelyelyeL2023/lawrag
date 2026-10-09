"""Train the baseline outside the notebook (progress is visible in the terminal).

    .venv/bin/python notebooks/midterm/train_model.py [--force]

The notebook then loads the saved model instead of training it again.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from midterm import data as D, experiment as E, retrieval as R, train as T
from midterm.config import default_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even if a saved model exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = default_settings(Path(__file__).resolve().parent)
    device = T.pick_device()
    chunks, pairs = D.load(settings)
    index = R.Index(chunks)
    pretrained = T.load_model(settings.train.base_model, settings.train, device)
    _, doc_emb = E.dense_system(E.PRETRAINED, pretrained, "e5-small-pretrained", index, settings)
    _, meta = E.train_or_load(pretrained, doc_emb, index, pairs, settings, device, force=args.force)
    logging.info("model at %s, trained in %ss", settings.paths.model_out, meta["train_seconds"])


if __name__ == "__main__":
    main()

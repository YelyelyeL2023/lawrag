# Midterm — trained baseline retriever

Fine-tuned `intfloat/multilingual-e5-small` bi-encoder for (query → article) retrieval over the Kazakhstan codes,
compared with TF-IDF and the pretrained model on the Assignment 2 split (grouped by article, 80/10/10).

## Run

```bash
.venv/bin/pip install -r notebooks/midterm/requirements.txt
# optional: train in a terminal with progress (~45 min on Apple M1 Pro); the notebook reuses the saved model
.venv/bin/python notebooks/midterm/train_model.py
# full experiment: Restart → Run All, or
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/midterm/midterm_baseline.ipynb
```

No paths to edit: the repository root is found automatically, all settings are in `midterm/config.py`.
On Apple Silicon with 32 GB, cap MPS memory so the machine does not swap:
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5`.

Caches (rebuilt when missing): parsed corpus and split in `data/train/midterm/`, corpus embeddings in
`data/train/midterm/emb/`, model in `models/e5-small-lawrag/` (with `midterm_train.json`: config, loss log, time).

## Layout

| File | Role |
|---|---|
| `midterm_baseline.ipynb` | the report: data, model rationale, metrics, results, error analysis, conclusion |
| `midterm/config.py` | every setting (split, TF-IDF, training, evaluation, paths) |
| `midterm/data.py` | corpus + pairs, Assignment 2 cleaning and StratifiedGroupKFold split |
| `midterm/retrieval.py` | ranking protocol, recall@1 / recall@10 / MRR@10, TF-IDF and dense scorers |
| `midterm/train.py` | hard-negative mining and CachedMNRL fine-tuning |
| `midterm/experiment.py` | the three compared systems |
| `train_model.py` | training entry point outside the notebook |
| `outputs/` | `test_metrics.csv`, `test_by_kind.png`, run logs |

## Test results (3 246 queries, 740 unseen articles)

| System | recall@1 | recall@10 | MRR@10 |
|---|---|---|---|
| TF-IDF | 0.259 | 0.505 | 0.329 |
| e5-small, pretrained | 0.398 | 0.591 | 0.459 |
| e5-small, fine-tuned | 0.577 | 0.797 | 0.647 |

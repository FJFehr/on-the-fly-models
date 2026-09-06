# Tests

House style for writing a new test lives in [`.agents/STYLE.md`](../.agents/STYLE.md) §6
and [`.agents/AGENTS.md`](../.agents/AGENTS.md)'s "Test planning" section — read those
before adding one. Short version: tests are opt-in, not default; one rigorous, high-signal
test beats several partially-redundant ones; every test needs a docstring stating the
scenario, why it matters, and what regression it guards. This file is the map, not the
policy.

## Running

```bash
pytest                          # the default, fast tier -- ~629 cases, seconds
pytest -m slow                  # the slow tier too (needs a built dataset, see below)
pytest tests/test_x.py -v       # one file
```

## The three tiers

1. **Fast, hermetic unit tests** (the default — everything except `test_data_leakage.py`).
   Tiny in-memory fixtures only; nothing touches the repo's real `data/`. Covers
   `data_modules/` (dataset/filtering contracts), `models/` (architecture components,
   forward/backward correctness), `training/config.py`/`trainer.py`'s pure functions, and
   `visualisation/core/`.
2. **One lifecycle/integration test** (`test_train_lifecycle.py`) — the same real sequence
   `train.py` calls (config → build → fit → write artifacts), with a real but 1-step CPU
   `pl.Trainer.fit()` and a tiny synthetic dataset, asserting `config.yaml`/`model.txt`/
   `results.txt` land at the exact paths the rest of the repo's tooling
   (`experiments/README.md`'s workflow, `scripts/run_config.sh`) assumes. Still fast (a few
   seconds) — it's in the default tier, not `slow`.
3. **`slow`-marked tests** (`test_data_leakage.py`) — depend on a real, fully-built dataset
   on disk (`data/arc_1d_looped_augmented`, ~700K rows) and take real time (~1 minute) to
   fingerprint it. Excluded from the default run (`addopts` in `pyproject.toml`),
   `skipif`-guarded so a fresh checkout with no `data/` built yet skips cleanly instead of
   failing. Run deliberately, before a large rerun: `pytest -m slow`.

## Coverage map — including deliberate gaps

Most of `data_modules/` and `models/` is directly tested; the map below is only the
noteworthy exceptions — code with real coverage decisions behind it, not oversights.

| Area | Status | Why |
|---|---|---|
| `models/rope_looped_transformer.py` | Covered (`test_rope_canon_looped_transformer.py`) | The target/hypernetwork architecture in all 6 paper experiments — was zero-coverage, direct or indirect, until this file. One high-signal test (forward/backward correctness), not a sweep over `canon_set`/`n_loops`/skip-flag combinations — those are already covered structurally by `test_canon_layer.py`/`test_canon_transformer.py`. |
| `models/looped_transformer.py`, `models/recursive_transformer.py` | Untested, on purpose | Confirmed dead code — zero usage in any config anywhere in the repo, including `legacy/`. Awaiting the still-deferred `models/` file consolidation (merging into a smaller `hypernetwork.py`/`transformer.py`/`canon.py` set); not worth testing before that lands. |
| `visualisation/paper/` (`arc_paper.py`, `plot_tasks.py`, `plot_paper_tasks.py`, `plot_embedding_clusters.py`) | Untested, on purpose | Paper figure generators — verified by direct execution during reproduction (see `experiments/README.md`'s "Task examples" / each experiment's plotting section), not unit-tested. A rendering bug shows up immediately as a wrong or missing figure when regenerating the paper's plots. |
| `training/logging.py`'s W&B / hard-example-export / visualisation-gallery helpers (`create_wandb_logger` beyond what `test_train_lifecycle.py` exercises, `export_hard_validation_examples`, `log_final_task_visualizations`) | Exercised only indirectly | `test_train_lifecycle.py` deliberately stops short of `run_post_training_artifacts` — that pulls in a separate concern (does figure/gallery rendering work), not "does a run's core artifacts land in the right place." `log_embedding_cluster_plots` is the one exception with its own direct test (`test_log_embedding_cluster_plots.py`). |
| `data_modules/task_filtering.py` | Covered (`test_task_filtering.py`) | Shared by every live datamodule (`arc1d_direct`, `arc1d_meta_multiclass`) — was zero-coverage before this file, including its on-disk caching behavior (a hardcoded cache directory, monkeypatched to `tmp_path` in the test so nothing writes into the repo's real `data/`). |

## Shared fixtures (`conftest.py`)

- `make_arc1d_dataset_dict` — builds a tiny synthetic ARC1D `DatasetDict`, saves it under
  `tmp_path`, returns the directory path. The real save/load round trip a datamodule uses,
  without touching the repo's real `data/`. Shared by the `arc1d_direct`/
  `arc1d_meta_multiclass` datamodule tests and `test_train_lifecycle.py`.
- `build_hypermodel` / `make_hypermodel_batch` — factory for a tiny real
  `HyperModelLightning` (transformer hyper_model, rnn target) and a matching batch. Shared
  by every hypermodel-adjacent test (embeddings, variational bottleneck, embedding-cluster
  logging) that needs one — promoted here after these were found byte-identical or
  near-identical across three files.

Note: synthetic task "pixel values" in `make_arc1d_dataset_dict` are deliberately kept
small (mod 4) even though `task_id`s are large and distinguishing — anything that runs a
real forward pass needs valid ARC-1D colour values (0-9, 10 reserved for padding), not just
countable rows.

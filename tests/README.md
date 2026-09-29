# Tests

House style for writing a new test lives in [`.agents/STYLE.md`](../.agents/STYLE.md) §6
and [`.agents/AGENTS.md`](../.agents/AGENTS.md)'s "Test planning" section. Short version:
one rigorous, high-signal test beats several partially redundant ones, and every test needs
a docstring stating the scenario and the regression it guards against.

## Running

```bash
uv run pytest                      # default tier, about a minute on CPU
uv run pytest -m slow              # also the slow tier (needs the built dataset)
uv run pytest tests/test_x.py -v   # one file
```

The default tier builds and briefly trains tiny models, so run it on a machine with CPU to
spare (the lab convention is a GPU node, not a laptop).

## What is covered

1. **Unit tests** (tiny in-memory fixtures, nothing touches `data/`): data modules and task
   filtering, `models/` (`test_transformer.py`: Canon layer, Transformer shapes, fused vs
   manual attention), `lightning_modules/` (`test_lightning_modules.py`: forward shapes, loss
   masking, Muon parameter groups, frozen target and task indicator, vmap vs per-task loop,
   warmup plus cosine schedule, seed determinism), config loading, and `visualisation/core/`.
2. **Refactor guard** (`test_structure_snapshot.py`): every experiment config builds exactly
   the model it built before the `models/` refactor: same parameters in the same order, same
   Muon grouping and the same initial weights for a fixed seed. The reference is
   `fixtures/structure_snapshot.json`, recorded from the pre-refactor code by
   `fixtures/make_structure_snapshot.py`; `fixtures/old_checkpoints.py` maps old parameter
   names to new ones (and converts old checkpoints).
3. **End-to-end runs**: `test_train_lifecycle.py` runs the same sequence as `train.py` (config,
   build, 1-step fit, write `config.yaml`/`model.txt`/`results.txt`) on a tiny synthetic
   dataset; `test_evaluate_on.py` checks that `evaluate_on: final` scores and saves the final
   weights and `evaluate_on: best` reloads the best checkpoint.
4. **`slow` tier** (`test_data_leakage.py`): fingerprints the real
   `data/arc_1d_looped_augmented` (about 700K rows, about a minute) and checks that no
   training example appears in the validation, test or compositional holdout splits.
   Skipped when the dataset has not been built.

Deliberately not unit-tested: the paper figure scripts in `visualisation/paper/` and the
experiment plot scripts (a rendering bug shows up immediately as a wrong or missing figure),
and the W&B and figure-gallery helpers in `training/logging.py` beyond what the end-to-end
tests exercise.

## Shared fixtures (`conftest.py`)

- `make_arc1d_dataset_dict`: a tiny synthetic ARC-1D `DatasetDict` saved under `tmp_path`,
  for the data module and end-to-end tests. Pixel values are kept small (mod 4) so a forward
  pass sees valid colours (0-9, 10 is padding).
- `build_direct`, `build_hypernetwork`, `make_hypernetwork_batch`: tiny real Lightning modules
  and a batch of tasks (with padding) for the model tests.

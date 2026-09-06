# Legacy

Everything under here predates or was superseded by the paper experiments
in `experiments/` (see `experiments/README.md` for the current story). Kept
for provenance — moved with `git mv`, so `git log --follow` still finds
each file's full history — not because any of it is expected to run again
without some work.

- **`configs/experiments/`** — ~40 exploratory experiment families
  (`arc1d_capacity_*`, `arc1d_hypermodel_looped_*`, `arc1d_uniform_ablation`,
  `arc1d_binary`, `arc1d_multiclass`, ...), each documented by its own
  README the same way the active experiments are.
- **`scripts/`** — the `gen_*.py`/`run_*.sh` scripts that built and launched
  those config families, plus a couple of confirmed-dead analysis scripts
  (`plot_ablation.py`, `analyze_weight_space_pca.py`) already flagged
  out-of-scope by `docs/arc1d_story/03_experiments.md` before this move.
- **`data_modules/`** — `arc1d_meta_padded_multiclass.py` /
  `arc1d_meta_simple.py`, each used by exactly one legacy config family
  (`arc1d_multiclass` / `arc1d_binary`) and nothing in `experiments/`.
- **`visualisation/`** — `plot_capacity.py` (the old capacity-track
  plotter), `plot_boxplots.py` (unreferenced sequence-length boxplot tool),
  `weight_space.py` (only used by `scripts/analyze_weight_space_pca.py`
  above), and `attention.py`/`plot_attention.py` (a task-attention-gallery
  renderer whose only call site, `Lightning.log_task_attention_gallery`, is
  `hasattr`-guarded in `training/trainer.py`/`training/logging.py` but not
  implemented by any current model class — dead code path, nothing
  currently calls it).
- **`tests/`** — the test files for everything above.

None of these datamodules are registered in the active
`data_modules.DATA_REGISTRY` any more. To rerun a legacy config, re-add its
datamodule's import and registry entry in `data_modules/__init__.py` first.

"""Generate per-category, per-level configs for 06_data_efficiency_ablation_individual.

Companion sweep to 06_data_efficiency_ablation_hypernetwork (experiments/06_data_efficiency_ablation/hypernetwork/gen_configs.py) testing the
same variants_per_base_task data-reduction axis WITHOUT the hypernetwork: one
direct_supervised model per task category, trained directly on
rope_canon_looped_transformer (the exact same architecture as 06_data_efficiency_ablation_hypernetwork's
target_model), with no cross-task sharing at all.

variants_per_base_task in {1, 2, 3, full} this round (levels 4/5/20 from
06_data_efficiency_ablation_hypernetwork are out of scope for now). "full" (variants_per_base_task left
at None -- no reduction, the entire per-category train split) is a
self-contained full-data anchor at this experiment's own fixed compute
budget (max_steps=8000, since the 2026-09-14 unification onto experiment 1's
own dim=4 Individual recipe exactly), added 2026-09-10 alongside the other
two arms' own "full" cells.

Same 14 in-distribution task categories as 06_data_efficiency_ablation_hypernetwork/base.yaml -- the
paper's standard set (matching 01/02/05), not the original scaffolding's 15
(dropped 1d_recolor_cmp, 2026-09-10, to match 06_data_efficiency_ablation_hypernetwork's own resize).

2026-09-15: added a second, independent data-reduction axis --
base_tasks_per_category in {1, 3, 5, 10, 20}, always paired with
variants_per_base_task=1 (zero additional augmentation) -- to go BELOW v1's
~40-base-tasks/category floor by reducing task *diversity* rather than
augmentation *depth*. This is the sharpest test of the paper's
cross-task-sharing hypothesis: at t1, a single per-category model here trains
on exactly one base task's 3 support pairs -- the isolated-model floor the
hypothesis predicts hypernetwork/joint should pull ahead of. Tagged
{category}/t{N}.yaml (distinct from {category}/v{N}.yaml) to avoid any
naming collision. See data_modules/arc1d_direct.py's
_stratified_base_tasks_per_category for the (separately data_seed-seeded,
nested/reproducible) selection.

14 categories x (4 v/full levels + 5 t levels) x 5 seeds (seed handled by the
runner script, not baked into configs) = 630 jobs. Validation/test are
untouched at every level (see base.yaml) - only train_dataset shrinks (or,
at "full", doesn't).
"""

from pathlib import Path

import yaml

DST = Path("experiments/06_data_efficiency_ablation/individual/configs")
BASE_CFG = "experiments/06_data_efficiency_ablation/individual/configs/base.yaml"
PROJECT = "06_data_efficiency_ablation_individual"

TASK_CATEGORIES = [
    "1d_denoising_1c",
    "1d_denoising_mc",
    "1d_fill",
    "1d_flip",
    "1d_hollow",
    "1d_mirror",
    "1d_move_1p",
    "1d_move_2p",
    "1d_move_2p_dp",
    "1d_move_3p",
    "1d_move_dp",
    "1d_pcopy_1c",
    "1d_pcopy_mc",
    "1d_scale_dp",
]

# None ("full") means no reduction -- the entire per-category train split,
# same as leaving variants_per_base_task unset.
LEVELS = [1, 2, 3, None]
# Sub-40-base-tasks-per-category axis (2026-09-15), always at
# variants_per_base_task=1 (zero additional augmentation) -- see module docstring.
TASK_COUNT_LEVELS = [20, 10, 5, 3, 1]

n_written = 0
for category in TASK_CATEGORIES:
    category_dir = DST / category
    category_dir.mkdir(parents=True, exist_ok=True)
    for v in LEVELS:
        level_tag = "full" if v is None else f"v{v}"
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_baseline_{category}_{level_tag}",
            "task_categories": [category],
            "val_task_categories": [category],
            "variants_per_base_task": v,
        }
        out_path = category_dir / f"{level_tag}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

    for t in TASK_COUNT_LEVELS:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_baseline_{category}_t{t}",
            "task_categories": [category],
            "val_task_categories": [category],
            "variants_per_base_task": 1,
            "base_tasks_per_category": t,
        }
        out_path = category_dir / f"t{t}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

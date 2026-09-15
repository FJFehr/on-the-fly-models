"""Generate the data-reduction levels for arc1d_lowdata_joint.

Third arm alongside arc1d_lowdata (hypernetwork) and arc1d_lowdata_baseline
(fully-isolated individual): one shared backbone (direct_supervised, no
hypernetwork) trained jointly across all 14 task categories, with vs.
without a task-identity embedding (td/notd) -- reusing experiment 1's own
direct_supervised + task_encoding.use_task_embedding mechanism, at
experiment 1's own dim=14 "~10K-param scaffold" (9,508/9,688 params,
notd/td) -- sized to the hypernetwork's own total parameter budget, not its
dim=4 target.

variants_per_base_task in {1, 2, 3, 4, 5, 20, full} x {td, notd} -- same 6
reduced levels as arc1d_lowdata, plus a "full" cell (variants_per_base_task
left at None -- no reduction, the entire train split) run at the same fixed
compute budget as every other level (max_steps=8000, since the 2026-09-14
unification onto experiment 1's own td_dim14/notd_dim14 recipe exactly), so
it's a fair anchor point *within* this experiment.

Sampling is stratified/nested per underlying base task, identical mechanism
to the other two arms -- see data_modules/arc1d_direct.py's
_stratified_variants_per_base_task.

2026-09-15: added a second, independent data-reduction axis --
base_tasks_per_category in {1, 3, 5, 10, 20}, always paired with
variants_per_base_task=1 (zero additional augmentation) -- to go BELOW v1's
~40-base-tasks/category floor by reducing task *diversity* rather than
augmentation *depth*. Tagged cell_{cond}_t{N}.yaml (distinct from
cell_{cond}_v{N}.yaml) to avoid any naming collision. See
data_modules/arc1d_direct.py's _stratified_base_tasks_per_category for the
(separately data_seed-seeded, nested/reproducible) selection.

7 v/full levels + 5 t levels, x 2 conditions x 5 seeds (seed handled by the
runner script, not baked into configs) = 120 jobs. Validation/test are
untouched at every level (see base.yaml) - only train_dataset shrinks (or,
at "full", doesn't).
"""

from pathlib import Path

import yaml

DST = Path("experiments/06_data_efficiency_ablation/joint/configs")
BASE_CFG = "experiments/06_data_efficiency_ablation/joint/configs/base.yaml"
PROJECT = "arc1d_lowdata_joint"

# None ("full") means no reduction -- the entire train split, same as leaving
# variants_per_base_task unset.
LEVELS = [1, 2, 3, 4, 5, 20, None]
# Sub-40-base-tasks-per-category axis (2026-09-15), always at
# variants_per_base_task=1 (zero additional augmentation) -- see module docstring.
TASK_COUNT_LEVELS = [20, 10, 5, 3, 1]
CONDITIONS = {"notd": False, "td": True}

n_written = 0
for cond, use_task_embedding in CONDITIONS.items():
    for v in LEVELS:
        level_tag = "full" if v is None else f"v{v}"
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_joint_{cond}_{level_tag}",
            "variants_per_base_task": v,
            "task_encoding": {"use_task_embedding": use_task_embedding},
        }
        out_path = DST / f"cell_{cond}_{level_tag}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

    for t in TASK_COUNT_LEVELS:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_joint_{cond}_t{t}",
            "variants_per_base_task": 1,
            "base_tasks_per_category": t,
            "task_encoding": {"use_task_embedding": use_task_embedding},
        }
        out_path = DST / f"cell_{cond}_t{t}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

"""Generate the data-reduction levels for arc1d_lowdata.

Tests whether the hypernetwork needs less training data than a model with no
cross-task transfer, by training experiment 2's own dim=4 matched-scale
recipe -- frozen_td and notd, Muon -- at progressively less training data
per base task, stratified and nested:

variants_per_base_task in {1, 2, 3, 4, 5, 20, full} x {frozentd, notd}

notd is included alongside frozentd (added 2026-09-10) specifically so it
pairs against ../joint/'s own notd arm at every data level -- frozentd alone
can't tell you whether a data-efficiency gap over ../joint/'s td arm is about
weight generation specifically or just the task-identity signal.

"full" (variants_per_base_task left at None -- no reduction, the entire
train split) is a self-contained full-data anchor at this experiment's own
fixed compute budget (max_steps=2000), rather than borrowing experiment 2's
own full-data numbers (which used max_steps=8000) -- needed for a fair
within-experiment comparison across every level, not just the reduced ones.

Sampling is stratified per underlying base task (not a raw random count over
the pool) and nested/cumulative: level K always includes the true original
(aug_index=0, unaugmented) example for every base task first, then K-1 more
augmented variants in a fixed data_seed order, so level K's training set is
always a subset of level K+1's. See
data_modules/arc1d_meta_multiclass.py's _stratified_variants_per_base_task.

7 levels x 2 conditions x 5 seeds (seed handled by the runner script, not
baked into configs) = 70 jobs. Validation/test are untouched at every level
(see base.yaml) - only train_dataset shrinks (or, at "full", doesn't).
"""

from pathlib import Path

import yaml

DST = Path("experiments/06_data_efficiency_ablation/hypernetwork/configs")
BASE_CFG = "experiments/06_data_efficiency_ablation/hypernetwork/configs/base.yaml"
PROJECT = "arc1d_lowdata"

# None ("full") means no reduction -- the entire train split, same as leaving
# variants_per_base_task unset.
LEVELS = [1, 2, 3, 4, 5, 20, None]
# Matches 02_hypernetwork_multitask/configs/dim4_frozentd.yaml / dim4_notd.yaml exactly.
CONDITIONS = {
    "frozentd": {"num_tasks": 18, "freeze_task_indicator": True},
    "notd": {"num_tasks": None, "freeze_task_indicator": False},
}

n_written = 0
for cond, hyper_head_overrides in CONDITIONS.items():
    for v in LEVELS:
        level_tag = "full" if v is None else f"v{v}"
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_{cond}_{level_tag}",
            "variants_per_base_task": v,
            "hyper_head": hyper_head_overrides,
        }
        out_path = DST / f"cell_{cond}_{level_tag}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

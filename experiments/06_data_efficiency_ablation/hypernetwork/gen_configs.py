"""Write the configs for experiment 6's hypernetwork arm (configs/cell_<condition>_<level>.yaml).

Experiment 2's dim-4 hypernetwork, frozentd and notd, at 12 data levels (t1 to t20, v1 to v5,
v20, full): 24 configs, 120 jobs at 5 seeds.

Levels (see ../README.md): t1, t3, t5, t10, t20 keep 1 to 20 base tasks per category with
original examples only (base_tasks_per_category); v{N} keeps all base tasks with N variants
each (variants_per_base_task); full applies no reduction. Selection is deterministic
(data_seed) and nested, and only the training split shrinks. Seeds are added by run.sh, not
written into the configs.
"""

from pathlib import Path

import yaml

DST = Path("experiments/06_data_efficiency_ablation/hypernetwork/configs")
BASE_CFG = "experiments/06_data_efficiency_ablation/hypernetwork/configs/base.yaml"
PROJECT = "06_data_efficiency_ablation_hypernetwork"

# None ("full") means no reduction -- the entire train split, same as leaving
# variants_per_base_task unset.
LEVELS = [1, 2, 3, 4, 5, 20, None]
# Sub-40-base-tasks-per-category axis (2026-09-15), always at
# variants_per_base_task=1 (zero additional augmentation) -- see module docstring.
TASK_COUNT_LEVELS = [20, 10, 5, 3, 1]
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

    for t in TASK_COUNT_LEVELS:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_{cond}_t{t}",
            "variants_per_base_task": 1,
            "base_tasks_per_category": t,
            "hyper_head": hyper_head_overrides,
        }
        out_path = DST / f"cell_{cond}_t{t}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

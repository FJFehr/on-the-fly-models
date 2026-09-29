"""Write the configs for experiment 6's individual arm (configs/<category>/<level>.yaml).

Experiment 1's individual dim-4 model, one per task category, at 9 data levels (t1 to t20,
v1 to v3, full): 14 x 9 = 126 configs, 630 jobs at 5 seeds.

Levels (see ../README.md): t1, t3, t5, t10, t20 keep 1 to 20 base tasks per category with
original examples only (base_tasks_per_category); v{N} keeps all base tasks with N variants
each (variants_per_base_task); full applies no reduction. Selection is deterministic
(data_seed) and nested, and only the training split shrinks. Seeds are added by run.sh, not
written into the configs.
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

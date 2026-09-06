"""Generate per-category, per-level configs for arc1d_lowdata_baseline.

Companion sweep to arc1d_lowdata (experiments/06_data_efficiency_ablation/hypernetwork/gen_configs.py) testing the
same variants_per_base_task data-reduction axis WITHOUT the hypernetwork: one
LoopedSupervisedLightning model per task category, trained directly on
rope_canon_looped_transformer (the exact same architecture as arc1d_lowdata's
target_model), with no cross-task sharing at all.

variants_per_base_task in {1, 2, 3} this round (levels 4/5/20 from
arc1d_lowdata are out of scope for now).

Same 15 in-distribution task categories as arc1d_lowdata/base.yaml (NOT
arc1d_uniform_ablation's 17 -- the 2 extra categories are intentionally
excluded).

15 categories x 3 levels x 3 seeds (seed handled by the runner script, not
baked into configs) = 135 jobs. Validation/test are untouched at every level
(see base.yaml) - only train_dataset shrinks.
"""

from pathlib import Path

import yaml

DST = Path("experiments/06_data_efficiency_ablation/individual/configs")
BASE_CFG = "experiments/06_data_efficiency_ablation/individual/configs/base.yaml"
PROJECT = "arc1d_lowdata_baseline"

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
    "1d_recolor_cmp",
    "1d_scale_dp",
]

VARIANTS_PER_BASE_TASK = [1, 2, 3]

n_written = 0
for category in TASK_CATEGORIES:
    category_dir = DST / category
    category_dir.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS_PER_BASE_TASK:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_baseline_{category}_v{v}",
            "task_categories": [category],
            "val_task_categories": [category],
            "variants_per_base_task": v,
        }
        out_path = category_dir / f"v{v}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

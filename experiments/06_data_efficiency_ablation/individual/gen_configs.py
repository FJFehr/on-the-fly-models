"""Generate per-category, per-level configs for arc1d_lowdata_baseline.

Companion sweep to arc1d_lowdata (experiments/06_data_efficiency_ablation/hypernetwork/gen_configs.py) testing the
same variants_per_base_task data-reduction axis WITHOUT the hypernetwork: one
LoopedSupervisedLightning model per task category, trained directly on
rope_canon_looped_transformer (the exact same architecture as arc1d_lowdata's
target_model), with no cross-task sharing at all.

variants_per_base_task in {1, 2, 3, full} this round (levels 4/5/20 from
arc1d_lowdata are out of scope for now). "full" (variants_per_base_task left
at None -- no reduction, the entire per-category train split) is a
self-contained full-data anchor at this experiment's own fixed compute
budget (max_steps=2000), added 2026-09-10 alongside the other two arms' own
"full" cells, so all three arms have a same-budget full-data comparison
point rather than borrowing experiment 1/2's own (max_steps=8000) numbers.

Same 14 in-distribution task categories as arc1d_lowdata/base.yaml -- the
paper's standard set (matching 01/02/05), not the original scaffolding's 15
(dropped 1d_recolor_cmp, 2026-09-10, to match arc1d_lowdata's own resize).

14 categories x 4 levels x 5 seeds (seed handled by the runner script, not
baked into configs) = 280 jobs. Validation/test are untouched at every level
(see base.yaml) - only train_dataset shrinks (or, at "full", doesn't).
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
    "1d_scale_dp",
]

# None ("full") means no reduction -- the entire per-category train split,
# same as leaving variants_per_base_task unset.
LEVELS = [1, 2, 3, None]

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

print(f"Written {n_written} configs to {DST}/")

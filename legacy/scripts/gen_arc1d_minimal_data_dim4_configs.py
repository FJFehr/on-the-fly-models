"""Experiment 3 rerun at dim=4: minimal data augmentation, 14-task set.

Generates per-category, per-level configs sweeping variants_per_base_task in
{300, 100, 10, 5, 2} against
configs/experiments/arc1d_v2_minimal_data_dim4/base.yaml (dim=4, the minimal
architecture from Experiment 1, not RC1's dim=10). Same task set and
mechanism as gen_arc1d_minimal_data_configs.py; only the base architecture
and level list differ (pushed lower since the dim=10 sweep never found a
floor, even at variants_per_base_task=10).

14-task set (1d_recolor_cmp excluded per the Phase 1 scope decision, see
docs/arc1d_story/06_phase1_findings.md).

Same category/v{N}.yaml leaf layout as gen_arc1d_minimal_data_configs.py, so
scripts/run_minimal_data_dim4.sh (a copy of run_minimal_data.sh with only
the top constants changed) needs no further changes.

5 levels x 14 categories x 5 seeds (seed handled by the runner script, not
baked into configs) = 350 jobs. Validation/test are untouched at every level
(see base.yaml) -- only train_dataset shrinks.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_v2_minimal_data_dim4")
BASE_CFG = "configs/experiments/arc1d_v2_minimal_data_dim4/base.yaml"

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

VARIANTS_PER_BASE_TASK = [300, 100, 10, 5, 2]

n_written = 0
for category in TASK_CATEGORIES:
    category_dir = DST / category
    category_dir.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS_PER_BASE_TASK:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"v2_minimal_data_dim4_{category}_v{v}",
            "task_categories": [category],
            "val_task_categories": [category],
            "variants_per_base_task": v,
        }
        out_path = category_dir / f"v{v}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/ (x5 seeds at launch = {n_written * 5} jobs)")

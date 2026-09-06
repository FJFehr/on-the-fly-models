"""Experiment 3: minimal data augmentation, 14-task set, RC1 architecture.

Generates per-category, per-level configs sweeping variants_per_base_task in
{500, 300, 100, 10} against configs/experiments/arc1d_v2_minimal_data/base.yaml
(RC1: RoPE + Canon, dim=10, flat n_loops=1, N_sup=1, direct_supervised, Muon).

variants_per_base_task=1000 (full pool, no cap) is already covered by
Phase 1's RC1 runs (arc1d_v2_backbone_capacity/) -- not regenerated here.

14-task set (1d_recolor_cmp excluded per the Phase 1 scope decision, see
docs/arc1d_story/06_phase1_findings.md) -- NOT the 15-category list used by
the older, now-superseded arc1d_lowdata_baseline generator.

Same category/v{N}.yaml leaf layout as gen_lowdata_baseline_configs.py, so
scripts/run_minimal_data.sh (a copy of run_lowdata_baseline.sh with only the
top constants changed) needs no further changes.

4 levels x 14 categories x 3 seeds (seed handled by the runner script, not
baked into configs) = 168 jobs. Validation/test are untouched at every level
(see base.yaml) -- only train_dataset shrinks.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_v2_minimal_data")
BASE_CFG = "configs/experiments/arc1d_v2_minimal_data/base.yaml"

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

VARIANTS_PER_BASE_TASK = [500, 300, 100, 10]

n_written = 0
for category in TASK_CATEGORIES:
    category_dir = DST / category
    category_dir.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS_PER_BASE_TASK:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"v2_minimal_data_{category}_v{v}",
            "task_categories": [category],
            "val_task_categories": [category],
            "variants_per_base_task": v,
        }
        out_path = category_dir / f"v{v}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/ (x3 seeds at launch = {n_written * 3} jobs)")

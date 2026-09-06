"""Generate a 5-seed sweep for arc1d_v2_compositional_generalization_14task's two arms.

The single-seed notd.yaml/frozen_td.yaml (seed=42, matching every sibling
arc1d_v2_compositional_generalization* experiment) stay as they are. This adds
seed-baked leaf configs alongside them so scripts/eval_compositional_holdout.py
(which has no dotlist-override support, unlike train.py -- it re-reads
cfg.output_path straight from --config) can resolve each seed's own checkpoint
directory without any script changes.

2 arms x 5 seeds = 10 configs, written as <arm>_seed<N>.yaml.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_v2_compositional_generalization_14task")
BASE_CFG = "configs/experiments/arc1d_v2_compositional_generalization_14task/base.yaml"

SEEDS = [1, 2, 3, 4, 5]
ARMS = {
    "notd": {"num_tasks": None, "freeze_task_indicator": False},
    "frozen_td": {"num_tasks": 28, "freeze_task_indicator": True},
}

n_written = 0
for arm, hyper_head_overrides in ARMS.items():
    for seed in SEEDS:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"v2_compositional_generalization_14task_{arm}_seed{seed}",
            "seed": seed,
            "hyper_head": hyper_head_overrides,
        }
        out_path = DST / f"{arm}_seed{seed}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

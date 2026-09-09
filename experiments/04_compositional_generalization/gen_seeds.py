"""Generate a 5-seed from-scratch-retrain sweep for this experiment's two arms.

Documentation/provenance only -- these leaf configs aren't used by the active pipeline
(run.sh reuses experiment 2's own checkpoints instead, see this experiment's README), but stay
buildable in case a genuine from-scratch retrain is ever needed (e.g. a future dim=6 version).

The single-seed notd.yaml/frozen_td.yaml (seed=42) stay as they are (frozen_td.yaml is also
directly reused by run.sh as the architecture template for the padded-checkpoint eval). This
adds seed-baked leaf configs alongside them so scripts/eval_compositional_holdout.py (which
has no dotlist-override support, unlike train.py -- it re-reads cfg.output_path straight from
--config) can resolve each seed's own checkpoint directory without any script changes.

2 arms x 5 seeds = 10 configs, written as <arm>_seed<N>.yaml (notd_seed<N>.yaml /
frozentd_seed<N>.yaml -- dirname-token naming, matching experiment 2's own convention).
"""

from pathlib import Path

import yaml

DST = Path("experiments/04_compositional_generalization/configs")
BASE_CFG = "experiments/04_compositional_generalization/configs/base.yaml"

SEEDS = [1, 2, 3, 4, 5]
ARMS = {
    "notd": {"num_tasks": None, "freeze_task_indicator": False},
    "frozentd": {"num_tasks": 28, "freeze_task_indicator": True},
}

n_written = 0
for arm, hyper_head_overrides in ARMS.items():
    for seed in SEEDS:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"{arm}_seed{seed}",
            "seed": seed,
            "hyper_head": hyper_head_overrides,
        }
        out_path = DST / f"{arm}_seed{seed}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

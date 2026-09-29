"""Generate experiment 7's leave-one-out leaf configs: every one of the 14 base ARC-1D task
categories held out in turn, x the 4 task-identity arms, x 5 seeds -- 280 leaf configs.

Each leaf inherits experiment 5's base.yaml unchanged, so the recipe matches experiment 5's
notd and frozentd runs exactly. As there, task_categories drops the held-out category and
val_task_categories keeps all 14, so results.txt reports the held-out category's zero-shot
score. The held-out category is also listed in hyper_head.zero_task_categories: it is scored
with a zero task vector, not with its (untrained) task-embedding column.

frozentd_latent is experiment 5's frozentd, retrained here because experiment 5 kept no
checkpoints to rescore: training is identical (same config and seed), only the held-out
category's scoring differs. notd has no task vector, so experiment 5's own runs are used.

Usage:
    uv run python experiments/07_task_identity_ablation/gen_loo_configs.py
"""

from pathlib import Path

import yaml

DST = Path("experiments/07_task_identity_ablation/configs/loo")
BASE_CFG = "experiments/05_leave_one_out_task_generalization/configs/base.yaml"

ALL_CATEGORIES = [
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

SEEDS = [1, 2, 3, 4, 5]
ARMS = {
    "frozentd_latent": {"freeze_task_indicator": True, "placement": "latent"},
    "learnedtd_latent": {"freeze_task_indicator": False, "placement": "latent"},
    "frozentd_input": {"freeze_task_indicator": True, "placement": "input"},
    "learnedtd_input": {"freeze_task_indicator": False, "placement": "input"},
}


def short_name(category: str) -> str:
    return category.removeprefix("1d_").replace("_", "")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for held_out in ALL_CATEGORIES:
        short = short_name(held_out)
        train_categories = [c for c in ALL_CATEGORIES if c != held_out]
        for arm, hyper_head_overrides in ARMS.items():
            for seed in SEEDS:
                name = f"{short}_{arm}_seed{seed}"
                cfg = {
                    "_base_": BASE_CFG,
                    "project_name": "07_task_identity_ablation",
                    "experiment_name": f"loo_{name}",
                    "seed": seed,
                    "task_categories": train_categories,
                    "val_task_categories": ALL_CATEGORIES,
                    "hyper_head": {
                        "num_tasks": 18,
                        **hyper_head_overrides,
                        "zero_task_categories": [held_out],
                    },
                }
                with open(DST / f"{name}.yaml", "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                n_written += 1

    print(f"Written {n_written} configs to {DST}/")
    print(f"({len(ALL_CATEGORIES)} held-out categories x {len(ARMS)} arms x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()

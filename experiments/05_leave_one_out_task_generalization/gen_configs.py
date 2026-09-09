"""Generate experiment 5's leave-one-out leaf configs: every one of the 14 base ARC-1D task
categories held out in turn, x {notd, frozentd}, x 5 seeds -- 140 leaf configs total.

Each leaf sets task_categories to the other 13 categories (the held-out one dropped) and
val_task_categories to the full 14, so results.txt's automatic per-category breakdown
(val_query_exact_match_by_task_<category>, val_query_accuracy_by_task_<category>) reports the
held-out category's zero-shot score alongside the 13 in-distribution ones, from the same
training run -- no separate eval script needed (see base.yaml and this experiment's README).

hyper_head.num_tasks stays 18 for frozentd regardless of which category is held out -- it's
sized off the fixed global TASK_CATEGORY_INDEX registry (models/hypermodel_lightning.py), not
off how many categories a given leaf trains on.

Short names (for filenames/experiment_name) are derived mechanically -- strip the "1d_" prefix
and drop underscores (e.g. 1d_move_2p_dp -> move2pdp) -- matching the token style
legacy/scripts/gen_v2_generalization_configs.py used by hand for its curated 5-category subset.

Usage:
    uv run python experiments/05_leave_one_out_task_generalization/gen_configs.py
"""

from pathlib import Path

import yaml

DST = Path("experiments/05_leave_one_out_task_generalization/configs")
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
    "notd": {"num_tasks": None, "freeze_task_indicator": False},
    "frozentd": {"num_tasks": 18, "freeze_task_indicator": True},
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
                    "experiment_name": name,
                    "seed": seed,
                    "task_categories": train_categories,
                    "val_task_categories": ALL_CATEGORIES,
                    "hyper_head": hyper_head_overrides,
                }
                out_path = DST / f"{name}.yaml"
                with open(out_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                n_written += 1

    print(f"Written {n_written} configs to {DST}/")
    print(f"({len(ALL_CATEGORIES)} held-out categories x {len(ARMS)} arms x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()

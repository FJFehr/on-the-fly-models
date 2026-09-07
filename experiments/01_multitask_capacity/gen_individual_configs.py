"""Generate the Individual-baseline configs for experiment 1 (multi-task capacity).

One model per task category, no cross-task sharing -- the third line on the
capacity-cliff plot, alongside the Joint (notd/td) arms already in this
folder. Same RoPE+Canon architecture, optimizer, and step budget as the
joint arms at each size, so the comparison is apples-to-apples; the only
difference is `task_categories` narrowed to a single category and no
task-identity embedding (meaningless for a model that only ever sees one
task).

Previously this data came from two now-gone sources outside this experiment
folder: `legacy/configs/experiments/arc1d_v2_backbone_capacity/` (dim=10,
3 seeds, the original Phase 1 "RC1" run) and a `arc1d_v2_minimal_size` sweep
(dim=4/6, 3 seeds) that was deleted outright rather than archived to
`legacy/` during the paper-repro reorg (recoverable from git history at
commit 40e91c0 if ever needed for reference). This generator makes
experiment 1 self-contained instead: same 4 sizes as the joint arms
(4/6/10/14), 5 seeds each, matching the seed count used everywhere else in
this experiment.

14 tasks x 4 dims = 56 configs. At 5 seeds via scripts/run_config.sh, that's
280 training jobs -- run separately from the 45-job joint-arm sweep (see
this experiment's README "Running" section).

Run once (regenerates configs/individual/ from scratch):

    uv run python experiments/01_multitask_capacity/gen_individual_configs.py
"""

from pathlib import Path

import yaml

DST = Path("experiments/01_multitask_capacity/configs/individual")
BASE_CFG = "experiments/01_multitask_capacity/configs/base.yaml"

# Same 14-task set as notd*.yaml/td*.yaml (drops 1d_padded_fill, 1d_recolor_cmp,
# 1d_recolor_cnt, 1d_recolor_oe from the full 18 -- see docs/arc1d_story/01_data.md).
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

# Same widths as notd_dim4/6/14.yaml and notd.yaml (dim=10); embedding_dim
# fixed at 10 throughout, matching every other size in this experiment.
DIMS = [4, 6, 10, 14]
EMBEDDING_DIM = 10
HEADS = 1

MUON_PARAMS = {
    "optimizer": "Muon",
    "muon_lr": 0.005,
    "muon_momentum": 0.95,
}

CANON_PARAMS = {
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
}


def config_at(dim: int, category: str) -> dict:
    return {
        "_base_": BASE_CFG,
        "experiment_name": f"individual_dim{dim}_{category}",
        "project_name": "01_multitask_capacity",
        "task_categories": [category],
        "save_checkpoints": False,
        "model": "direct_supervised",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "task_encoding": {
            "embedding_dim": EMBEDDING_DIM,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
            "use_task_embedding": False,
        },
        "backbone_model": {
            "name": "rope_canon_looped_transformer",
            "params": {
                "hidden_dim": dim,
                "num_heads": HEADS,
                "inner_dim": dim,
                "inner_num_heads": HEADS,
                "n_loops": 1,
                "dropout": 0.1,
                **CANON_PARAMS,
                "use_block_skip": False,
                "use_loop_skip": False,
            },
        },
    }


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for category in TASK_CATEGORIES:
        out_dir = DST / category
        out_dir.mkdir(exist_ok=True)
        for dim in DIMS:
            cfg = config_at(dim, category)
            out_path = out_dir / f"dim{dim}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n_written += 1
    n_seeds = 5
    print(
        f"Written {n_written} configs to {DST}/ "
        f"(x{n_seeds} seeds at launch = {n_written * n_seeds} jobs)"
    )


if __name__ == "__main__":
    main()

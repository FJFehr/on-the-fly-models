"""Generate the AdamW-optimizer ablation configs for experiment 1.

Every current Joint/Individual config uses Muon (`muon_lr=0.005,
muon_momentum=0.95`). This generates a mirror of all of them with
`optimizer: AdamW` instead, Canon left enabled (`canon_set: 'ABCD'`, same as
the main configs) so this isolates the optimizer choice alone, not
optimizer-and-Canon together.

Learning rate: falls back to this experiment's base.yaml default
(`learning_rate: 0.001`, `weight_decay: 0.01`, both already tuned for
RAdam) rather than the Muon arms' 0.0005 -- Muon's learning-rate scale
doesn't transfer to AdamW (a different, materially larger update rule), so
reusing it would understate AdamW unfairly. This is a reasonable starting
point, not a value tuned specifically for AdamW on this task; revisit if
the ablation numbers look off in a way a learning-rate mismatch would
explain.

Writes into configs/adamw/joint/ and configs/adamw/individual/ (kept
separate from the canon/no-canon configs' own directories so `CFG_DIR` can
target this sweep independently):

    experiments/01_multitask_capacity/configs/adamw/joint/{notd,td}_dim{4,6,10,14}.yaml
    experiments/01_multitask_capacity/configs/adamw/individual/<category>/dim{4,6,10,14}.yaml

8 joint configs + 56 individual configs = 64 configs; x5 seeds at launch =
320 jobs.

Run once (regenerates configs/adamw/ from scratch):

    uv run python experiments/01_multitask_capacity/gen_adamw_configs.py
"""

from pathlib import Path

import yaml

DST = Path("experiments/01_multitask_capacity/configs/adamw")
BASE_CFG = "experiments/01_multitask_capacity/configs/base.yaml"

# Same 14-task set as the canon joint/individual configs.
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

DIMS = [4, 6, 10, 14]
EMBEDDING_DIM = 10
HEADS = 1


def backbone_at(dim: int) -> dict:
    """Same architecture as the main configs, Canon left enabled."""
    return {
        "name": "rope_canon_looped_transformer",
        "params": {
            "hidden_dim": dim,
            "num_heads": HEADS,
            "inner_dim": dim,
            "inner_num_heads": HEADS,
            "n_loops": 1,
            "dropout": 0.1,
            "canon_set": "ABCD",
            "canon_kernel": 5,
            "canon_activation": True,
            "canon_residual": True,
            "canon_causal": False,
            "use_block_skip": False,
            "use_loop_skip": False,
        },
    }


def common_fields(dim: int) -> dict:
    return {
        "_base_": BASE_CFG,
        "project_name": "01_multitask_capacity",
        "save_checkpoints": False,
        "model": "direct_supervised",
        "optimizer": "AdamW",  # <- the ablation: no Muon, no muon_lr/muon_momentum
        "max_steps": 8000,
        "backbone_model": backbone_at(dim),
    }


def gen_joint() -> int:
    out_dir = DST / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for use_task_embedding, cond in ((False, "notd"), (True, "td")):
        for dim in DIMS:
            cfg = {
                **common_fields(dim),
                "experiment_name": f"joint_adamw_{cond}_dim{dim}",
                "task_categories": list(TASK_CATEGORIES),
                "task_encoding": {
                    "embedding_dim": EMBEDDING_DIM,
                    "value_vocab_size": 11,
                    "use_sinusoidal_pe": False,
                    "use_task_embedding": use_task_embedding,
                },
            }
            with open(out_dir / f"{cond}_dim{dim}.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n += 1
    return n


def gen_individual() -> int:
    out_root = DST / "individual"
    n = 0
    for category in TASK_CATEGORIES:
        out_dir = out_root / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for dim in DIMS:
            cfg = {
                **common_fields(dim),
                "experiment_name": f"individual_adamw_dim{dim}_{category}",
                "task_categories": [category],
                "task_encoding": {
                    "embedding_dim": EMBEDDING_DIM,
                    "value_vocab_size": 11,
                    "use_sinusoidal_pe": False,
                    "use_task_embedding": False,
                },
            }
            with open(out_dir / f"dim{dim}.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n += 1
    return n


def main() -> None:
    n_joint = gen_joint()
    n_individual = gen_individual()
    n_seeds = 5
    total = n_joint + n_individual
    print(
        f"Written {n_joint} joint + {n_individual} individual = {total} configs "
        f"to {DST}/ (x{n_seeds} seeds at launch = {total * n_seeds} jobs)"
    )


if __name__ == "__main__":
    main()

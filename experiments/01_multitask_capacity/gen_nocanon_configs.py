"""Generate the Canon ablation configs for experiment 1 (multi-task capacity).

Hypothesis: the Canon layers (short causal convolutions applied at various
points in the RoPE+Canon backbone) boost accuracy across the board, and the
boost is largest at the smallest sizes -- i.e. Canon buys the most where raw
capacity is scarcest. This generates a no-canon mirror of every config
already in this experiment (Joint notd/td and Individual, all 4 dims, 5
seeds) so the two can be compared directly, size for size.

Disabling Canon needs no code change and no new model: `canon_set: ''` on
the same Transformer backbone instantiates none of the
Canon positions (A/B/C/D), leaving pure RoPE -- the same mechanism already
used by the legacy `R2_*_rope_only.yaml` ablation configs. Everything else
(architecture width, optimizer, step budget, task set) is identical to the
corresponding canon config, so any accuracy gap is attributable to Canon
alone.

Writes into configs/nocanon/joint/ and configs/nocanon/individual/ (kept
separate from the canon configs' own directories so `CFG_DIR` can target
either sweep independently):

    experiments/01_multitask_capacity/configs/nocanon/joint/{notd,td}_dim{4,6,10,14}.yaml
    experiments/01_multitask_capacity/configs/nocanon/individual/<category>/dim{4,6,10,14}.yaml

8 joint configs + 56 individual configs = 64 configs; x5 seeds at launch =
320 jobs.

Run once (regenerates configs/nocanon/ from scratch):

    uv run python experiments/01_multitask_capacity/gen_nocanon_configs.py
"""

from pathlib import Path

import yaml

DST = Path("experiments/01_multitask_capacity/configs/nocanon")
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

MUON_PARAMS = {
    "optimizer": "Muon",
    "muon_lr": 0.005,
    "muon_momentum": 0.95,
}


def backbone_at(dim: int) -> dict:
    """Same architecture as the canon configs, canon_set emptied."""
    return {
        "hidden_dim": dim,
        "num_layers": 3,
        "num_heads": HEADS,
        "dropout": 0.1,
        "canon_set": "",  # <- the ablation: no Canon positions, pure RoPE
        "canon_kernel": 5,
    }


def common_fields(dim: int) -> dict:
    return {
        "_base_": BASE_CFG,
        "project_name": "01_multitask_capacity",
        "save_checkpoints": False,
        "model": "direct",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "backbone": backbone_at(dim),
    }


def gen_joint() -> int:
    out_dir = DST / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for use_task_embedding, cond in ((False, "notd"), (True, "td")):
        for dim in DIMS:
            cfg = {
                **common_fields(dim),
                "experiment_name": f"joint_nocanon_{cond}_dim{dim}",
                "task_categories": list(TASK_CATEGORIES),
                "task_encoding": {
                    "embedding_dim": EMBEDDING_DIM,
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
                "experiment_name": f"individual_nocanon_dim{dim}_{category}",
                "task_categories": [category],
                "task_encoding": {
                    "embedding_dim": EMBEDDING_DIM,
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

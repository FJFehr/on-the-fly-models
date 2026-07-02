"""Generate configs for arc1d_rope_unet_skip_ablation.

Tests proper U-Net style skip connections vs the existing per-iteration loop skip.

Base: outer=8, inner=32, Canon ABCD, RoPE, n_loops=4, N_sup=2, lr=0.0005, 4000 steps.
Conditions A, B, F reused from previous experiments.

  N: inner bypass only  — single skip over all N loops after last iteration
  O: outer bypass only  — single skip over all 3 blocks (pre + loops + post)
  P: inner + outer      — U-Net: both bypasses applied once
  Q: block + inner + outer — block skip inside each block + both U-Net bypasses
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_unet_skip_ablation")
BASE_CFG = "configs/experiments/arc1d_rope_unet_skip_ablation/base_unet_skip.yaml"
PROJECT = "arc1d_rope_unet_skip_ablation"
SKIP_DIRS = {"overfit"}

_BASE_PARAMS = {
    "hidden_dim": 8,
    "num_heads": 1,
    "inner_dim": 32,
    "inner_num_heads": 4,
    "dropout": 0.1,
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
    "n_loops": 4,
    "use_block_skip": False,
    "use_loop_skip": False,
}

CONDITIONS = {
    # N: inner bypass — single skip over all loops, applied once after last iteration
    "N": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "use_inner_bypass": True, "use_outer_bypass": False},
        },
    },
    # O: outer bypass — single skip from input_proj to after post_layer
    "O": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "use_inner_bypass": False, "use_outer_bypass": True},
        },
    },
    # P: U-Net — inner + outer bypass (no block skip)
    "P": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "use_inner_bypass": True, "use_outer_bypass": True},
        },
    },
    # Q: block + U-Net — all three skips combined
    "Q": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "use_block_skip": True,
                       "use_inner_bypass": True, "use_outer_bypass": True},
        },
    },
}

COND_SUFFIX = {
    "N": "unet_n_inner",
    "O": "unet_o_outer",
    "P": "unet_p_both",
    "Q": "unet_q_all",
}

DST.mkdir(parents=True, exist_ok=True)

n_written = 0
for task_dir in sorted(d for d in SRC_TASKS.iterdir() if d.is_dir()):
    if task_dir.name in SKIP_DIRS:
        continue
    task = task_dir.name
    out_task_dir = DST / task
    out_task_dir.mkdir(exist_ok=True)

    src_a = task_dir / "A_transformer.yaml"
    with open(src_a) as f:
        src_cfg = yaml.safe_load(f)
    task_categories = src_cfg.get("task_categories", [task])

    for cond, overrides in CONDITIONS.items():
        exp_name = f"ablation_{cond}_{task}_{COND_SUFFIX[cond]}"
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": exp_name,
            "project_name": PROJECT,
            "task_categories": task_categories,
            **overrides,
        }
        out_path = out_task_dir / f"{cond}_{COND_SUFFIX[cond]}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

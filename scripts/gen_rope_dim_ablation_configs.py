"""Generate configs for arc1d_rope_dim_ablation.

Jointly sweep inner_dim and skip configuration to find the smallest architecture
that solves all 18 ARC-1D tasks. Also includes outer hidden_dim=16 to test
whether the output representation bottleneck matters.

References:
  A  (inner=32, no skip)      — reused from arc1d_rope_wide_middle_ablation
  F  (inner=32, block+loop)   — reused from arc1d_rope_loop_skip_ablation

New conditions:
  H  inner=16, outer=8,  no skip           → 6,448 params
  I  inner=16, outer=8,  block+loop skip   → 6,448 params
  J  inner=64, outer=8,  no skip           → 55,552 params
  K  inner=64, outer=8,  block+loop skip   → 55,552 params
  L  inner=32, outer=16, no skip           → 22,624 params  (larger outer dim)
  M  inner=32, outer=16, block+loop skip   → 22,624 params

num_heads chosen to maintain head_dim=8 throughout.
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_dim_ablation")
BASE_CFG = "configs/experiments/arc1d_rope_dim_ablation/base_dim_ablation.yaml"
PROJECT = "arc1d_rope_dim_ablation"
SKIP_DIRS = {"overfit"}

_CANON_PARAMS = {
    "dropout": 0.1,
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
    "n_loops": 4,
}

CONDITIONS = {
    # H: smallest — inner=16, no skip
    "H": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 8, "num_heads": 1,
                "inner_dim": 16, "inner_num_heads": 2,
                "use_block_skip": False, "use_loop_skip": False,
                **_CANON_PARAMS,
            },
        },
    },
    # I: smallest + best skips — inner=16, block+loop
    "I": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 8, "num_heads": 1,
                "inner_dim": 16, "inner_num_heads": 2,
                "use_block_skip": True, "use_loop_skip": True,
                **_CANON_PARAMS,
            },
        },
    },
    # J: large capacity alone — inner=64, no skip
    "J": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 8, "num_heads": 1,
                "inner_dim": 64, "inner_num_heads": 8,
                "use_block_skip": False, "use_loop_skip": False,
                **_CANON_PARAMS,
            },
        },
    },
    # K: large capacity + best skips — inner=64, block+loop
    "K": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 8, "num_heads": 1,
                "inner_dim": 64, "inner_num_heads": 8,
                "use_block_skip": True, "use_loop_skip": True,
                **_CANON_PARAMS,
            },
        },
    },
    # L: wider outer layers — outer=16, inner=32, no skip
    "L": {
        "task_encoding": {"embedding_dim": 8, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 16, "num_heads": 2,
                "inner_dim": 32, "inner_num_heads": 4,
                "use_block_skip": False, "use_loop_skip": False,
                **_CANON_PARAMS,
            },
        },
    },
    # M: wider outer layers + best skips — outer=16, inner=32, block+loop
    "M": {
        "task_encoding": {"embedding_dim": 8, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 16, "num_heads": 2,
                "inner_dim": 32, "inner_num_heads": 4,
                "use_block_skip": True, "use_loop_skip": True,
                **_CANON_PARAMS,
            },
        },
    },
}

COND_SUFFIX = {
    "H": "dim_h16_base",
    "I": "dim_i16_skip",
    "J": "dim_j64_base",
    "K": "dim_k64_skip",
    "L": "dim_l16out_base",
    "M": "dim_m16out_skip",
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

"""Generate configs for arc1d_rope_wide_middle_ablation.

Wide-middle sandwich: small outer layers + wide looped middle (inner_dim=32).
Sweeps outer_dim ∈ {8, 16} × n_loops ∈ {4, 8, 16}.
All conditions: RoPE, Canon ABCD, N_sup=2, lr=0.0005, 4000 steps.

  A: outer=8,  n_loops=4  (6L)
  B: outer=8,  n_loops=8  (10L)
  C: outer=8,  n_loops=16 (18L)
  D: outer=16, n_loops=4  (6L)
  E: outer=16, n_loops=8  (10L)
  F: outer=16, n_loops=16 (18L)
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_wide_middle_ablation")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_rope_wide_middle_ablation"
SKIP_DIRS = {"overfit"}

_CANON_PARAMS = {
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
}

_COMMON = {
    "model": "looped_supervised",
    "N_supervision": 2,
    "learning_rate": 0.0005,
    "max_steps": 4000,
}

CONDITIONS = {
    # outer_dim=8, inner_dim=32
    "A": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 8, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 8, "num_heads": 1,
                "inner_dim": 32, "inner_num_heads": 4,
                "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS,
            },
        },
    },
    "B": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 8, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 8, "num_heads": 1,
                "inner_dim": 32, "inner_num_heads": 4,
                "n_loops": 8, "dropout": 0.1, **_CANON_PARAMS,
            },
        },
    },
    "C": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 8, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 8, "num_heads": 1,
                "inner_dim": 32, "inner_num_heads": 4,
                "n_loops": 16, "dropout": 0.1, **_CANON_PARAMS,
            },
        },
    },
    # outer_dim=16, inner_dim=32
    "D": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 16, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 16, "num_heads": 2,
                "inner_dim": 32, "inner_num_heads": 4,
                "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS,
            },
        },
    },
    "E": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 16, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 16, "num_heads": 2,
                "inner_dim": 32, "inner_num_heads": 4,
                "n_loops": 8, "dropout": 0.1, **_CANON_PARAMS,
            },
        },
    },
    "F": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 16, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                "hidden_dim": 16, "num_heads": 2,
                "inner_dim": 32, "inner_num_heads": 4,
                "n_loops": 16, "dropout": 0.1, **_CANON_PARAMS,
            },
        },
    },
}

COND_SUFFIX = {
    "A": "wide8_6L",
    "B": "wide8_10L",
    "C": "wide8_18L",
    "D": "wide16_6L",
    "E": "wide16_10L",
    "F": "wide16_18L",
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

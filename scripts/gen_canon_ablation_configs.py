"""Generate configs for arc1d_recursion_ablation_128_canon.

8-condition ablation: the original 4 conditions (A–D) from arc1d_recursion_ablation_128
paired with 4 Canon variants (E–H) that use the same dims and compute budget.

  A: direct,  transformer              128h 2L           8000 steps  lr=0.001
  B: direct,  recursive_transformer   128h 1L n_loops=2  8000 steps  lr=0.001
  C: looped,  transformer              128h 2L  N_sup=2  4000 steps  lr=0.0005
  D: looped,  recursive_transformer   128h 1L n_loops=2  N_sup=2     4000 steps  lr=0.0005
  E: direct,  canon_transformer        128h 2L           8000 steps  lr=0.001
  F: direct,  canon_recursive_transformer 128h 1L n_loops=2  8000 steps  lr=0.001
  G: looped,  canon_transformer        128h 2L  N_sup=2  4000 steps  lr=0.0005
  H: looped,  canon_recursive_transformer 128h 1L n_loops=2  N_sup=2 4000 steps  lr=0.0005

Canon params: canon_set="ABCD", canon_kernel=4, canon_activation=True, canon_residual=True.
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_recursion_ablation_128_canon")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_recursion_ablation_128_canon"
SKIP_DIRS = {"overfit"}

_CANON_PARAMS = {
    "canon_set": "ABCD",
    "canon_kernel": 4,
    "canon_activation": True,
    "canon_residual": True,
}

CONDITIONS = {
    # --- Baselines (mirror of arc1d_recursion_ablation_128) ---
    "A": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.1},
        },
    },
    "B": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "recursive_transformer",
            "params": {"hidden_dim": 128, "num_layers": 1, "num_heads": 4, "n_loops": 2, "dropout": 0.1},
        },
    },
    "C": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.1},
        },
    },
    "D": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "recursive_transformer",
            "params": {"hidden_dim": 128, "num_layers": 1, "num_heads": 4, "n_loops": 2, "dropout": 0.1},
        },
    },
    # --- Canon variants (same dims/compute, ABCD canon positions) ---
    "E": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "canon_transformer",
            "params": {"hidden_dim": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "F": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "canon_recursive_transformer",
            "params": {"hidden_dim": 128, "num_layers": 1, "num_heads": 4, "n_loops": 2, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "G": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "canon_transformer",
            "params": {"hidden_dim": 128, "num_layers": 2, "num_heads": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "H": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "canon_recursive_transformer",
            "params": {"hidden_dim": 128, "num_layers": 1, "num_heads": 4, "n_loops": 2, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
}

COND_SUFFIX = {
    "A": "transformer",
    "B": "recursive_transformer",
    "C": "looped_transformer",
    "D": "looped_recursive_transformer",
    "E": "canon_transformer",
    "F": "canon_recursive_transformer",
    "G": "looped_canon_transformer",
    "H": "looped_canon_recursive_transformer",
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

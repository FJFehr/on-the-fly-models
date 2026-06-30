"""Generate configs for arc1d_sandwich_ablation.

Sandwich architecture: fixed pre/post layers with a weight-shared looped middle.
All conditions share hidden_dim=32, lr=0.0005.

  A: direct,  sandwich,        n_loops=1 (3L baseline)   8000 steps
  B: direct,  sandwich,        n_loops=2 (4L)            8000 steps
  C: direct,  sandwich,        n_loops=4 (6L)            8000 steps
  D: looped,  sandwich,        n_loops=1, N_sup=2         4000 steps
  E: looped,  sandwich,        n_loops=2, N_sup=2         4000 steps
  F: looped,  sandwich,        n_loops=4, N_sup=2         4000 steps
  G: direct,  canon_sandwich,  n_loops=1 (3L + Canon)    8000 steps
  H: direct,  canon_sandwich,  n_loops=2 (4L + Canon)    8000 steps
  I: direct,  canon_sandwich,  n_loops=4 (6L + Canon)    8000 steps
  J: looped,  canon_sandwich,  n_loops=1, N_sup=2         4000 steps
  K: looped,  canon_sandwich,  n_loops=2, N_sup=2         4000 steps
  L: looped,  canon_sandwich,  n_loops=4, N_sup=2         4000 steps

N_sup=2 uses 4000 steps (4000 × 2 = 8000 optimizer updates, matching N_sup=1 at 8000 steps).
Canon params: canon_set="ABCD", canon_kernel=5, canon_causal=False (non-causal).
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_sandwich_ablation")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_sandwich_ablation"
SKIP_DIRS = {"overfit"}

_CANON_PARAMS = {
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
}

CONDITIONS = {
    # --- Direct supervision, plain sandwich ---
    "A": {
        "model": "direct_supervised",
        "learning_rate": 0.0005,
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 1, "dropout": 0.1},
        },
    },
    "B": {
        "model": "direct_supervised",
        "learning_rate": 0.0005,
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 2, "dropout": 0.1},
        },
    },
    "C": {
        "model": "direct_supervised",
        "learning_rate": 0.0005,
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 4, "dropout": 0.1},
        },
    },
    # --- Looped supervision (N_sup=2), plain sandwich ---
    "D": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 1, "dropout": 0.1},
        },
    },
    "E": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 2, "dropout": 0.1},
        },
    },
    "F": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 4, "dropout": 0.1},
        },
    },
    # --- Direct supervision, Canon sandwich ---
    "G": {
        "model": "direct_supervised",
        "learning_rate": 0.0005,
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 1, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "H": {
        "model": "direct_supervised",
        "learning_rate": 0.0005,
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 2, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "I": {
        "model": "direct_supervised",
        "learning_rate": 0.0005,
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    # --- Looped supervision (N_sup=2), Canon sandwich ---
    "J": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 1, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "K": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 2, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "L": {
        "model": "looped_supervised",
        "N_supervision": 2,
        "learning_rate": 0.0005,
        "max_steps": 4000,
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
}

COND_SUFFIX = {
    "A": "sandwich",
    "B": "sandwich_2loops",
    "C": "sandwich_4loops",
    "D": "looped_sandwich",
    "E": "looped_sandwich_2loops",
    "F": "looped_sandwich_4loops",
    "G": "canon_sandwich",
    "H": "canon_sandwich_2loops",
    "I": "canon_sandwich_4loops",
    "J": "looped_canon_sandwich",
    "K": "looped_canon_sandwich_2loops",
    "L": "looped_canon_sandwich_4loops",
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

"""Generate configs for arc1d_rope_sandwich_ablation.

Compares sinusoidal PE vs RoPE in the Canon Sandwich architecture across
two model sizes (dim=16, dim=32) and two effective depths (6L, 8L).
All conditions: N_sup=2, lr=0.0005, 4000 steps (= 8000 optimizer updates).

  A: sinusoidal, dim=16, heads=2, n_loops=4 (6L)
  B: sinusoidal, dim=16, heads=2, n_loops=6 (8L)
  C: sinusoidal, dim=32, heads=4, n_loops=4 (6L)
  D: sinusoidal, dim=32, heads=4, n_loops=6 (8L)
  E: RoPE,       dim=16, heads=2, n_loops=4 (6L)
  F: RoPE,       dim=16, heads=2, n_loops=6 (8L)
  G: RoPE,       dim=32, heads=4, n_loops=4 (6L)
  H: RoPE,       dim=32, heads=4, n_loops=6 (8L)

Canon params (all): canon_set="ABCD", canon_kernel=5, canon_causal=False.
embedding_dim = hidden_dim per condition (model scales together).
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_sandwich_ablation")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_rope_sandwich_ablation"
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
    # --- Sinusoidal PE, CanonSandwichTransformer ---
    "A": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 16, "value_vocab_size": 11, "use_sinusoidal_pe": True},
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 16, "num_heads": 2, "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "B": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 16, "value_vocab_size": 11, "use_sinusoidal_pe": True},
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 16, "num_heads": 2, "n_loops": 6, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "C": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 32, "value_vocab_size": 11, "use_sinusoidal_pe": True},
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "D": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 32, "value_vocab_size": 11, "use_sinusoidal_pe": True},
        "backbone_model": {
            "name": "canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 6, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    # --- RoPE, RoPECanonSandwichTransformer ---
    "E": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 16, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {"hidden_dim": 16, "num_heads": 2, "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "F": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 16, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {"hidden_dim": 16, "num_heads": 2, "n_loops": 6, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "G": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 32, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 4, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
    "H": {
        **_COMMON,
        "task_encoding": {"embedding_dim": 32, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {"hidden_dim": 32, "num_heads": 4, "n_loops": 6, "dropout": 0.1, **_CANON_PARAMS},
        },
    },
}

COND_SUFFIX = {
    "A": "sin_dim16_6L",
    "B": "sin_dim16_8L",
    "C": "sin_dim32_6L",
    "D": "sin_dim32_8L",
    "E": "rope_dim16_6L",
    "F": "rope_dim16_8L",
    "G": "rope_dim32_6L",
    "H": "rope_dim32_8L",
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

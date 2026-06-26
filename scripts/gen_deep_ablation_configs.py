"""Generate configs for arc1d_recursion_ablation_deep.

4 conditions, all with 256h and 8k equivalent steps:
  A: Transformer,          4 layers,  direct,         8000 steps
  B: RecursiveTransformer, 4 loops,   direct,         8000 steps
  C: Transformer,          4 layers,  N_sup=8 looped, 1000 steps (8000 equiv)
  D: RecursiveTransformer, 8 loops,   N_sup=8 looped, 1000 steps (8000 equiv)
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST       = Path("configs/experiments/arc1d_recursion_ablation_deep")
BASE_CFG  = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT   = "arc1d_recursion_ablation_deep"

SKIP_DIRS = {"overfit"}

CONDITIONS = {
    "A": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": 256, "num_layers": 4, "num_heads": 4, "dropout": 0.1},
        },
    },
    "B": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "recursive_transformer",
            "params": {"hidden_dim": 256, "num_layers": 1, "num_heads": 4, "n_loops": 4, "dropout": 0.1},
        },
    },
    "C": {
        "model": "looped_supervised",
        "N_supervision": 8,
        "learning_rate": 0.000125,
        "max_steps": 1000,
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": 256, "num_layers": 4, "num_heads": 4, "dropout": 0.1},
        },
    },
    "D": {
        "model": "looped_supervised",
        "N_supervision": 8,
        "learning_rate": 0.000125,
        "max_steps": 1000,
        "backbone_model": {
            "name": "recursive_transformer",
            "params": {"hidden_dim": 256, "num_layers": 1, "num_heads": 4, "n_loops": 8, "dropout": 0.1},
        },
    },
}

COND_SUFFIX = {
    "A": "transformer",
    "B": "recursive_transformer",
    "C": "looped_transformer",
    "D": "looped_recursive_transformer",
}

DST.mkdir(parents=True, exist_ok=True)

n_written = 0
for task_dir in sorted(d for d in SRC_TASKS.iterdir() if d.is_dir()):
    if task_dir.name in SKIP_DIRS:
        continue

    task = task_dir.name
    out_task_dir = DST / task
    out_task_dir.mkdir(exist_ok=True)

    # Read task_categories from the existing A config
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

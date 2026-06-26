"""Generate configs for arc1d_recursion_ablation_2l6.

4 conditions, using the deep ablation task set and seed launch pattern:
  A: Transformer,          2 layers, direct,         8000 steps
  B: RecursiveTransformer, 6 loops,  direct,         8000 steps
  C: Transformer,          2 layers, N_sup=6 looped, 1000 steps
  D: RecursiveTransformer, 6 loops,  N_sup=6 looped, 1000 steps
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_recursion_ablation_2l6")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_recursion_ablation_2l6"

SKIP_DIRS = {"overfit"}

CONDITIONS = {
    "A": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": 256, "num_layers": 2, "num_heads": 4, "dropout": 0.1},
        },
    },
    "B": {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "max_steps": 8000,
        "backbone_model": {
            "name": "recursive_transformer",
            "params": {
                "hidden_dim": 256,
                "num_layers": 1,
                "num_heads": 4,
                "n_loops": 6,
                "dropout": 0.1,
            },
        },
    },
    "C": {
        "model": "looped_supervised",
        "N_supervision": 6,
        "learning_rate": 1 / 6000,
        "max_steps": 1000,
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": 256, "num_layers": 2, "num_heads": 4, "dropout": 0.1},
        },
    },
    "D": {
        "model": "looped_supervised",
        "N_supervision": 6,
        "learning_rate": 1 / 6000,
        "max_steps": 1000,
        "backbone_model": {
            "name": "recursive_transformer",
            "params": {
                "hidden_dim": 256,
                "num_layers": 1,
                "num_heads": 4,
                "n_loops": 6,
                "dropout": 0.1,
            },
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

    # Read task_categories from the existing A config.
    src_a = task_dir / "A_transformer.yaml"
    with src_a.open() as f:
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
        out_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

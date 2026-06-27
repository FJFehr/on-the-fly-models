"""Generate configs for arc1d_recursion_ablation_128.

Small/shallow models, 8k equivalent steps:
  A: Transformer,          128h, 2 layers, direct,  8000 steps, lr=0.001
  B: RecursiveTransformer, 128h, 2 loops,  direct,  8000 steps, lr=0.001
  C: Transformer,          128h, 2 layers, N_sup=2, 4000 steps, lr=0.0005
  D: RecursiveTransformer, 128h, 2 loops,  N_sup=2, 4000 steps, lr=0.0005
"""

from pathlib import Path
import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST       = Path("configs/experiments/arc1d_recursion_ablation_128")
BASE_CFG  = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT   = "arc1d_recursion_ablation_128"
SKIP_DIRS = {"overfit"}

CONDITIONS = {
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

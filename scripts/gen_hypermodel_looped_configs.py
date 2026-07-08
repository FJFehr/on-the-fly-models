"""Generate single-task configs for arc1d_hypermodel_looped.

Phase 1 of the hypernetwork-looped-canon-rope story: one hypernetwork target
per task category (no task descriptor, no multi-task mixing yet), all using
the locked-in target model (dim=16, n_loops=8, no block/loop skip -- see
configs/experiments/arc1d_uniform_ablation's L1 result).

1d_padded_fill is excluded (never part of the per-task hypermodel_augmented
family this mirrors; see base_hypermodel_looped.yaml's header comment).
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_hypermodel_looped")
BASE_CFG = "configs/experiments/arc1d_hypermodel_looped/base_hypermodel_looped.yaml"
PROJECT = "arc1d_hypermodel_looped"
SKIP_DIRS = {"overfit", "1d_padded_fill"}

TARGET_MODEL_PARAMS = {
    "hidden_dim": 16,
    "num_heads": 2,
    "inner_dim": 16,
    "inner_num_heads": 2,
    "n_loops": 8,
    "dropout": 0.1,
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
    "use_block_skip": False,
    "use_loop_skip": False,
}

DST.mkdir(parents=True, exist_ok=True)

n_written = 0
for task_dir in sorted(d for d in SRC_TASKS.iterdir() if d.is_dir()):
    if task_dir.name in SKIP_DIRS:
        continue
    task = task_dir.name  # e.g. "1d_move_1p"
    short_task = task.removeprefix("1d_")  # e.g. "move_1p"
    out_task_dir = DST / task
    out_task_dir.mkdir(exist_ok=True)

    cfg = {
        "_base_": BASE_CFG,
        "task": short_task,
        "experiment_name": f"looped_hyper_{short_task}_rope_canon_looped",
        "task_categories": [task],
        "val_task_categories": [task],
        "target_model": {
            "name": "rope_canon_looped_transformer",
            "params": dict(TARGET_MODEL_PARAMS),
        },
    }
    out_path = out_task_dir / "rope_canon_looped.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    n_written += 1

print(f"Written {n_written} configs to {DST}/")

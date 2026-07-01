"""Generate configs for arc1d_rope_loop_skip_ablation.

Loop-level skip connection experiment: inject the pre-loop anchor state at every
loop iteration, giving the model a guaranteed direct path to its initial representation.

Base: outer=8, inner=32, RoPE, Canon ABCD, n_loops=4, N_sup=2, lr=0.0005, 4000 steps.
Conditions A (wide8_6L) and B (skip_abcd_hw) are reused from previous experiments.

  E: loop_skip only, 4 loops        — core new mechanism
  F: block_skip + loop_skip, 4L     — both skips combined
  G: loop_skip only, 8 loops        — more iterations with anchor injection
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_loop_skip_ablation")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_rope_loop_skip_ablation"
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
}

_COMMON = {
    "model": "looped_supervised",
    "N_supervision": 2,
    "learning_rate": 0.0005,
    "max_steps": 4000,
    "task_encoding": {"embedding_dim": 8, "value_vocab_size": 11, "use_sinusoidal_pe": False},
}

CONDITIONS = {
    # E: loop-level anchor injection only (new mechanism)
    "E": {
        **_COMMON,
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "n_loops": 4, "use_block_skip": False, "use_loop_skip": True},
        },
    },
    # F: block skip + loop skip combined
    "F": {
        **_COMMON,
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "n_loops": 4, "use_block_skip": True, "use_loop_skip": True},
        },
    },
    # G: loop skip with 8 loops — more compute with anchor injection
    "G": {
        **_COMMON,
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "n_loops": 8, "use_block_skip": False, "use_loop_skip": True},
        },
    },
}

COND_SUFFIX = {
    "E": "loop_skip_e4",
    "F": "loop_skip_f4",
    "G": "loop_skip_g8",
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

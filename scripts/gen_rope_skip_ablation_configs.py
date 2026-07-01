"""Generate configs for arc1d_rope_skip_ablation.

Diagnostic experiment for flip/padded_fill failure modes.
Base architecture: outer=8, inner=32, RoPE, n_loops=4, N_sup=2, lr=0.0005, 4000 steps.

  B: Canon ABCD + block highway skip (all layers)  → does highway fix flip?
  C: Canon BCD  (no A, no skip)                    → is Canon-A the attention culprit?
  D: Canon ACD  (no B, no skip)                    → is Canon-B (QKV mixing) the culprit?

Condition A (Canon ABCD, no skip) already exists in arc1d_rope_wide_middle_ablation.
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_skip_ablation")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_rope_skip_ablation"
SKIP_DIRS = {"overfit"}

_BASE_PARAMS = {
    "hidden_dim": 8,
    "num_heads": 1,
    "inner_dim": 32,
    "inner_num_heads": 4,
    "n_loops": 4,
    "dropout": 0.1,
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
    # B: Canon ABCD + block-level highway skip on all layers
    "B": {
        **_COMMON,
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "canon_set": "ABCD", "use_block_skip": True},
        },
    },
    # C: Canon BCD (remove A — keep QKV and MLP convolutions, clean attention input)
    "C": {
        **_COMMON,
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "canon_set": "BCD", "use_block_skip": False},
        },
    },
    # D: Canon ACD (remove B — keep attention-input and MLP convolutions, clean QKV)
    "D": {
        **_COMMON,
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {**_BASE_PARAMS, "canon_set": "ACD", "use_block_skip": False},
        },
    },
}

COND_SUFFIX = {
    "B": "skip_abcd_hw",
    "C": "skip_bcd",
    "D": "skip_acd",
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

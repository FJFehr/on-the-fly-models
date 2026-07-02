"""Generate configs for arc1d_rope_story_ablation.

Three new conditions that fill in the story between the plain transformer
(arc1d_recursion_ablation) and the full Canon+skip model. All use RoPE, no Canon,
N_sup=2, outer=8 embedding dim.

  S3: Flat 3L transformer with RoPE  (n_loops=1, dim=16, no Canon)
  S4: Looped middle with RoPE        (n_loops=4, dim=16, no Canon)
  S5: Wide middle with RoPE          (outer=8, inner=32, n_loops=4, no Canon)

1d_padded_fill is excluded (not included in the story narrative).
"""

from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_rope_story_ablation")
BASE_CFG = "configs/experiments/arc1d_rope_story_ablation/base_story.yaml"
PROJECT = "arc1d_rope_story_ablation"
SKIP_DIRS = {"overfit", "1d_padded_fill"}

_ROPE_COMMON = {
    "dropout": 0.1,
    "canon_set": "",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
    "use_block_skip": False,
    "use_loop_skip": False,
}

CONDITIONS = {
    # S3: flat 3-layer RoPE transformer (n_loops=1 = pre + 1×middle + post = 3 blocks)
    "S3": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 1,
            },
        },
    },
    # S4: looped middle, same dim (n_loops=4 = 6 effective blocks, weight-shared middle)
    "S4": {
        "task_encoding": {
            "embedding_dim": 16,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": False,
        },
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "hidden_dim": 16,
                "num_heads": 2,
                "inner_dim": 16,
                "inner_num_heads": 2,
                "n_loops": 4,
            },
        },
    },
    # S5: wide middle (outer=8, inner=32, n_loops=4) — concentrates capacity, no Canon yet
    "S5": {
        "backbone_model": {
            "name": "rope_canon_sandwich_transformer",
            "params": {
                **_ROPE_COMMON,
                "hidden_dim": 8,
                "num_heads": 1,
                "inner_dim": 32,
                "inner_num_heads": 4,
                "n_loops": 4,
            },
        },
    },
}

COND_SUFFIX = {
    "S3": "story_rope_flat",
    "S4": "story_rope_loop",
    "S5": "story_wide_nc",
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

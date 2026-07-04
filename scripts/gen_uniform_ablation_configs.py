"""Generate configs for arc1d_uniform_ablation.

Clean, single-project rebuild of the ARC-1D design-journey story, replacing the
scattered SC1-SC10 diagnostics in arc1d_rope_story_ablation and the various
legacy projects (arc1d_32_canon_ablation, arc1d_rope_sandwich_ablation, etc.)
with one fully self-contained set of configs, run at two widths (dim=16 and
dim=32) uniformly throughout (no outer/inner "sandwich" split - diagnostics
this session showed the sandwich buys nothing over parameter-matched uniform
width, and looping is essential).

Seven steps, identical structure at both widths:

  T1: Vanilla transformer, sin PE, N_sup=1        (direct_supervised)
  T2: + N_sup=2 loop training
  T3: + Canon ABCD
  T4: + RoPE, flat (n_loops=1)
  T5: + Looped middle (n_loops=4)
  T6: + Block skip
  T7: + Per-loop h0 (loop skip)

Design choices (confirmed by user):
  - num_layers=3 always for the plain transformer/canon_transformer family
    (T1-T3), for block-count parity with the RoPE-sandwich family's
    pre+middle+post=3 blocks at n_loops=1 (T4). Verified: canon_transformer
    (num_layers=3) and rope_canon_sandwich_transformer (n_loops=1) both give
    exactly 11,760 backbone params at dim=16, and 42,464 at dim=32.
  - N_supervision=2 for every looped step (T2-T7); T1 is direct (N_sup=1).
  - learning_rate=0.0005 throughout, including T1.
  - max_steps scaled inversely with N_sup so total compute is constant:
    T1 (N_sup=1) -> max_steps=8000; T2-T7 (N_sup=2) -> max_steps=4000.
  - head_dim=8 convention: dim=16 -> 2 heads, dim=32 -> 4 heads.

1d_padded_fill is excluded (established convention for this story family).
"""

import copy
from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_uniform_ablation")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_uniform_ablation"
SKIP_DIRS = {"overfit", "1d_padded_fill"}

DIMS = {16: 2, 32: 4}  # dim -> num_heads (head_dim=8 convention)

CANON_PARAMS = {
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
}


def step1(dim: int, heads: int) -> dict:
    """Vanilla transformer, sin PE, N_sup=1."""
    return {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "task_encoding": {
            "embedding_dim": dim,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": True,
        },
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": dim, "num_layers": 3, "num_heads": heads, "dropout": 0.1},
        },
    }


def step2(dim: int, heads: int) -> dict:
    """+ N_sup=2 loop training."""
    cfg = step1(dim, heads)
    del cfg["gradient_clip_val"]
    cfg["model"] = "looped_supervised"
    cfg["N_supervision"] = 2
    cfg["max_steps"] = 4000
    return cfg


def step3(dim: int, heads: int) -> dict:
    """+ Canon ABCD."""
    cfg = step2(dim, heads)
    cfg["backbone_model"] = {
        "name": "canon_transformer",
        "params": {"hidden_dim": dim, "num_layers": 3, "num_heads": heads, "dropout": 0.1, **CANON_PARAMS},
    }
    return cfg


def step4(dim: int, heads: int) -> dict:
    """+ RoPE, flat (n_loops=1)."""
    cfg = step3(dim, heads)
    cfg["task_encoding"]["use_sinusoidal_pe"] = False
    cfg["backbone_model"] = {
        "name": "rope_canon_sandwich_transformer",
        "params": {
            "hidden_dim": dim,
            "num_heads": heads,
            "inner_dim": dim,
            "inner_num_heads": heads,
            "n_loops": 1,
            "dropout": 0.1,
            **CANON_PARAMS,
            "use_block_skip": False,
            "use_loop_skip": False,
        },
    }
    return cfg


def step5(dim: int, heads: int) -> dict:
    """+ Looped middle (n_loops=4)."""
    cfg = step4(dim, heads)
    cfg["backbone_model"]["params"]["n_loops"] = 4
    return cfg


def step6(dim: int, heads: int) -> dict:
    """+ Block skip."""
    cfg = step5(dim, heads)
    cfg["backbone_model"]["params"]["use_block_skip"] = True
    return cfg


def step7(dim: int, heads: int) -> dict:
    """+ Per-loop h0 (loop skip)."""
    cfg = step6(dim, heads)
    cfg["backbone_model"]["params"]["use_loop_skip"] = True
    return cfg


STEP_BUILDERS = {
    "T1": step1,
    "T2": step2,
    "T3": step3,
    "T4": step4,
    "T5": step5,
    "T6": step6,
    "T7": step7,
}

STEP_SUFFIX = {
    "T1": "vanilla",
    "T2": "nsup",
    "T3": "canon",
    "T4": "rope_flat",
    "T5": "looped",
    "T6": "block_skip",
    "T7": "loop_skip",
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

    for dim, heads in DIMS.items():
        for step, builder in STEP_BUILDERS.items():
            suffix = STEP_SUFFIX[step]
            exp_name = f"ablation_{step}_dim{dim}_{task}_{suffix}"
            cfg = {
                "_base_": BASE_CFG,
                "experiment_name": exp_name,
                "project_name": PROJECT,
                "task_categories": task_categories,
                **copy.deepcopy(builder(dim, heads)),
            }
            out_path = out_task_dir / f"{step}_dim{dim}_{suffix}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n_written += 1

print(f"Written {n_written} configs to {DST}/")

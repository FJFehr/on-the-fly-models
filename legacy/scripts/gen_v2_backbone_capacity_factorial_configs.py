"""Completes the RoPE x Canon x N_sup factorial, dim=10, 14-task set.

T1/T2 (neither), R2/R3 (RoPE only), T3 (Canon only, N_sup=2), T4 (both,
N_sup=2) already exist. Two cells were never tested - both at N_sup=1
(direct_supervised, matching T1/R2's training regime, flat/n_loops=1
throughout):

  C1:  Canon only, sin PE, N_sup=1   (= T3's architecture at N_sup=1)
  RC1: RoPE + Canon, N_sup=1          (= T4's architecture at N_sup=1)

Motivated by T3 (Canon alone, N_sup=2) scoring 0.9952 - matching or beating
T4 (RoPE+Canon, N_sup=2, 0.9905) - suggesting RoPE may add nothing once
Canon is present. These two cells ask whether N_sup=2 is doing anything
either, once Canon is present.

14-task set (1d_recolor_cmp excluded per the Phase 1 scope decision, see
docs/arc1d_story/05_round2_plan.md and 01_data.md).

2 configs x 14 tasks x 3 seeds = 84 jobs.
"""

import copy
from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_v2_backbone_capacity")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_v2_backbone_capacity"

SKIP_DIRS = {"overfit", "1d_padded_fill", "1d_recolor_cnt", "1d_recolor_oe", "1d_recolor_cmp"}

DIM = 10
HEADS = 1

MUON_PARAMS = {
    "optimizer": "Muon",
    "muon_lr": 0.005,
    "muon_momentum": 0.95,
}

CANON_PARAMS = {
    "canon_set": "ABCD",
    "canon_kernel": 5,
    "canon_activation": True,
    "canon_residual": True,
    "canon_causal": False,
}


def c1() -> dict:
    """Canon only, sin PE, N_sup=1 - T3's architecture (canon_transformer,
    num_layers=3) at N_sup=1 instead of 2."""
    return {
        "model": "direct_supervised",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "task_encoding": {"embedding_dim": DIM, "value_vocab_size": 11, "use_sinusoidal_pe": True},
        "backbone_model": {
            "name": "canon_transformer",
            "params": {"hidden_dim": DIM, "num_layers": 3, "num_heads": HEADS, "dropout": 0.1, **CANON_PARAMS},
        },
    }


def rc1() -> dict:
    """RoPE + Canon, N_sup=1 - T4's architecture (rope_canon_looped_transformer,
    n_loops=1) at N_sup=1 instead of 2."""
    return {
        "model": "direct_supervised",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "task_encoding": {"embedding_dim": DIM, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_looped_transformer",
            "params": {
                "hidden_dim": DIM,
                "num_heads": HEADS,
                "inner_dim": DIM,
                "inner_num_heads": HEADS,
                "n_loops": 1,
                "dropout": 0.1,
                **CANON_PARAMS,
                "use_block_skip": False,
                "use_loop_skip": False,
            },
        },
    }


STEP_BUILDERS = {"C1": c1, "RC1": rc1}
STEP_SUFFIX = {"C1": "canon_n1", "RC1": "rope_canon_n1"}

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

    for step, builder in STEP_BUILDERS.items():
        suffix = STEP_SUFFIX[step]
        exp_name = f"v2_{step}_dim{DIM}_{task}_{suffix}"
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": exp_name,
            "project_name": PROJECT,
            "task_categories": task_categories,
            "save_checkpoints": False,
            **copy.deepcopy(builder()),
        }
        out_path = out_task_dir / f"{step}_dim{DIM}_{suffix}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/ (x3 seeds at launch = {n_written * 3} jobs)")

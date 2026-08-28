"""Generate configs for arc1d_v2_backbone_capacity (Round 2, Phase 1).

Round 2's backbone progression demo: re-derives the T1-T7 story from
arc1d_uniform_ablation under Muon (fixed muon_lr=0.005, muon_momentum=0.95,
per docs/arc1d_story/05_round2_plan.md's Phase 0.5 validation) instead of
RAdam, on the 15-task working set (arc1d_uniform_ablation's 17 minus the two
structurally-dead recolor tasks, 1d_recolor_cnt/1d_recolor_oe, which no
configuration in the whole historical chain ever solved), at dim=16 only
(the width actually carried forward everywhere downstream). No L1-L6 sweep:
n_loops=4 is fixed by precedent for this round (see 05_round2_plan.md),
not re-searched.

  T1: Vanilla transformer, sin PE, N_sup=1        (direct_supervised)
  T2: + N_sup=2 loop training
  T3: + Canon ABCD
  T4: + RoPE, flat (n_loops=1)
  T5: + Looped middle (n_loops=4)
  T6: + Block skip
  T7: + Per-loop h0 (loop skip)

Same head_dim=8 convention (dim=16 -> 2 heads), same total-compute-matched
max_steps (T1 N_sup=1 -> max_steps=8000; T2-T7 N_sup=2 -> max_steps=4000),
same learning_rate=0.0005 legacy field left in place (unused by Muon, kept
only because the AdamW aux group in the direct-training path reads
learning_rate/weight_decay from the base config, not from this override).

Seeds are not baked in here (matches the repo-wide convention) - the run
script loops seed=1,2,3 via train.py's seed=N CLI override.

15 tasks x 7 steps = 105 configs. At 3 seeds/config (looped by the run
script, not generated here), that's 315 jobs total.
"""

import copy
from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_v2_backbone_capacity")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_v2_backbone_capacity"

# 1d_padded_fill: excluded from this story family by established convention.
# 1d_recolor_cnt/1d_recolor_oe: structurally dead - no configuration in the
# entire historical chain ever solved these (0.000 exact match everywhere
# tried), verified in docs/arc1d_story/01_data.md. Excluding them from Round
# 2's working set entirely rather than spending compute re-confirming a
# known result.
SKIP_DIRS = {"overfit", "1d_padded_fill", "1d_recolor_cnt", "1d_recolor_oe"}

DIM = 16
HEADS = 2  # head_dim=8 convention

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


def step1() -> dict:
    """Vanilla transformer, sin PE, N_sup=1. Muon has nothing to exclude here
    (no LoRA, no task-identity head) beyond looped_supervised_lightning's own
    fixed 'head' exclusion - just set optimizer/muon_lr/muon_momentum."""
    return {
        "model": "direct_supervised",
        "gradient_clip_val": 1.0,
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "max_steps": 8000,
        "task_encoding": {
            "embedding_dim": DIM,
            "value_vocab_size": 11,
            "use_sinusoidal_pe": True,
        },
        "backbone_model": {
            "name": "transformer",
            "params": {"hidden_dim": DIM, "num_layers": 3, "num_heads": HEADS, "dropout": 0.1},
        },
    }


def step2() -> dict:
    """+ N_sup=2 loop training."""
    cfg = step1()
    del cfg["gradient_clip_val"]
    cfg["model"] = "looped_supervised"
    cfg["N_supervision"] = 2
    cfg["max_steps"] = 4000
    return cfg


def step3() -> dict:
    """+ Canon ABCD."""
    cfg = step2()
    cfg["backbone_model"] = {
        "name": "canon_transformer",
        "params": {"hidden_dim": DIM, "num_layers": 3, "num_heads": HEADS, "dropout": 0.1, **CANON_PARAMS},
    }
    return cfg


def step4() -> dict:
    """+ RoPE, flat (n_loops=1)."""
    cfg = step3()
    cfg["task_encoding"]["use_sinusoidal_pe"] = False
    cfg["backbone_model"] = {
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
    }
    return cfg


def step5() -> dict:
    """+ Looped middle (n_loops=4) - the Round 2 fixed-by-precedent value."""
    cfg = step4()
    cfg["backbone_model"]["params"]["n_loops"] = 4
    return cfg


def step6() -> dict:
    """+ Block skip."""
    cfg = step5()
    cfg["backbone_model"]["params"]["use_block_skip"] = True
    return cfg


def step7() -> dict:
    """+ Per-loop h0 (loop skip)."""
    cfg = step6()
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

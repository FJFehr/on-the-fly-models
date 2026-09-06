"""Generate configs for arc1d_v2_backbone_capacity (Round 2, Phase 1).

Round 2's backbone progression demo: re-derives the T1-T5 story from
arc1d_uniform_ablation under Muon (fixed muon_lr=0.005, muon_momentum=0.95,
per docs/arc1d_story/05_round2_plan.md's Phase 0.5 validation) instead of
RAdam, on the 15-task working set (arc1d_uniform_ablation's 17 minus the two
structurally-dead recolor tasks, 1d_recolor_cnt/1d_recolor_oe, which no
configuration in the whole historical chain ever solved).

  T1: Vanilla transformer, sin PE, N_sup=1        (direct_supervised)
  T2: + N_sup=2 loop training
  T3: + Canon ABCD
  T4: + RoPE, flat (n_loops=1)
  T5: + Looped middle (n_loops=4)

T6 (+ block skip) and T7 (+ per-loop h0/loop skip) are dropped entirely -
n_loops=4 with no skip is fixed by precedent for the whole round (see
05_round2_plan.md), so skip variants are never used by any downstream phase.
Keeping them in this story only tests a question already settled elsewhere.

Two widths, both run at the full T1-T5 sequence:

  dim=16 (head_dim=8, 2 heads): 11,760 backbone params. The width actually
    carried forward by every downstream Round 2 phase.
  dim=10 (head_dim=10, 1 head): 5,130 backbone params - close to the ~5k
    scale used in the older arc1d_capacity_* family. Added because dim=16's
    T1-T5 results turned out uninformative: 14 of 15 tasks already sit at
    1.0 exact match from T1 onward under Muon (see
    docs/arc1d_story/05_round2_plan.md's Phase 1 write-up) - the model is
    already overkill for nearly every task at this size, so the T1-T5
    progression shows almost no separation between steps. A smaller model
    is more likely to actually need the later steps (Canon, RoPE, looping)
    to solve these tasks, which would make for a real progression story
    instead of a flat one. dim=10/heads=1 is the closest even-head_dim
    size to 5k params (head_dim must be even for RoPE's rotation pairs;
    dim=8 -> 3,512 and dim=12 -> 7,044 bracket it, dim=10 -> 5,130 is the
    tightest fit).

Note dim=10 breaks the head_dim=8 convention used at dim=16/32 in the
original ablation (head_dim=10 here) - that convention was a heuristic for
matching param counts across widths in a wider sweep, not a hard constraint,
and doesn't apply when the target is a specific absolute param count instead.

Same total-compute-matched max_steps (T1 N_sup=1 -> max_steps=8000; T2-T5
N_sup=2 -> max_steps=4000), same learning_rate=0.0005 legacy field left in
place (unused by Muon, kept only because the AdamW aux group in the
direct-training path reads learning_rate/weight_decay from the base config,
not from this override).

Seeds are not baked in here (matches the repo-wide convention) - the run
script loops seed=1,2,3 via train.py's seed=N CLI override.

15 tasks x 5 steps x 2 widths = 150 configs. At 3 seeds/config (looped by
the run script, not generated here), that's 450 jobs total (225 already run
at dim=16 under the earlier T1-T7 generator - this script's dim=16 output is
identical for T1-T5, so the run script's skip-if-done logic leaves those
alone and only the new dim=10 jobs, 225 of them, actually run).
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

# dim -> num_heads. head_dim (dim/heads) must be even (RoPE constraint).
DIMS = {16: 2, 10: 1}

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


def step1(dim: int, heads: int) -> dict:
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
        "name": "rope_canon_looped_transformer",
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
    """+ Looped middle (n_loops=4) - the Round 2 fixed-by-precedent value."""
    cfg = step4(dim, heads)
    cfg["backbone_model"]["params"]["n_loops"] = 4
    return cfg


STEP_BUILDERS = {
    "T1": step1,
    "T2": step2,
    "T3": step3,
    "T4": step4,
    "T5": step5,
}

STEP_SUFFIX = {
    "T1": "vanilla",
    "T2": "nsup",
    "T3": "canon",
    "T4": "rope_flat",
    "T5": "looped",
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
            exp_name = f"v2_{step}_dim{dim}_{task}_{suffix}"
            cfg = {
                "_base_": BASE_CFG,
                "experiment_name": exp_name,
                "project_name": PROJECT,
                "task_categories": task_categories,
                "save_checkpoints": False,
                **copy.deepcopy(builder(dim, heads)),
            }
            out_path = out_task_dir / f"{step}_dim{dim}_{suffix}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            n_written += 1

print(f"Written {n_written} configs to {DST}/ (x3 seeds at launch = {n_written * 3} jobs)")

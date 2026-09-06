"""Loop-count / skip diagnostic for arc1d_v2_backbone_capacity, dim=10 only.

Follow-up to the ordering ablation, on the now-adopted "RoPE before Canon"
progression (T1 vanilla -> T2 + N_sup=2 -> R3 + RoPE -> T4 + Canon -> T5 +
looped, n_loops=4). All five configs here vary only the final step's
n_loops/skip flags on top of that same RoPE+Canon backbone (T4's state) -
T5 itself (n_loops=4, no skip) is the reference point, already run.

Mirrors arc1d_uniform_ablation's L1-L6 loop diagnostic (dim=16/32, RAdam,
pre-Muon-fix) - never run at this scale (dim=10, ~5k params) or under
Muon. Historical finding at dim=16/32: no-skip sweep peaked around
n_loops=8-16, degraded by 32; loop-skip-alone never beat no-skip; combined
block+loop skip only helped at dim=32/n_loops=4.

  L8:  n_loops=8,  no skip   - does one more doubling help?
  L16: n_loops=16, no skip   - historical peak was around here
  L32: n_loops=32, no skip   - historical finding: degrades by here

Full skip factorial at n_loops in {4, 8} (T5/L8 are the "no skip" corner
at each, already run) - historical finding at dim=16/32 was that loop-skip
alone never beat no-skip, and combined block+loop skip only helped at
dim=32/n_loops=4, so block-skip-alone was never isolated before:
  B4: n_loops=4, block skip only (no loop skip)
  B8: n_loops=8, block skip only (no loop skip)
  P4: n_loops=4, loop skip only (no block skip)  - "P" for per-loop h0
  P8: n_loops=8, loop skip only (no block skip)
  S4: n_loops=4, block+loop skip
  S8: n_loops=8, block+loop skip

max_steps=4000/N_supervision=2 held fixed regardless of n_loops (wall-clock
scales with n_loops, the step budget doesn't - matches the historical
convention).

9 configs x 15 tasks x 3 seeds = 405 jobs.
"""

import copy
from pathlib import Path

import yaml

SRC_TASKS = Path("configs/experiments/arc1d_recursion_ablation")
DST = Path("configs/experiments/arc1d_v2_backbone_capacity")
BASE_CFG = "configs/experiments/arc1d_recursion_ablation/base_ablation.yaml"
PROJECT = "arc1d_v2_backbone_capacity"

SKIP_DIRS = {"overfit", "1d_padded_fill", "1d_recolor_cnt", "1d_recolor_oe"}

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


def _variant(n_loops: int, use_block_skip: bool, use_loop_skip: bool) -> dict:
    return {
        "model": "looped_supervised",
        **MUON_PARAMS,
        "learning_rate": 0.0005,
        "N_supervision": 2,
        "max_steps": 4000,
        "task_encoding": {"embedding_dim": DIM, "value_vocab_size": 11, "use_sinusoidal_pe": False},
        "backbone_model": {
            "name": "rope_canon_looped_transformer",
            "params": {
                "hidden_dim": DIM,
                "num_heads": HEADS,
                "inner_dim": DIM,
                "inner_num_heads": HEADS,
                "n_loops": n_loops,
                "dropout": 0.1,
                **CANON_PARAMS,
                "use_block_skip": use_block_skip,
                "use_loop_skip": use_loop_skip,
            },
        },
    }


STEP_BUILDERS = {
    "L8": lambda: _variant(8, False, False),
    "L16": lambda: _variant(16, False, False),
    "L32": lambda: _variant(32, False, False),
    "B4": lambda: _variant(4, True, False),
    "B8": lambda: _variant(8, True, False),
    "P4": lambda: _variant(4, False, True),
    "P8": lambda: _variant(8, False, True),
    "S4": lambda: _variant(4, True, True),
    "S8": lambda: _variant(8, True, True),
}

STEP_SUFFIX = {
    "L8": "n8_noskip",
    "L16": "n16_noskip",
    "L32": "n32_noskip",
    "B4": "n4_block_skip",
    "B8": "n8_block_skip",
    "P4": "n4_loop_skip",
    "P8": "n8_loop_skip",
    "S4": "n4_both_skip",
    "S8": "n8_both_skip",
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

"""Generate the move-ablation follow-up grid for
arc1d_hypermodel_looped_rope_canon_muon_moveablation.

Follow-up to the 216-cell arc1d_hypermodel_looped_rope_canon_muon_sweep (99 cells completed
before being cancelled): locks in muon_lr=0.02 and muon_exclude_lora_heads=true (both fixed, see
base.yaml), and crosses three axes:

- weight_decay: {0.01, 0.1}.
- adam_lr (the AdamW aux group's learning_rate): {3e-4, 6e-4}.
- moves: "with" (standard 15 task categories) vs "without" (13 categories, dropping
  1d_move_1p/1d_move_2p entirely from both task_categories and val_task_categories) -- tests
  whether removing the notd move-family interference source (see base.yaml) helps.

batch_size is fixed at 2048 (base.yaml), not swept: bsz=4096 was tried and genuinely OOMs
(44.21/44.40 GiB on a single GPU, reproducible across every other axis combination) -- the
hypernetwork generates per-example target-model weights, so activation memory scales with
batch_size much faster than a normal transformer's. Dropped rather than chasing a memory fix.

2 x 2 x 2 = 8 configs, 3 seeds each (seed handled by the runner script, not baked in) = 24 jobs.
"moves" is the outermost loop so CELL_GLOB="arm_bsz2048_*_with.yaml" /
CELL_GLOB="arm_bsz2048_*_without.yaml" splits the grid into two even 4-cell halves, one per node.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_moveablation")
BASE_CFG = "configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_moveablation/base.yaml"

BATCH_SIZE = [2048]
WEIGHT_DECAY = [0.01, 0.1]
ADAM_LR = [3e-4, 6e-4]
MOVES = ["with", "without"]

CATEGORIES_WITH_MOVES = [
    "1d_denoising_1c",
    "1d_denoising_mc",
    "1d_fill",
    "1d_flip",
    "1d_hollow",
    "1d_mirror",
    "1d_move_1p",
    "1d_move_2p",
    "1d_move_2p_dp",
    "1d_move_3p",
    "1d_move_dp",
    "1d_pcopy_1c",
    "1d_pcopy_mc",
    "1d_recolor_cmp",
    "1d_scale_dp",
]
CATEGORIES_WITHOUT_MOVES = [
    c for c in CATEGORIES_WITH_MOVES if c not in ("1d_move_1p", "1d_move_2p")
]


def _fmt(x: float) -> str:
    return f"{x:g}"


n_written = 0
for bsz in BATCH_SIZE:
    for wd in WEIGHT_DECAY:
        for adam_lr in ADAM_LR:
            for moves in MOVES:
                name = f"arm_bsz{bsz}_wd{_fmt(wd)}_alr{_fmt(adam_lr)}_{moves}"
                cfg = {
                    "_base_": BASE_CFG,
                    "experiment_name": name,
                    "batch_size": bsz,
                    "weight_decay": wd,
                    "learning_rate": adam_lr,
                }
                if moves == "without":
                    cfg["task_categories"] = CATEGORIES_WITHOUT_MOVES
                    cfg["val_task_categories"] = CATEGORIES_WITHOUT_MOVES
                out_path = DST / f"{name}.yaml"
                with open(out_path, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                n_written += 1

print(f"Written {n_written} configs to {DST}/")

"""Generate the full-factorial Muon hyperparameter/architecture sweep for
arc1d_hypermodel_looped_rope_canon_muon_sweep.

Follow-up to arc1d_hypermodel_looped_rope_canon_muon's arm_notd_muon/arm_frozentd_muon, which
used an untuned muon_lr=0.005 (see that experiment's README.md) alongside the AdamW arms'
learning_rate/weight_decay/batch_size. This crosses 5 axes, notd only:

- muon_lr: Muon's own learning rate.
- learning_rate: the AdamW aux group's learning rate (embeddings/output_head/task_indicator_proj/
  norms/biases, and lora_proj_a/lora_proj_b when excluded from Muon -- see below).
- weight_decay: shared by both the Muon and AdamW-aux groups (matching this model's existing
  convention of one weight_decay value across both).
- batch_size: current baseline (512) up to 4x (2048).
- muon_exclude_lora_heads: whether the LoRA head's A/B factor projections
  (lora_proj_a/lora_proj_b) are Muon-eligible (False, current default) or routed to AdamW
  (True) -- Fabio's hypothesis that Muon's orthogonalization may be a poor fit for these
  specifically.

4 x 3 x 3 x 3 x 2 = 216 configs, 1 seed each (seed handled by the runner script, not baked in).
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_sweep")
BASE_CFG = "configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_sweep/base.yaml"

MUON_LR = [0.008, 0.01, 0.016, 0.02]
ADAM_LR = [3e-4, 6e-4, 3e-3]
WEIGHT_DECAY = [0.0, 0.01, 0.1]
BATCH_SIZE = [512, 1024, 2048]
LORA_EXCLUDE = [False, True]


def _fmt(x: float) -> str:
    """Filesystem-safe encoding of a float for filenames, e.g. 0.0003 -> '0.0003'."""
    return f"{x:g}"


n_written = 0
for muon_lr in MUON_LR:
    for adam_lr in ADAM_LR:
        for wd in WEIGHT_DECAY:
            for bsz in BATCH_SIZE:
                for lora_exclude in LORA_EXCLUDE:
                    lora_tag = "loraexcl" if lora_exclude else "loraincl"
                    name = (
                        f"arm_mlr{_fmt(muon_lr)}_alr{_fmt(adam_lr)}"
                        f"_wd{_fmt(wd)}_bsz{bsz}_{lora_tag}"
                    )
                    cfg = {
                        "_base_": BASE_CFG,
                        "experiment_name": name,
                        "muon_lr": muon_lr,
                        "learning_rate": adam_lr,
                        "weight_decay": wd,
                        "batch_size": bsz,
                        "muon_exclude_lora_heads": lora_exclude,
                    }
                    out_path = DST / f"{name}.yaml"
                    with open(out_path, "w") as f:
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                    n_written += 1

print(f"Written {n_written} configs to {DST}/")

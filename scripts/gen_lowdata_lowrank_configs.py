"""Generate the rank x data-reduction grid for arc1d_lowdata_lowrank.

Follow-on from arc1d_lowdata: that experiment found variants_per_base_task=2 already
reaches full performance at the default lora_adapter_rank=8. This crosses lower LoRA
ranks against the same low data levels to see whether rank becomes the bottleneck once
data is this scarce.

lora_adapter_rank in {1, 2, 4} x variants_per_base_task in {1, 2, 3}

rank=8 is deliberately excluded -- it's already covered by arc1d_lowdata's
cell_v1/cell_v2/cell_v3 results, which serve as this experiment's rank=8 reference
column. Data sampling is the same stratified/nested mechanism as arc1d_lowdata (see
data_modules/arc1d_meta_multiclass.py's _stratified_variants_per_base_task); data_seed is
fixed in base.yaml so all cells at a given variants_per_base_task level see identical
training data regardless of rank.

3 ranks x 3 data levels x 3 seeds (seed handled by the runner script, not baked into
configs) = 27 jobs.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_lowdata_lowrank")
BASE_CFG = "configs/experiments/arc1d_lowdata_lowrank/base.yaml"
PROJECT = "arc1d_lowdata_lowrank"

LORA_ADAPTER_RANKS = [4, 2, 1]
VARIANTS_PER_BASE_TASK = [1, 2, 3]

n_written = 0
for rank in LORA_ADAPTER_RANKS:
    for v in VARIANTS_PER_BASE_TASK:
        cfg = {
            "_base_": BASE_CFG,
            "experiment_name": f"lowdata_lowrank_r{rank}_v{v}",
            "variants_per_base_task": v,
            "hyper_head": {"lora_adapter_rank": rank},
        }
        out_path = DST / f"cell_r{rank}_v{v}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        n_written += 1

print(f"Written {n_written} configs to {DST}/")

"""Generate the data-reduction levels for arc1d_lowdata.

Tests whether the hypernetwork needs less training data than a model with no
cross-task transfer, by training the fixed frozen_td/Muon architecture from
arc1d_hypermodel_looped_rope_canon_muon at progressively less training data
per base task, stratified and nested:

variants_per_base_task in {1, 2, 3, 4, 5, 20}

Sampling is stratified per underlying base task (not a raw random count over
the pool) and nested/cumulative: level K always includes the true original
(aug_index=0, unaugmented) example for every base task first, then K-1 more
augmented variants in a fixed data_seed order, so level K's training set is
always a subset of level K+1's. See
data_modules/arc1d_meta_multiclass.py's _stratified_variants_per_base_task.

6 levels x 3 seeds (seed handled by the runner script, not baked into
configs) = 18 jobs. Validation/test are untouched at every level (see
base.yaml) - only train_dataset shrinks.
"""

from pathlib import Path

import yaml

DST = Path("configs/experiments/arc1d_lowdata")
BASE_CFG = "configs/experiments/arc1d_lowdata/base.yaml"
PROJECT = "arc1d_lowdata"

VARIANTS_PER_BASE_TASK = [1, 2, 3, 4, 5, 20]

n_written = 0
for v in VARIANTS_PER_BASE_TASK:
    cfg = {
        "_base_": BASE_CFG,
        "experiment_name": f"lowdata_v{v}",
        "variants_per_base_task": v,
    }
    out_path = DST / f"cell_v{v}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    n_written += 1

print(f"Written {n_written} configs to {DST}/")

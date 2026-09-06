# data_modules/__init__.py
# ---------------------------------------------------------------------------
# Data module registry for config-driven experiment selection.
#
# Each entry maps a short string key (used in YAML config files under the
# "data" field) to the corresponding PyTorch Lightning data module class.
#
# To add a new data module:
#   1. Create the LightningDataModule class in a new file under data_modules/
#   2. Import it here
#   3. Add a single entry to DATA_REGISTRY
#   4. No changes to train.py are needed — it looks up data modules by key.
#
# arc_1d_meta_simple / arc_1d_meta_padded_multiclass (Arc1dMetaSimpleDataModule /
# Arc1dMetaPaddedMulticlassDataModule) are no longer registered here -- each
# was used by exactly one legacy config family (arc1d_binary / arc1d_multiclass,
# see legacy/configs/experiments/), not by any of the paper experiments under
# experiments/. Their code moved to legacy/data_modules/ for
# provenance; to rerun those legacy configs, re-register them here first.
# ---------------------------------------------------------------------------

from data_modules.arc1d_direct import Arc1dDirectDataModule
from data_modules.arc1d_meta_multiclass import Arc1dMetaMulticlassDataModule

# Maps config string -> LightningDataModule class
# The key must match the "data" field in experiment YAML configs
DATA_REGISTRY = {
    "arc_1d_meta_multiclass": Arc1dMetaMulticlassDataModule,
    "arc_1d_direct": Arc1dDirectDataModule,
}

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
# ---------------------------------------------------------------------------

from data_modules.arc1d_direct import Arc1dDirectDataModule
from data_modules.arc1d_meta_multiclass import Arc1dMetaMulticlassDataModule

# Maps config string -> LightningDataModule class
# The key must match the "data" field in experiment YAML configs
DATA_REGISTRY = {
    "arc_1d_meta_multiclass": Arc1dMetaMulticlassDataModule,
    "arc_1d_direct": Arc1dDirectDataModule,
}

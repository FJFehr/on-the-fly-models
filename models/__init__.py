# models/__init__.py
# ---------------------------------------------------------------------------
# Model registry for config-driven experiment selection.
#
# Each entry maps a short string key (used in YAML config files under the
# "model" field) to the corresponding PyTorch Lightning module class.
#
# Target models live in models/target_models/ and all inherit from
# BaseTargetModel (models/target_models/base.py), which handles training
# loops, loss, metrics, optimizer, and WandB logging. Concrete model files
# only define the architecture and forward pass.
#
# To add a new target model:
#   1. Create a new file under models/target_models/ that subclasses BaseTargetModel
#   2. Implement __init__ (build self.model) and forward(inputs)
#   3. Import it here and add a single entry to MODEL_REGISTRY
#   4. No changes to train.py are needed — it looks up models by key.
# ---------------------------------------------------------------------------

from models.target_models.cnn import TargetCNNModelLightning
from models.target_models.mlp import TargetModelLightning
from models.target_models.positional_cnn import TargetPositionalCNNModelLightning
from models.target_models.positional_mlp import TargetPositionalMLPLightning
from models.target_models.positional_rnn import TargetPositionalRNNModelLightning

# Maps config string -> LightningModule class
# The key must match the "model" field in experiment YAML configs
MODEL_REGISTRY = {
    "mlp": TargetModelLightning,
    "cnn": TargetCNNModelLightning,
    "positional_cnn": TargetPositionalCNNModelLightning,
    "positional_mlp": TargetPositionalMLPLightning,
    "positional_rnn": TargetPositionalRNNModelLightning,
}

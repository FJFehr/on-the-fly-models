# models/target_models/__init__.py
# ---------------------------------------------------------------------------
# Re-exports for the target model subpackage.
#
# All target models inherit from BaseTargetModel (base.py) and only need
# to define their architecture (__init__) and forward pass (forward).
# ---------------------------------------------------------------------------

from models.target_models.cnn import TargetCNNModelLightning
from models.target_models.mlp import TargetModelLightning
from models.target_models.positional_cnn import TargetPositionalCNNModelLightning
from models.target_models.positional_mlp import TargetPositionalMLPLightning
from models.target_models.positional_rnn import TargetPositionalRNNModelLightning

__all__ = [
    "BaseTargetModel",
    "TargetCNNModelLightning",
    "TargetModelLightning",
    "TargetPositionalCNNModelLightning",
    "TargetPositionalMLPLightning",
    "TargetPositionalRNNModelLightning",
]

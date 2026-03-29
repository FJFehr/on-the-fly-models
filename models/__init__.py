"""Model registry for config-driven experiment selection."""

from models.target_models.cnn import TargetCNNModelLightning
from models.target_models.mlp import TargetMLPModelLightning
from models.target_models.rnn import TargetRNNModelLightning
from models.target_models.transformer import TargetTransformerModelLightning


MODEL_REGISTRY = {
    "mlp": TargetMLPModelLightning,
    "cnn": TargetCNNModelLightning,
    "rnn": TargetRNNModelLightning,
    "transformer": TargetTransformerModelLightning,
}

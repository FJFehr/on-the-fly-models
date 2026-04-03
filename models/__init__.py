"""Model registry for config-driven experiment selection."""

from models.hypermodel_lightning import HyperModelLightning
from models.hypermodels import (
    BinaryHyperRNNMetaModelLightning,
    HyperCNNMetaModelLightning,
    HyperRNNMetaModelLightning,
)
from models.target_models.cnn import TargetCNNModelLightning
from models.target_models.deepset import TargetDeepSetModelLightning
from models.target_models.mlp import TargetMLPModelLightning
from models.target_models.rnn import TargetRNNModelLightning
from models.target_models.transformer import TargetTransformerModelLightning

MODEL_REGISTRY = {
    "mlp": TargetMLPModelLightning,
    "cnn": TargetCNNModelLightning,
    "deepset": TargetDeepSetModelLightning,
    "rnn": TargetRNNModelLightning,
    "transformer": TargetTransformerModelLightning,
    "hyper_cnn": HyperCNNMetaModelLightning,
    "hyper_rnn": HyperRNNMetaModelLightning,
    "binary_hyper_rnn": BinaryHyperRNNMetaModelLightning,
    # Simplified binary hypermodel architecture with one generated weight set per task.
    "hyper_model": HyperModelLightning,
    "binary_hyper_model": HyperModelLightning,
}

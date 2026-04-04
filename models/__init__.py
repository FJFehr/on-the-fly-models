"""Model registry for config-driven experiment selection."""

from models.hypermodel_lightning import HyperModelLightning

MODEL_REGISTRY = {
    "hyper_model": HyperModelLightning,
    "binary_hyper_model": HyperModelLightning,
}

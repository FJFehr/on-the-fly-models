"""Model registry for config-driven experiment selection."""

from models.direct_supervised_lightning import DirectSupervisedLightning
from models.hypermodel_lightning import HyperModelLightning
from models.looped_supervised_lightning import LoopedSupervisedLightning

MODEL_REGISTRY = {
    "hyper_model": HyperModelLightning,
    "binary_hyper_model": HyperModelLightning,
    "direct_supervised": DirectSupervisedLightning,
    "looped_supervised": LoopedSupervisedLightning,
}

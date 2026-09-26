"""Lightning modules, selected by the `model:` key of an experiment config."""

from lightning_modules.direct import DirectLightning
from lightning_modules.hypernetwork import HypernetworkLightning

MODEL_REGISTRY = {
    "direct": DirectLightning,
    "hypernetwork": HypernetworkLightning,
}

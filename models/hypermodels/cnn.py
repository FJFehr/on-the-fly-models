"""Task-conditioned hypernetwork that predicts a target CNN."""

from models.hypermodels.base import BaseHyperMetaModelLightning
from models.target_models.cnn_core import TargetCNNCore


class HyperCNNMetaModelLightning(BaseHyperMetaModelLightning):
    """Task-conditioned hypernetwork that predicts a task-specific target CNN."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_classes: int,
        task_encoder_hidden_dim: int,
        task_encoder_num_heads: int,
        task_encoder_num_layers: int,
        target_cnn_hidden_channels: int,
        target_cnn_kernel_size: int = 3,
        target_cnn_num_layers: int = 1,
        target_cnn_use_skip_connections: bool = False,
        **kwargs,
    ):
        target_model_template = TargetCNNCore(
            num_classes=num_classes,
            hidden_channels=target_cnn_hidden_channels,
            kernel_size=target_cnn_kernel_size,
            num_layers=target_cnn_num_layers,
            sequence_length=input_dim,
            use_skip_connections=target_cnn_use_skip_connections,
            use_positional_feature=True,
        )
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            task_encoder_hidden_dim=task_encoder_hidden_dim,
            task_encoder_num_heads=task_encoder_num_heads,
            task_encoder_num_layers=task_encoder_num_layers,
            target_model_template=target_model_template,
            **kwargs,
        )


__all__ = ["HyperCNNMetaModelLightning"]

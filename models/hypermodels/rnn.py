"""Task-conditioned hypernetwork that predicts a target RNN."""

from models.hypermodels.base import BaseHyperMetaModelLightning
from models.target_models.rnn_core import TargetRNNCore


class HyperRNNMetaModelLightning(BaseHyperMetaModelLightning):
    """Task-conditioned hypernetwork that predicts a task-specific target RNN."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_classes: int,
        task_encoder_hidden_dim: int,
        task_encoder_num_heads: int,
        task_encoder_num_layers: int,
        target_rnn_hidden_dim: int,
        target_rnn_bidirectional: bool = True,
        target_rnn_num_layers: int = 1,
        **kwargs,
    ):
        target_model_template = TargetRNNCore(
            num_classes=num_classes,
            hidden_dim=target_rnn_hidden_dim,
            bidirectional=target_rnn_bidirectional,
            num_layers=target_rnn_num_layers,
            use_positional_feature=False,
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


__all__ = ["HyperRNNMetaModelLightning"]

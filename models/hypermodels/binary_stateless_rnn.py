"""Binary hypernetwork with a stateless target RNN."""

import os
from collections import OrderedDict
from math import prod

import lightning as pl
import torch
import torch.nn.functional as F

import wandb
from metrics import accuracy
from models.target_models.transformer import Block, LayerNorm, TransformerConfig
from visualisation import (
    figure_to_wandb_image,
    render_task_attention_figure,
    render_task_prediction_figure,
    resolve_attention_matrix,
)

TASK_SEGMENT_SHORT_NAMES = (
    "s1_in",
    "s1_out",
    "s2_in",
    "s2_out",
    "s3_in",
    "s3_out",
    "q_in",
)

TASK_SEGMENT_DISPLAY_NAMES = (
    "Support 1 Input",
    "Support 1 Output",
    "Support 2 Input",
    "Support 2 Output",
    "Support 3 Input",
    "Support 3 Output",
    "Query Input",
)


def serialise_binary_task_segments(
    support_inputs: torch.Tensor,
    support_outputs: torch.Tensor,
    query_input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten a binary ARC task into the fixed legacy task-segment order."""
    segment_values = torch.stack(
        [
            support_inputs[:, 0],
            support_outputs[:, 0],
            support_inputs[:, 1],
            support_outputs[:, 1],
            support_inputs[:, 2],
            support_outputs[:, 2],
            query_input,
        ],
        dim=1,
    )
    example_ids = support_inputs.new_tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.float32)
    role_ids = support_inputs.new_tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    is_query_ids = support_inputs.new_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    return segment_values, example_ids, role_ids, is_query_ids


def build_binary_task_features(
    support_inputs: torch.Tensor,
    support_outputs: torch.Tensor,
    query_input: torch.Tensor,
) -> torch.Tensor:
    """Build the 5 scalar features used by the legacy binary task encoder."""
    segment_values, example_ids, role_ids, is_query_ids = serialise_binary_task_segments(
        support_inputs,
        support_outputs,
        query_input,
    )
    batch_size, _, sequence_length = segment_values.shape
    flat_values = segment_values.reshape(batch_size, -1, 1).float()

    normalized_positions = torch.linspace(
        0.0,
        1.0,
        steps=sequence_length,
        device=segment_values.device,
    )
    position_feature = normalized_positions.repeat(7).view(1, -1, 1)
    example_feature = (example_ids / 3.0).repeat_interleave(sequence_length).view(1, -1, 1)
    role_feature = role_ids.repeat_interleave(sequence_length).view(1, -1, 1)
    is_query_feature = is_query_ids.repeat_interleave(sequence_length).view(1, -1, 1)

    return torch.cat(
        [
            flat_values,
            position_feature.expand(batch_size, -1, -1),
            example_feature.expand(batch_size, -1, -1),
            role_feature.expand(batch_size, -1, -1),
            is_query_feature.expand(batch_size, -1, -1),
        ],
        dim=-1,
    )


def build_binary_targets(
    support_outputs: torch.Tensor,
    query_output: torch.Tensor,
) -> torch.Tensor:
    """Stack support and query outputs into the shared 4-example layout."""
    return torch.cat([support_outputs.float(), query_output.float().unsqueeze(1)], dim=1)


def compute_binary_meta_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_on_support: bool,
    loss_on_query: bool,
) -> torch.Tensor:
    """Match the legacy support/query-masked BCE reduction."""
    per_position_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )
    per_example_loss = per_position_loss.mean(dim=2)

    selected_losses = []
    if loss_on_support:
        selected_losses.append(per_example_loss[:, :3])
    if loss_on_query:
        selected_losses.append(per_example_loss[:, 3:])
    return torch.cat(selected_losses, dim=1).mean()


def decode_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode binary logits with the legacy 0.5 sigmoid threshold."""
    return (torch.sigmoid(logits) >= 0.5).long()


def compute_binary_meta_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the legacy binary support/query metrics."""
    predictions = decode_binary_logits(logits)
    targets_long = targets.long()
    exact_matches = (predictions == targets_long).all(dim=2).float()

    return {
        "support_accuracy": accuracy(
            targets_long[:, :3].reshape(-1, targets_long.shape[-1]),
            predictions[:, :3].reshape(-1, predictions.shape[-1]),
        ),
        "query_accuracy": accuracy(
            targets_long[:, 3:].reshape(-1, targets_long.shape[-1]),
            predictions[:, 3:].reshape(-1, predictions.shape[-1]),
        ),
        "support_exact_match_accuracy": exact_matches[:, :3].mean(),
        "query_exact_match_accuracy": exact_matches[:, 3:].mean(),
        "all_examples_exact_match_accuracy": exact_matches.all(dim=1).float().mean(),
    }


class BinaryTaskFeatureEncoder(torch.nn.Module):
    """Transformer encoder over binary scalar task features."""

    def __init__(
        self,
        input_feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        block_size: int,
        bias: bool = False,
    ):
        super().__init__()
        config = TransformerConfig(
            block_size=block_size,
            input_feature_dim=hidden_dim,
            logit_dim=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_embd=hidden_dim,
            dropout=0.0,
            bias=bias,
            causal=False,
        )
        self.input_projection = torch.nn.Linear(input_feature_dim, hidden_dim, bias=bias)
        self.blocks = torch.nn.ModuleList([Block(config) for _ in range(num_layers)])
        self.ln_f = LayerNorm(hidden_dim, bias=bias)

    def forward(
        self,
        task_features: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.input_projection(task_features)
        attentions = []
        for block in self.blocks:
            if return_attentions:
                hidden_states, layer_attentions = block(
                    hidden_states,
                    return_attention_weights=True,
                )
                attentions.append(layer_attentions)
            else:
                hidden_states = block(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        if return_attentions:
            return hidden_states, attentions
        return hidden_states


class BinaryStatelessTargetRNN(torch.nn.Module):
    """Visible zero-parameter target module with generated stateless weights."""

    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_skip_connections = use_skip_connections
        self.register_buffer(
            "position_ramp",
            torch.linspace(0.0, 1.0, steps=sequence_length, dtype=torch.float32),
        )

    @property
    def input_feature_dim(self) -> int:
        return 2

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    @property
    def output_feature_dim(self) -> int:
        return self.hidden_dim * self.num_directions

    def build_parameter_specs(self) -> list[dict]:
        specs = []
        layer_input_dim = self.input_feature_dim
        directions = ("forward", "backward") if self.bidirectional else ("forward",)

        for layer_index in range(self.num_layers):
            for direction_name in directions:
                specs.append(
                    {
                        "name": f"weight_ih_l{layer_index}_{direction_name}",
                        "shape": (self.hidden_dim, layer_input_dim),
                    }
                )
                specs.append(
                    {
                        "name": f"weight_hh_l{layer_index}_{direction_name}",
                        "shape": (self.hidden_dim, self.hidden_dim),
                    }
                )
            layer_input_dim = self.output_feature_dim

        if self.use_skip_connections:
            specs.append(
                {
                    "name": "input_skip_weight",
                    "shape": (self.output_feature_dim, self.input_feature_dim),
                }
            )

        specs.append({"name": "output_weight", "shape": (1, self.output_feature_dim)})
        return specs

    def build_input_features(self, example_inputs: torch.Tensor) -> torch.Tensor:
        position_features = self.position_ramp.view(1, self.sequence_length, 1).expand(
            example_inputs.shape[0], -1, -1
        )
        return torch.cat([example_inputs.unsqueeze(-1), position_features], dim=-1)

    def run_direction(
        self,
        inputs: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        hidden_state = inputs.new_zeros(inputs.shape[0], self.hidden_dim)
        hidden_states = []
        time_indices = (
            range(self.sequence_length - 1, -1, -1) if reverse else range(self.sequence_length)
        )

        for step in time_indices:
            input_step = inputs[:, step]
            hidden_state = F.relu(
                F.linear(input_step, weight_ih) + F.linear(hidden_state, weight_hh)
            )
            hidden_states.append(hidden_state)

        if reverse:
            hidden_states.reverse()
        return torch.stack(hidden_states, dim=1)

    def forward_with_params(
        self,
        example_inputs: torch.Tensor,
        parameter_mapping: OrderedDict,
    ) -> torch.Tensor:
        base_features = self.build_input_features(example_inputs)
        hidden_stack = base_features

        for layer_index in range(self.num_layers):
            forward_states = self.run_direction(
                hidden_stack,
                parameter_mapping[f"weight_ih_l{layer_index}_forward"],
                parameter_mapping[f"weight_hh_l{layer_index}_forward"],
                reverse=False,
            )
            layer_outputs = [forward_states]

            if self.bidirectional:
                backward_states = self.run_direction(
                    hidden_stack,
                    parameter_mapping[f"weight_ih_l{layer_index}_backward"],
                    parameter_mapping[f"weight_hh_l{layer_index}_backward"],
                    reverse=True,
                )
                layer_outputs.append(backward_states)

            hidden_stack = torch.cat(layer_outputs, dim=-1)

        if self.use_skip_connections:
            hidden_stack = hidden_stack + F.linear(
                base_features,
                parameter_mapping["input_skip_weight"],
            )

        return F.linear(hidden_stack, parameter_mapping["output_weight"]).squeeze(-1)

    def forward(self, example_inputs: torch.Tensor) -> torch.Tensor:
        msg = "BinaryStatelessTargetRNN requires generated parameters via forward_with_params()."
        raise RuntimeError(msg)


class BinaryHyperRNNMetaModelLightning(pl.LightningModule):
    """Binary hypernetwork that predicts a stateless task-specific target RNN."""

    supports_hard_val_examples = False
    supports_task_visualization = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_encoder_hidden_dim: int = 64,
        task_encoder_num_heads: int = 4,
        task_encoder_num_layers: int = 3,
        task_encoder_bias: bool = False,
        target_rnn_hidden_dim: int = 16,
        target_rnn_bidirectional: bool = True,
        target_rnn_num_layers: int = 2,
        target_rnn_use_skip_connections: bool = False,
        learning_rate: float = 0.001,
        optimizer: str = "Adam",
        weight_decay: float = 0.01,
        loss_on_support: bool = True,
        loss_on_query: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])

        if input_dim != output_dim:
            msg = "The binary hypernetwork expects input_dim and output_dim to match."
            raise ValueError(msg)
        if not loss_on_support and not loss_on_query:
            msg = "At least one of loss_on_support or loss_on_query must be enabled."
            raise ValueError(msg)
        if task_encoder_hidden_dim < 1:
            msg = "task_encoder_hidden_dim must be at least 1."
            raise ValueError(msg)
        if task_encoder_num_heads < 1:
            msg = "task_encoder_num_heads must be at least 1."
            raise ValueError(msg)
        if task_encoder_num_layers < 1:
            msg = "task_encoder_num_layers must be at least 1."
            raise ValueError(msg)
        if target_rnn_hidden_dim < 1:
            msg = "target_rnn_hidden_dim must be at least 1."
            raise ValueError(msg)
        if target_rnn_num_layers < 1:
            msg = "target_rnn_num_layers must be at least 1."
            raise ValueError(msg)

        self.sequence_length = input_dim
        self.support_example_count = 3
        self.num_examples = 4
        self.num_segments = 7
        self.task_feature_dim = 5
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.loss_on_support = loss_on_support
        self.loss_on_query = loss_on_query
        self.log_task_examples = kwargs.get("log_task_examples", False)
        self.log_task_examples_every_n_epochs = kwargs.get("log_task_examples_every_n_epochs", 25)
        self.log_task_attention = kwargs.get("log_task_attention", False)
        self.log_task_attention_every_n_epochs = kwargs.get(
            "log_task_attention_every_n_epochs",
            self.log_task_examples_every_n_epochs,
        )
        self.attention_head_reduction = kwargs.get("attention_head_reduction", "mean")
        self.num_periodic_train_task_examples = kwargs.get("num_periodic_train_task_examples", 1)
        self.num_periodic_val_task_examples = kwargs.get("num_periodic_val_task_examples", 1)
        self.num_final_hard_val_task_examples = kwargs.get("num_final_hard_val_task_examples", 3)
        self.selected_representative_task_ids: dict[str, list[int]] = {"train": [], "val": []}

        self.register_buffer(
            "position_ramp",
            torch.linspace(0.0, 1.0, steps=self.sequence_length, dtype=torch.float32),
        )

        self.task_encoder = BinaryTaskFeatureEncoder(
            input_feature_dim=self.task_feature_dim,
            hidden_dim=task_encoder_hidden_dim,
            num_heads=task_encoder_num_heads,
            num_layers=task_encoder_num_layers,
            block_size=self.num_segments * self.sequence_length,
            bias=task_encoder_bias,
        )
        self.target_model = BinaryStatelessTargetRNN(
            sequence_length=self.sequence_length,
            hidden_dim=target_rnn_hidden_dim,
            num_layers=target_rnn_num_layers,
            bidirectional=target_rnn_bidirectional,
            use_skip_connections=target_rnn_use_skip_connections,
        )
        self.target_parameter_specs = self.target_model.build_parameter_specs()
        self.hyper_head = torch.nn.Sequential(
            torch.nn.Linear(
                task_encoder_hidden_dim,
                task_encoder_hidden_dim,
                bias=task_encoder_bias,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                task_encoder_hidden_dim,
                self.target_parameter_count,
                bias=task_encoder_bias,
            ),
        )

    @property
    def target_parameter_count(self) -> int:
        return sum(prod(spec["shape"]) for spec in self.target_parameter_specs)

    def serialise_task_segments(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return serialise_binary_task_segments(
            support_inputs=support_inputs,
            support_outputs=support_outputs,
            query_input=query_input,
        )

    def build_task_features(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> torch.Tensor:
        return build_binary_task_features(
            support_inputs=support_inputs,
            support_outputs=support_outputs,
            query_input=query_input,
        )

    def format_attention_token_label(self, token_metadata: dict) -> str:
        return (
            f"{token_metadata['segment_short_label']}"
            f"|p{token_metadata['position']:02d}"
            f"|v{int(token_metadata['value'])}"
        )

    def build_serialised_token_metadata(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> list[list[dict]]:
        segment_values, example_ids, role_ids, _ = self.serialise_task_segments(
            support_inputs,
            support_outputs,
            query_input,
        )
        task_metadata = []
        for batch_index in range(segment_values.shape[0]):
            serialized_metadata = []
            for segment_index, (segment_short_label, segment_display_label) in enumerate(
                zip(TASK_SEGMENT_SHORT_NAMES, TASK_SEGMENT_DISPLAY_NAMES, strict=True)
            ):
                example_index = int(example_ids[segment_index].item())
                role_name = "input" if int(role_ids[segment_index].item()) == 0 else "output"
                segment_value_list = segment_values[batch_index, segment_index].tolist()
                for position, value in enumerate(segment_value_list):
                    serialized_metadata.append(
                        {
                            "segment_index": segment_index,
                            "segment_short_label": segment_short_label,
                            "segment_display_label": segment_display_label,
                            "example_index": example_index,
                            "role": role_name,
                            "position": position,
                            "value": int(value),
                        }
                    )
            task_metadata.append(serialized_metadata)
        return task_metadata

    def encode_task(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        task_features = self.build_task_features(
            support_inputs=support_inputs,
            support_outputs=support_outputs,
            query_input=query_input,
        )
        if return_attentions:
            encoded_tokens, attentions = self.task_encoder(
                task_features,
                return_attentions=True,
            )
            return encoded_tokens.mean(dim=1), attentions
        encoded_tokens = self.task_encoder(task_features)
        return encoded_tokens.mean(dim=1)

    def predict_target_parameter_vectors(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> torch.Tensor:
        task_representations = self.encode_task(support_inputs, support_outputs, query_input)
        return self.hyper_head(task_representations)

    def get_task_encoder_attention_data(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> dict:
        if support_inputs.dim() == 2:
            support_inputs = support_inputs.unsqueeze(0)
            support_outputs = support_outputs.unsqueeze(0)
            query_input = query_input.unsqueeze(0)

        _, attentions = self.encode_task(
            support_inputs=support_inputs.float(),
            support_outputs=support_outputs.float(),
            query_input=query_input.float(),
            return_attentions=True,
        )
        token_metadata = self.build_serialised_token_metadata(
            support_inputs=support_inputs.float(),
            support_outputs=support_outputs.float(),
            query_input=query_input.float(),
        )
        token_labels = [
            [self.format_attention_token_label(token_info) for token_info in task_tokens]
            for task_tokens in token_metadata
        ]
        return {
            "attentions": [layer_attention.detach().cpu() for layer_attention in attentions],
            "token_metadata": token_metadata,
            "token_labels": token_labels,
        }

    def parameter_vector_to_mapping(self, parameter_vector: torch.Tensor) -> OrderedDict:
        parameter_mapping = OrderedDict()
        start = 0
        for spec in self.target_parameter_specs:
            numel = prod(spec["shape"])
            end = start + numel
            parameter_mapping[spec["name"]] = parameter_vector[start:end].view(spec["shape"])
            start = end
        return parameter_mapping

    def apply_generated_target_model(
        self,
        example_inputs: torch.Tensor,
        parameter_mapping: OrderedDict,
    ) -> torch.Tensor:
        return self.target_model.forward_with_params(example_inputs, parameter_mapping)

    def build_targets(self, batch: dict) -> torch.Tensor:
        return build_binary_targets(
            support_outputs=batch["support_outputs"],
            query_output=batch["query_output"],
        )

    def build_task_records(
        self,
        batch: dict,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> list[dict]:
        support_inputs = batch["support_inputs"].detach().cpu().long().tolist()
        support_targets = targets[:, : self.support_example_count].detach().cpu().long().tolist()
        support_predictions = (
            predictions[:, : self.support_example_count].detach().cpu().long().tolist()
        )
        query_inputs = batch["query_input"].detach().cpu().long().tolist()
        query_targets = targets[:, self.support_example_count].detach().cpu().long().tolist()
        query_predictions = (
            predictions[:, self.support_example_count].detach().cpu().long().tolist()
        )
        task_categories = list(batch["task_category"])
        task_ids = batch["task_id"].detach().cpu().tolist()

        records = []
        for row in zip(
            support_inputs,
            support_targets,
            support_predictions,
            query_inputs,
            query_targets,
            query_predictions,
            task_categories,
            task_ids,
            strict=True,
        ):
            (
                support_input,
                support_target,
                support_prediction,
                query_input,
                query_target,
                query_prediction,
                task_category,
                task_id,
            ) = row
            support_exact_matches = [
                prediction == target
                for prediction, target in zip(support_prediction, support_target, strict=True)
            ]
            query_exact_match = query_prediction == query_target
            query_accuracy = (
                (torch.tensor(query_prediction) == torch.tensor(query_target))
                .float()
                .mean()
                .item()
            )
            records.append(
                {
                    "support_inputs": support_input,
                    "support_outputs": support_target,
                    "support_predictions": support_prediction,
                    "query_input": query_input,
                    "query_output": query_target,
                    "query_prediction": query_prediction,
                    "task_category": task_category,
                    "task_id": task_id,
                    "support_exact_matches": support_exact_matches,
                    "query_exact_match": query_exact_match,
                    "query_accuracy": query_accuracy,
                }
            )
        return records

    def build_task_attention_record(self, record: dict) -> dict:
        attention_data = self.get_task_encoder_attention_data(
            support_inputs=torch.tensor(record["support_inputs"], device=self.device),
            support_outputs=torch.tensor(record["support_outputs"], device=self.device),
            query_input=torch.tensor(record["query_input"], device=self.device),
        )
        return {
            "task_category": record["task_category"],
            "task_id": record["task_id"],
            "query_exact_match": record.get("query_exact_match"),
            "query_accuracy": record.get("query_accuracy"),
            "token_labels": attention_data["token_labels"][0],
            "token_metadata": attention_data["token_metadata"][0],
            "attentions": [layer_attention[0] for layer_attention in attention_data["attentions"]],
        }

    def build_task_visualization_figure(self, record: dict):
        return render_task_prediction_figure(
            support_inputs=record["support_inputs"],
            support_targets=record["support_outputs"],
            support_predictions=record["support_predictions"],
            query_input=record["query_input"],
            query_target=record["query_output"],
            query_prediction=record["query_prediction"],
            task_category=record["task_category"],
            task_id=record["task_id"],
            query_exact_match=record["query_exact_match"],
        )

    def build_task_visualization_image(self, record: dict, caption_prefix: str) -> wandb.Image:
        caption = (
            f"{caption_prefix} | {record['task_category']}:{record['task_id']} | "
            f"query_exact_match={record['query_exact_match']} | "
            f"query_acc={record['query_accuracy']:.2f}"
        )
        return figure_to_wandb_image(self.build_task_visualization_figure(record), caption=caption)

    def collect_task_records_from_dataloader(
        self,
        dataloader,
        limit: int | None = None,
    ) -> list[dict]:
        self.eval()
        device = next(self.parameters()).device
        records = []

        with torch.no_grad():
            for batch in dataloader:
                tensor_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                _, predictions, targets = self.predict_batch(tensor_batch)
                batch_records = self.build_task_records(batch, predictions.cpu(), targets.cpu())
                records.extend(batch_records)
                if limit is not None and len(records) >= limit:
                    return records[:limit]

        return records

    def select_representative_task_records_from_dataset(
        self,
        dataset,
        split_name: str,
        limit: int,
    ) -> list[int]:
        if limit < 1:
            return []
        seen_categories = set()
        selected_task_ids = []

        for task in dataset.tasks:
            task_category = task["task_category"]
            if task_category in seen_categories:
                continue
            seen_categories.add(task_category)
            selected_task_ids.append(task["task_id"])
            if len(selected_task_ids) >= limit:
                break

        self.selected_representative_task_ids[split_name] = selected_task_ids
        return selected_task_ids

    def collect_task_records_from_dataset_by_task_ids(
        self,
        dataset,
        task_ids: list[int],
    ) -> list[dict]:
        if not task_ids:
            return []

        selected_records = []
        task_id_set = set(task_ids)
        for task in dataset.tasks:
            if task["task_id"] not in task_id_set:
                continue
            batch = {
                "support_inputs": torch.tensor([task["support_inputs"]], device=self.device),
                "support_outputs": torch.tensor([task["support_outputs"]], device=self.device),
                "query_input": torch.tensor([task["query_input"]], device=self.device),
                "query_output": torch.tensor([task["query_output"]], device=self.device),
                "task_category": [task["task_category"]],
                "task_id": torch.tensor([task["task_id"]], device=self.device),
            }
            _, predictions, targets = self.predict_batch(batch)
            selected_records.extend(
                self.build_task_records(batch, predictions.cpu(), targets.cpu())
            )
        selected_records.sort(key=lambda record: task_ids.index(record["task_id"]))
        return selected_records

    def select_hard_task_records(self, dataloader, limit: int | None = None) -> list[dict]:
        hard_records = [
            record
            for record in self.collect_task_records_from_dataloader(dataloader)
            if not record["query_exact_match"]
        ]
        hard_records.sort(key=lambda record: record["query_accuracy"])
        if limit is None:
            return hard_records
        return hard_records[:limit]

    def log_task_gallery(
        self,
        split_name: str,
        records: list[dict],
        output_path: str,
        wandb_logger=None,
        key_prefix: str | None = None,
    ) -> None:
        if not records:
            return

        split_dir = os.path.join(output_path, f"{split_name}_task_examples")
        os.makedirs(split_dir, exist_ok=True)
        wandb_payload = {}
        payload_prefix = key_prefix or split_name

        for index, record in enumerate(records):
            figure = self.build_task_visualization_figure(record)
            filename = (
                f"{payload_prefix}_{index}_{record['task_category']}_{record['task_id']}"
                f"_query{int(record['query_exact_match'])}.png"
            )
            figure.savefig(os.path.join(split_dir, filename), dpi=150, bbox_inches="tight")
            wandb_key = f"{payload_prefix}_{record['task_category']}_{record['task_id']}"
            wandb_payload[wandb_key] = self.build_task_visualization_image(
                record,
                caption_prefix=payload_prefix,
            )

        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "experiment")
            and isinstance(wandb_logger.experiment, wandb.sdk.wandb_run.Run)
        ):
            wandb_logger.experiment.log(wandb_payload)

    def log_task_attention_gallery(
        self,
        split_name: str,
        records: list[dict],
        output_path: str,
        wandb_logger=None,
        key_prefix: str | None = None,
    ) -> None:
        if not records:
            return

        split_dir = os.path.join(output_path, f"{split_name}_task_attention")
        os.makedirs(split_dir, exist_ok=True)
        wandb_payload = {}
        payload_prefix = key_prefix or split_name

        for index, record in enumerate(records):
            attention_record = self.build_task_attention_record(record)
            for layer_index, layer_attention in enumerate(attention_record["attentions"]):
                attention_matrix, head_label = resolve_attention_matrix(
                    layer_attention,
                    reduction=self.attention_head_reduction,
                )
                figure = render_task_attention_figure(
                    attention_matrix=attention_matrix,
                    token_labels=attention_record["token_labels"],
                    token_metadata=attention_record["token_metadata"],
                    task_category=attention_record["task_category"],
                    task_id=attention_record["task_id"],
                    layer_index=layer_index,
                    head_label=head_label,
                    query_exact_match=attention_record.get("query_exact_match"),
                    query_accuracy=attention_record.get("query_accuracy"),
                )
                filename = (
                    f"{payload_prefix}_{index}_{attention_record['task_category']}_"
                    f"{attention_record['task_id']}_layer{layer_index}_{head_label}.png"
                )
                figure.savefig(os.path.join(split_dir, filename), dpi=150, bbox_inches="tight")
                wandb_key = (
                    f"{payload_prefix}_{attention_record['task_category']}_"
                    f"{attention_record['task_id']}_layer{layer_index}"
                )
                caption = (
                    f"{payload_prefix} | {attention_record['task_category']}:"
                    f"{attention_record['task_id']} | layer={layer_index} | head={head_label}"
                )
                wandb_payload[wandb_key] = figure_to_wandb_image(figure, caption=caption)

        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "experiment")
            and isinstance(wandb_logger.experiment, wandb.sdk.wandb_run.Run)
        ):
            wandb_logger.experiment.log(wandb_payload)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        support_inputs = batch["support_inputs"].float()
        support_outputs = batch["support_outputs"].float()
        query_input = batch["query_input"].float()
        task_inputs = torch.cat([support_inputs, query_input.unsqueeze(1)], dim=1)

        parameter_vectors = self.predict_target_parameter_vectors(
            support_inputs=support_inputs,
            support_outputs=support_outputs,
            query_input=query_input,
        )

        if parameter_vectors.shape[0] != task_inputs.shape[0]:
            msg = "Expected exactly one generated parameter vector per task."
            raise RuntimeError(msg)

        task_logits = []
        for task_input, parameter_vector in zip(task_inputs, parameter_vectors, strict=True):
            parameter_mapping = self.parameter_vector_to_mapping(parameter_vector)
            task_logits.append(
                self.apply_generated_target_model(
                    example_inputs=task_input,
                    parameter_mapping=parameter_mapping,
                )
            )

        return torch.stack(task_logits, dim=0), parameter_vectors

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return compute_binary_meta_loss(
            logits,
            targets,
            loss_on_support=self.loss_on_support,
            loss_on_query=self.loss_on_query,
        )

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return decode_binary_logits(logits)

    def compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return compute_binary_meta_metrics(logits, targets)

    def common_step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits, parameter_vectors = self(batch)
        targets = self.build_targets(batch)
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)
        batch_size = targets.shape[0]
        log_on_step = prefix == "train"

        self.log(
            f"{prefix}_loss",
            loss,
            prog_bar=True,
            batch_size=batch_size,
            on_step=log_on_step,
            on_epoch=True,
        )
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{prefix}_{metric_name}",
                metric_value,
                prog_bar=metric_name == "query_exact_match_accuracy",
                batch_size=batch_size,
                on_step=log_on_step,
                on_epoch=True,
            )
        self.log(
            f"{prefix}_generated_parameter_count",
            torch.tensor(float(parameter_vectors.shape[1]), device=self.device),
            prog_bar=False,
            batch_size=batch_size,
            on_step=log_on_step,
            on_epoch=True,
        )
        self.log(
            f"{prefix}_generated_parameter_l2",
            parameter_vectors.norm(dim=1).mean(),
            prog_bar=False,
            batch_size=batch_size,
            on_step=log_on_step,
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, "test")

    def predict_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _ = self(batch)
        targets = self.build_targets(batch)
        return logits, self.decode_logits(logits), targets.long()

    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        return optimizer_cls(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        filtered_state_dict = dict(state_dict)
        filtered_state_dict.setdefault(
            "target_model.position_ramp",
            self.target_model.position_ramp.detach().clone(),
        )
        current_state_dict = self.state_dict()
        mismatched_keys = [
            key
            for key, value in list(filtered_state_dict.items())
            if key in current_state_dict and current_state_dict[key].shape != value.shape
        ]
        for key in mismatched_keys:
            filtered_state_dict.pop(key)
        if mismatched_keys:
            print(
                "Ignoring incompatible checkpoint keys for BinaryHyperRNNMetaModelLightning: "
                + ", ".join(sorted(mismatched_keys))
            )
            strict = False
        return super().load_state_dict(filtered_state_dict, strict=strict)

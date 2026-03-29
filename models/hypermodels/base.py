"""Shared task-conditioned hypernetwork components."""

import os
from collections import OrderedDict

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - fallback for older torch versions
    from torch.nn.utils.stateless import functional_call

from models.target_models.transformer import Block, LayerNorm, TransformerConfig
from visualisation import figure_to_wandb_image, render_task_prediction_figure


class TaskTransformerEncoder(nn.Module):
    """Small transformer encoder reused for task conditioning."""

    def __init__(
        self,
        block_size: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        config = TransformerConfig(
            block_size=block_size,
            input_feature_dim=hidden_dim,
            logit_dim=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_embd=hidden_dim,
            dropout=dropout,
            bias=bias,
            causal=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(num_layers)])
        self.ln_f = LayerNorm(hidden_dim, bias=bias)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.ln_f(hidden_states)


class BaseHyperMetaModelLightning(pl.LightningModule):
    """Shared Lightning module for task-conditioned hypernetwork experiments."""

    supports_hard_val_examples = False
    supports_task_visualization = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_classes: int,
        task_encoder_hidden_dim: int,
        task_encoder_num_heads: int,
        task_encoder_num_layers: int,
        target_model_template: nn.Module,
        learning_rate: float = 0.001,
        optimizer: str = "Adam",
        weight_decay: float = 0.01,
        loss_on_support: bool = True,
        loss_on_query: bool = True,
        task_encoder_dropout: float = 0.0,
        task_encoder_bias: bool = True,
        log_task_examples: bool = True,
        log_task_examples_every_n_epochs: int = 1,
        num_hard_task_examples: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs", "target_model_template"])

        if input_dim != output_dim:
            msg = "The hypernetwork path expects input_dim and output_dim to match."
            raise ValueError(msg)
        if num_classes < 2:
            msg = "num_classes must be at least 2."
            raise ValueError(msg)
        if not loss_on_support and not loss_on_query:
            msg = "At least one of loss_on_support or loss_on_query must be enabled."
            raise ValueError(msg)

        self.sequence_length = input_dim
        self.num_classes = num_classes
        self.loss_on_support = loss_on_support
        self.loss_on_query = loss_on_query
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.log_task_examples = log_task_examples
        self.log_task_examples_every_n_epochs = log_task_examples_every_n_epochs
        self.num_periodic_train_task_examples = kwargs.get("num_periodic_train_task_examples", 1)
        self.num_periodic_val_task_examples = kwargs.get("num_periodic_val_task_examples", 1)
        self.num_final_hard_val_task_examples = kwargs.get("num_final_hard_val_task_examples", 3)
        self.num_hard_task_examples = num_hard_task_examples
        self.selected_representative_task_ids: dict[str, list[int]] = {"train": [], "val": []}

        self.num_segments = 7
        self.num_examples = 4
        self.support_example_count = 3
        self.segment_sequence_length = self.num_segments * self.sequence_length

        self.value_embedding = nn.Embedding(num_classes, task_encoder_hidden_dim)
        self.position_embedding = nn.Embedding(self.sequence_length, task_encoder_hidden_dim)
        self.example_embedding = nn.Embedding(self.num_examples, task_encoder_hidden_dim)
        self.role_embedding = nn.Embedding(2, task_encoder_hidden_dim)
        self.task_encoder = TaskTransformerEncoder(
            block_size=self.segment_sequence_length,
            hidden_dim=task_encoder_hidden_dim,
            num_heads=task_encoder_num_heads,
            num_layers=task_encoder_num_layers,
            dropout=task_encoder_dropout,
            bias=task_encoder_bias,
        )

        self.target_model_template = target_model_template
        self.target_parameter_specs = self.build_target_parameter_specs()
        self.target_parameter_count = sum(spec["numel"] for spec in self.target_parameter_specs)
        self.hyper_head = nn.Sequential(
            nn.Linear(task_encoder_hidden_dim, task_encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(task_encoder_hidden_dim, self.target_parameter_count),
        )

    def build_target_parameter_specs(self) -> list[dict]:
        specs = []
        for name, parameter in self.target_model_template.named_parameters():
            specs.append(
                {
                    "name": name,
                    "shape": tuple(parameter.shape),
                    "numel": parameter.numel(),
                }
            )
        return specs

    def serialise_task_segments(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        example_ids = support_inputs.new_tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.long)
        role_ids = support_inputs.new_tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
        return segment_values, example_ids, role_ids

    def encode_task(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> torch.Tensor:
        segment_values, example_ids, role_ids = self.serialise_task_segments(
            support_inputs, support_outputs, query_input
        )
        batch_size = segment_values.shape[0]
        flat_values = segment_values.reshape(batch_size, self.segment_sequence_length)

        value_embeddings = self.value_embedding(flat_values.long())
        position_ids = torch.arange(self.sequence_length, device=flat_values.device).repeat(
            self.num_segments
        )
        position_embeddings = self.position_embedding(position_ids).unsqueeze(0)

        flat_example_ids = example_ids.repeat_interleave(self.sequence_length)
        example_embeddings = self.example_embedding(flat_example_ids).unsqueeze(0)

        flat_role_ids = role_ids.repeat_interleave(self.sequence_length)
        role_embeddings = self.role_embedding(flat_role_ids).unsqueeze(0)

        encoded_tokens = self.task_encoder(
            value_embeddings + position_embeddings + example_embeddings + role_embeddings
        )
        return encoded_tokens.mean(dim=1)

    def predict_target_parameter_vectors(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> torch.Tensor:
        task_representations = self.encode_task(support_inputs, support_outputs, query_input)
        return self.hyper_head(task_representations)

    def parameter_vector_to_mapping(self, parameter_vector: torch.Tensor) -> OrderedDict:
        parameter_mapping = OrderedDict()
        start = 0
        for spec in self.target_parameter_specs:
            end = start + spec["numel"]
            parameter_mapping[spec["name"]] = parameter_vector[start:end].view(spec["shape"])
            start = end
        return parameter_mapping

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        support_inputs = batch["support_inputs"].long()
        support_outputs = batch["support_outputs"].long()
        query_input = batch["query_input"].long()

        parameter_vectors = self.predict_target_parameter_vectors(
            support_inputs=support_inputs,
            support_outputs=support_outputs,
            query_input=query_input,
        )

        task_inputs = torch.cat([support_inputs, query_input.unsqueeze(1)], dim=1)
        task_logits = []
        for task_input, parameter_vector in zip(task_inputs, parameter_vectors, strict=True):
            parameter_mapping = self.parameter_vector_to_mapping(parameter_vector)
            example_logits = []
            for example_input in task_input:
                example_logits.append(
                    functional_call(
                        self.target_model_template,
                        parameter_mapping,
                        (example_input.unsqueeze(0),),
                    ).squeeze(0)
                )
            task_logits.append(torch.stack(example_logits, dim=0))

        return torch.stack(task_logits, dim=0), parameter_vectors

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_position_loss = F.cross_entropy(
            logits.reshape(-1, self.num_classes),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape[0], targets.shape[1], targets.shape[2])
        per_example_loss = per_position_loss.mean(dim=2)

        selected_losses = []
        if self.loss_on_support:
            selected_losses.append(per_example_loss[:, : self.support_example_count])
        if self.loss_on_query:
            selected_losses.append(per_example_loss[:, self.support_example_count :])
        return torch.cat(selected_losses, dim=1).mean()

    def build_targets(self, batch: dict) -> torch.Tensor:
        return torch.cat(
            [batch["support_outputs"].long(), batch["query_output"].long().unsqueeze(1)],
            dim=1,
        )

    def predict_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _ = self(batch)
        targets = self.build_targets(batch)
        predictions = logits.argmax(dim=-1)
        return logits, predictions, targets

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

    def select_representative_task_records(self, dataloader) -> list[dict]:
        seen_categories = set()
        selected_records = []
        for record in self.collect_task_records_from_dataloader(dataloader):
            if record["task_category"] in seen_categories:
                continue
            seen_categories.add(record["task_category"])
            selected_records.append(record)
        return selected_records

    def select_representative_task_records_from_dataset(
        self,
        dataset,
        split_name: str,
        limit: int,
    ) -> list[dict]:
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
                record, caption_prefix=payload_prefix
            )

        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "experiment")
            and isinstance(wandb_logger.experiment, wandb.sdk.wandb_run.Run)
        ):
            wandb_logger.experiment.log(wandb_payload)

    def compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predictions = logits.argmax(dim=-1)
        exact_matches = (predictions == targets).all(dim=2)

        support_predictions = predictions[:, : self.support_example_count]
        support_targets = targets[:, : self.support_example_count]
        query_predictions = predictions[:, self.support_example_count]
        query_targets = targets[:, self.support_example_count]

        return {
            "support_accuracy": (support_predictions == support_targets).float().mean(),
            "query_accuracy": (query_predictions == query_targets).float().mean(),
            "support_exact_match_accuracy": exact_matches[:, : self.support_example_count]
            .float()
            .mean(),
            "query_exact_match_accuracy": exact_matches[:, self.support_example_count]
            .float()
            .mean(),
            "all_examples_exact_match_accuracy": exact_matches.all(dim=1).float().mean(),
        }

    def common_step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits, parameter_vectors = self(batch)
        targets = self.build_targets(batch)
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)
        batch_size = targets.shape[0]

        self.log(f"{prefix}_loss", loss, prog_bar=(prefix != "test"), batch_size=batch_size)
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{prefix}_{metric_name}",
                metric_value,
                prog_bar=metric_name == "query_exact_match_accuracy",
                batch_size=batch_size,
            )
        self.log(
            f"{prefix}_generated_parameter_l2",
            parameter_vectors.norm(dim=1).mean(),
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, "test")

    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        return optimizer_cls(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


__all__ = ["BaseHyperMetaModelLightning", "TaskTransformerEncoder"]

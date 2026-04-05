"""Lightning training wrapper for the simplified binary HyperModel path."""

import os
from collections.abc import Mapping

import lightning as pl
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import wandb
from metrics import accuracy, exact_match_accuracy
from models.cnn import CNN
from models.hypermodel import HyperModel
from models.rnn import RNN
from models.task_token_embedder import TaskTokenEmbedder
from models.transformer import Transformer
from visualisation import figure_to_wandb_image, render_task_prediction_figure

NUM_SUPPORT_EXAMPLES = 3
NUM_TASK_EXAMPLES = 4
NUM_TASK_SEGMENTS = 7
TASK_SCALAR_FEATURE_DIM = 5
TASK_ENCODING_SCALAR = "scalar"
TASK_ENCODING_SHARED_EMBEDDINGS = "shared_embeddings"


HYPERNETWORK_REGISTRY = {
    "cnn": {
        "class": CNN,
        "output_dim_key": "output_dim",
    },
    "transformer": {
        "class": Transformer,
        "output_dim_key": "output_dim",
    },
    "rnn": {
        "class": RNN,
        "output_dim_key": "output_dim",
    },
}

TARGET_MODEL_REGISTRY = {
    "cnn": {
        "class": CNN,
        "owned_params": {"output_dim": 1},
    },
    "rnn": {
        "class": RNN,
        "owned_params": {"output_dim": 1},
    },
    "transformer": {
        "class": Transformer,
        "owned_params": {"output_dim": 1},
    },
}


def _ensure_typed_model_config(config: Mapping, config_name: str) -> tuple[str, dict]:
    """Validate and normalize a typed model config section."""
    name = config.get("name")
    if not isinstance(name, str) or not name:
        msg = f"{config_name}.name must be a non-empty string."
        raise ValueError(msg)

    params = config.get("params", {})
    if not isinstance(params, Mapping):
        msg = f"{config_name}.params must be a mapping."
        raise ValueError(msg)
    return name, dict(params)


def _apply_owned_params(
    params: dict,
    owned_params: Mapping[str, int],
    config_name: str,
) -> dict:
    """Inject wrapper-owned parameters and reject conflicting user values."""
    merged_params = dict(params)
    for key, value in owned_params.items():
        if key in merged_params and merged_params[key] != value:
            msg = (
                f"{config_name}.params.{key} is owned by HyperModelLightning and must be "
                f"{value}, got {merged_params[key]!r}."
            )
            raise ValueError(msg)
        merged_params[key] = value
    return merged_params


class HyperModelLightning(pl.LightningModule):
    """Binary ARC1D training wrapper around a generic HyperModel."""

    supports_hard_val_examples = False
    supports_task_visualization = True

    def __init__(
        self,
        hyper_model: dict,
        target_model: dict,
        hyper_head: dict | None = None,
        task_encoding: dict | None = None,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_name: str = "Adam",
        weight_decay: float = 0.01,
        lr_scheduler: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        (
            self.task_encoding_name,
            self.task_encoding_params,
            self.hyper_input_dim,
            self.target_input_dim,
        ) = self.resolve_task_encoding(task_encoding, kwargs)
        self.shared_task_token_embedder = self.build_task_token_embedder()
        hypernetwork, hyper_output_dim = self.build_hypernetwork(hyper_model)
        target = self.build_target_model(target_model)
        hyper_head_cfg = hyper_head or {}
        bottleneck_dim = hyper_head_cfg.get("bottleneck_dim")
        use_vae = bool(hyper_head_cfg.get("use_vae", False))
        kl_weight = float(hyper_head_cfg.get("kl_weight", 0.0))
        if bottleneck_dim is not None and (not isinstance(bottleneck_dim, int) or bottleneck_dim < 1):
            msg = "hyper_head.bottleneck_dim must be a positive integer."
            raise ValueError(msg)
        self.kl_weight = kl_weight
        self.hypermodel = HyperModel(
            hypernetwork=hypernetwork,
            target_model=target,
            hyper_output_dim=hyper_output_dim,
            bottleneck_dim=bottleneck_dim,
            use_vae=use_vae,
        )
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name or optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler
        self.log_task_examples = kwargs.get("log_task_examples", False)
        self.log_task_examples_every_n_epochs = kwargs.get("log_task_examples_every_n_epochs", 25)
        self.num_periodic_train_task_examples = kwargs.get("num_periodic_train_task_examples", 1)
        self.num_periodic_val_task_examples = kwargs.get("num_periodic_val_task_examples", 1)
        self.selected_representative_task_ids: dict[str, list[int]] = {"train": [], "val": []}

    def resolve_task_encoding(
        self,
        task_encoding: Mapping | None,
        runtime_kwargs: Mapping,
    ) -> tuple[str, dict, int, int]:
        """Resolve the wrapper-owned task encoding contract."""
        if task_encoding is None:
            task_encoding = {}
        if not isinstance(task_encoding, Mapping):
            msg = "task_encoding must be a mapping with name and params fields."
            raise ValueError(msg)

        task_encoding_name = task_encoding.get("name", TASK_ENCODING_SCALAR)
        if task_encoding_name not in {TASK_ENCODING_SCALAR, TASK_ENCODING_SHARED_EMBEDDINGS}:
            msg = (
                f"Unknown task_encoding.name {task_encoding_name!r}. Expected one of "
                f"{[TASK_ENCODING_SCALAR, TASK_ENCODING_SHARED_EMBEDDINGS]}."
            )
            raise ValueError(msg)

        params = task_encoding.get("params", {})
        if not isinstance(params, Mapping):
            msg = "task_encoding.params must be a mapping."
            raise ValueError(msg)
        resolved_params = dict(params)

        if task_encoding_name == TASK_ENCODING_SCALAR:
            return task_encoding_name, resolved_params, TASK_SCALAR_FEATURE_DIM, 2

        share_input_embeddings = resolved_params.get("share_input_embeddings", True)
        if share_input_embeddings is not True:
            msg = "task_encoding.params.share_input_embeddings must be true when enabled."
            raise ValueError(msg)

        embedding_dim = resolved_params.get("embedding_dim")
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            msg = "task_encoding.params.embedding_dim must be a positive integer."
            raise ValueError(msg)

        position_vocab_size = runtime_kwargs.get("input_dim")
        if not isinstance(position_vocab_size, int) or position_vocab_size < 1:
            msg = (
                "shared_embeddings requires a positive integer top-level input_dim so "
                "position embeddings know their vocabulary size."
            )
            raise ValueError(msg)

        resolved_params["embedding_dim"] = embedding_dim
        resolved_params["position_vocab_size"] = position_vocab_size
        resolved_params["share_input_embeddings"] = True
        return task_encoding_name, resolved_params, embedding_dim, embedding_dim

    def build_task_token_embedder(self) -> TaskTokenEmbedder | None:
        """Instantiate the shared task-token embedder when requested."""
        if self.task_encoding_name != TASK_ENCODING_SHARED_EMBEDDINGS:
            return None
        return TaskTokenEmbedder(
            embedding_dim=self.task_encoding_params["embedding_dim"],
            position_vocab_size=self.task_encoding_params["position_vocab_size"],
        )

    def build_hypernetwork(self, hyper_model: Mapping) -> tuple[torch.nn.Module, int]:
        """Instantiate the configured hypernetwork and return its output width."""
        if not isinstance(hyper_model, Mapping):
            msg = "hyper_model must be a mapping with name and params fields."
            raise ValueError(msg)

        hypernetwork_name, hypernetwork_params = _ensure_typed_model_config(
            hyper_model,
            "hyper_model",
        )
        hypernetwork_spec = HYPERNETWORK_REGISTRY.get(hypernetwork_name)
        if hypernetwork_spec is None:
            msg = (
                f"Unknown hyper_model.name {hypernetwork_name!r}. "
                f"Expected one of {sorted(HYPERNETWORK_REGISTRY)}."
            )
            raise ValueError(msg)

        resolved_params = _apply_owned_params(
            hypernetwork_params,
            {"input_dim": self.hyper_input_dim},
            "hyper_model",
        )
        output_dim_key = hypernetwork_spec["output_dim_key"]
        hyper_output_dim = resolved_params.get(output_dim_key)
        if not isinstance(hyper_output_dim, int) or hyper_output_dim < 1:
            msg = (
                "hyper_model.params must define a positive integer "
                f"{output_dim_key!r} so HyperModel can size the hyper head."
            )
            raise ValueError(msg)

        hypernetwork_cls = hypernetwork_spec["class"]
        return hypernetwork_cls(**resolved_params), hyper_output_dim

    def build_target_model(self, target_model: Mapping) -> torch.nn.Module:
        """Instantiate the configured stateless target-model template."""
        if not isinstance(target_model, Mapping):
            msg = "target_model must be a mapping with name and params fields."
            raise ValueError(msg)

        target_model_name, target_model_params = _ensure_typed_model_config(
            target_model,
            "target_model",
        )
        target_model_spec = TARGET_MODEL_REGISTRY.get(target_model_name)
        if target_model_spec is None:
            msg = (
                f"Unknown target_model.name {target_model_name!r}. "
                f"Expected one of {sorted(TARGET_MODEL_REGISTRY)}."
            )
            raise ValueError(msg)

        resolved_params = _apply_owned_params(
            target_model_params,
            {"input_dim": self.target_input_dim, **target_model_spec["owned_params"]},
            "target_model",
        )
        target_model_cls = target_model_spec["class"]
        return target_model_cls(**resolved_params)

    def build_task_context_token_ids(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build token ids for the serialized full-task hypernetwork input."""
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
        batch_size, _, sequence_length = segment_values.shape
        flat_values = segment_values.reshape(batch_size, -1).long()

        position_ids = torch.arange(sequence_length, device=segment_values.device).repeat(
            NUM_TASK_SEGMENTS
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        example_ids = support_inputs.new_tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.long)
        example_ids = (
            example_ids.repeat_interleave(sequence_length)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        role_ids = support_inputs.new_tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
        role_ids = role_ids.repeat_interleave(sequence_length).unsqueeze(0).expand(batch_size, -1)

        is_query_ids = support_inputs.new_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.long)
        is_query_ids = (
            is_query_ids.repeat_interleave(sequence_length).unsqueeze(0).expand(batch_size, -1)
        )
        return flat_values, position_ids, example_ids, role_ids, is_query_ids

    def build_target_token_ids(
        self,
        support_inputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build token ids for input-side target-model sequences."""
        all_inputs = torch.cat([support_inputs, query_input.unsqueeze(1)], dim=1)
        batch_size, num_examples, sequence_length = all_inputs.shape

        value_ids = all_inputs.long()
        position_ids = torch.arange(sequence_length, device=all_inputs.device)
        position_ids = position_ids.view(1, 1, -1).expand(batch_size, num_examples, -1)

        example_ids = torch.arange(num_examples, device=all_inputs.device)
        example_ids = example_ids.view(1, num_examples, 1).expand(batch_size, -1, sequence_length)

        role_ids = torch.zeros_like(value_ids)
        is_query_ids = torch.zeros_like(value_ids)
        is_query_ids[:, -1, :] = 1
        return value_ids, position_ids, example_ids, role_ids, is_query_ids

    def prepare_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a task batch into hypernetwork features, target inputs, and targets."""
        support_inputs = batch["support_inputs"].float()
        support_outputs = batch["support_outputs"].float()
        query_input = batch["query_input"].float()
        query_output = batch["query_output"].float()
        batch_size, _, sequence_length = support_inputs.shape

        if self.task_encoding_name == TASK_ENCODING_SHARED_EMBEDDINGS:
            task_features = self.shared_task_token_embedder(
                *self.build_task_context_token_ids(support_inputs, support_outputs, query_input)
            )
            example_inputs = self.shared_task_token_embedder(
                *self.build_target_token_ids(support_inputs, query_input)
            )
        else:
            # The hypernetwork sees the full task in the legacy ARC order:
            # support 1 input/output, support 2 input/output, support 3 input/output, query input.
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

            # Flatten the 7 task segments into one token sequence for the task encoder.
            flat_values = segment_values.reshape(batch_size, -1, 1)

            # This is just normalized positional information for each 1D location.
            normalized_positions = torch.linspace(
                0.0,
                1.0,
                steps=sequence_length,
                device=segment_values.device,
            )
            position_feature = normalized_positions.repeat(NUM_TASK_SEGMENTS).view(1, -1, 1)

            # These metadata features tell the hypernetwork which example each token belongs to,
            # whether it is an input or output token, and whether it came from the query example.
            example_ids = support_inputs.new_tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.float32)
            example_feature = (
                (example_ids / (NUM_TASK_EXAMPLES - 1))
                .repeat_interleave(sequence_length)
                .view(1, -1, 1)
            )
            role_ids = support_inputs.new_tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
            role_feature = role_ids.repeat_interleave(sequence_length).view(1, -1, 1)
            is_query_ids = support_inputs.new_tensor(
                [0, 0, 0, 0, 0, 0, 1], dtype=torch.float32
            )
            is_query_feature = is_query_ids.repeat_interleave(sequence_length).view(1, -1, 1)

            task_features = torch.cat(
                [
                    flat_values,
                    position_feature.expand(batch_size, -1, -1),
                    example_feature.expand(batch_size, -1, -1),
                    role_feature.expand(batch_size, -1, -1),
                    is_query_feature.expand(batch_size, -1, -1),
                ],
                dim=-1,
            )

            # The target model stays simple: each example is represented only
            # by value and position.
            all_inputs = torch.cat([support_inputs, query_input.unsqueeze(1)], dim=1)
            position_values = normalized_positions.view(1, 1, sequence_length, 1).expand(
                batch_size,
                NUM_TASK_EXAMPLES,
                -1,
                -1,
            )
            example_inputs = torch.cat([all_inputs.unsqueeze(-1), position_values], dim=-1)

        # Targets follow the same 4-example layout as the logits.
        example_targets = torch.cat([support_outputs, query_output.unsqueeze(1)], dim=1)

        return task_features, example_inputs, example_targets

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """batch -> (logits, targets) with shape (batch, 4, seq_len)."""
        task_features, example_inputs, example_targets = self.prepare_inputs(batch)
        logits = self.hypermodel(task_features, example_inputs)
        return logits, example_targets

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total_loss, task_loss, kl_loss) for logging and backprop."""
        task_loss = F.binary_cross_entropy_with_logits(logits, targets)
        kl_loss = self.hypermodel._last_kl
        total_loss = task_loss + self.kl_weight * kl_loss
        return total_loss, task_loss, kl_loss

    def compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        predictions = (torch.sigmoid(logits) >= 0.5).long()
        targets_long = targets.long()
        exact_matches = (predictions == targets_long).all(dim=2).float()

        support_predictions = predictions[:, :NUM_SUPPORT_EXAMPLES].reshape(
            -1, predictions.shape[-1]
        )
        support_targets = targets_long[:, :NUM_SUPPORT_EXAMPLES].reshape(
            -1, targets_long.shape[-1]
        )
        query_predictions = predictions[:, NUM_SUPPORT_EXAMPLES:].reshape(
            -1, predictions.shape[-1]
        )
        query_targets = targets_long[:, NUM_SUPPORT_EXAMPLES:].reshape(-1, targets_long.shape[-1])

        key = f"{prefix}_" if prefix else ""
        return {
            f"{key}support_accuracy": accuracy(support_targets, support_predictions),
            f"{key}support_exact_match": exact_match_accuracy(
                support_targets,
                support_predictions,
            ),
            f"{key}query_accuracy": accuracy(query_targets, query_predictions),
            f"{key}query_exact_match": exact_match_accuracy(
                query_targets,
                query_predictions,
            ),
            f"{key}all_examples_exact_match": exact_matches.all(dim=1).float().mean(),
        }

    def common_step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits, targets = self(batch)
        loss, task_loss, kl_loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets, prefix=prefix)
        batch_size = batch["support_inputs"].shape[0]
        log_on_step = prefix == "train"
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        log_kwargs = {
            "on_step": log_on_step,
            "on_epoch": True,
            "batch_size": batch_size,
            "sync_dist": sync_dist,
        }
        self.log(f"{prefix}_loss", loss, prog_bar=True, **log_kwargs)
        self.log(f"{prefix}_task_loss", task_loss, prog_bar=False, **log_kwargs)
        self.log(f"{prefix}_kl_loss", kl_loss, prog_bar=False, **log_kwargs)
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=log_on_step,
                on_epoch=True,
                prog_bar=name.endswith("query_exact_match"),
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        self.common_step(batch, prefix="test")

    def predict_batch(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, binary predictions, and targets for a task batch."""
        logits, targets = self(batch)
        predictions = (torch.sigmoid(logits) >= 0.5).long()
        return logits, predictions, targets.long()

    def build_task_records(
        self,
        batch: dict,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> list[dict]:
        """Convert a task batch into visualisation-friendly prediction records."""
        support_inputs = batch["support_inputs"].detach().cpu().long().tolist()
        support_targets = targets[:, :NUM_SUPPORT_EXAMPLES].detach().cpu().long().tolist()
        support_predictions = (
            predictions[:, :NUM_SUPPORT_EXAMPLES].detach().cpu().long().tolist()
        )
        query_inputs = batch["query_input"].detach().cpu().long().tolist()
        query_targets = targets[:, NUM_SUPPORT_EXAMPLES].detach().cpu().long().tolist()
        query_predictions = (
            predictions[:, NUM_SUPPORT_EXAMPLES].detach().cpu().long().tolist()
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
                    "query_exact_match": query_exact_match,
                    "query_accuracy": query_accuracy,
                }
            )
        return records

    def build_task_visualization_figure(self, record: dict):
        """Render support/query predictions for one task."""
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
        """Convert a rendered task figure to a W&B image payload."""
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
        """Run inference over a dataloader and collect task prediction records."""
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
        """Select at most one task id per category for recurring visual logging."""
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
        """Collect prediction records for a fixed set of task ids."""
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

    def log_task_gallery(
        self,
        split_name: str,
        records: list[dict],
        output_path: str,
        wandb_logger=None,
        key_prefix: str | None = None,
    ) -> None:
        """Save per-task prediction figures and optionally push them to W&B."""
        if not records:
            return
        if self.trainer is not None and not self.trainer.is_global_zero:
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
            plt.close(figure)
            wandb_key = f"{payload_prefix}_{record['task_category']}_{record['task_id']}"
            wandb_payload[wandb_key] = self.build_task_visualization_image(
                record,
                caption_prefix=payload_prefix,
            )

        experiment = getattr(wandb_logger, "experiment", None)
        if wandb_payload and experiment is not None and hasattr(experiment, "log"):
            experiment.log(wandb_payload)

    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_cls(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.lr_scheduler_cfg:
            sched_cls = getattr(torch.optim.lr_scheduler, self.lr_scheduler_cfg["name"])
            scheduler = sched_cls(optimizer, **self.lr_scheduler_cfg.get("params", {}))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return optimizer

    def on_fit_start(self) -> None:
        print(repr(self.hypermodel))
        n = sum(p.numel() for p in self.hypermodel.parameters() if p.requires_grad)
        print(f"Trainable params : {n:,}")

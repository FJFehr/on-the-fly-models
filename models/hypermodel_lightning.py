"""Lightning training wrapper for the simplified HyperModel path."""

import os
import re
from collections.abc import Mapping

import lightning as pl
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import wandb
from models.cnn import CNN
from models.hypermodel import AttentionPooler, HierarchicalPooler, HyperModel
from models.rnn import RNN
from models.task_token_embedder import TaskTokenEmbedder
from models.transformer import Transformer
from visualisation import figure_to_wandb_image, render_task_prediction_figure

NUM_SUPPORT_EXAMPLES = 3
NUM_TASK_EXAMPLES = 4
PREDICTION_TASK_BINARY = "binary"
PREDICTION_TASK_MULTICLASS = "multiclass"

TASK_CATEGORY_INDEX: dict[str, int] = {
    "1d_move_1p": 0,
    "1d_move_2p": 1,
    "1d_move_3p": 2,
    "1d_move_dp": 3,
    "1d_move_2p_dp": 4,
    "1d_fill": 5,
    "1d_hollow": 6,
    "1d_flip": 7,
    "1d_mirror": 8,
    "1d_denoising_1c": 9,
    "1d_denoising_mc": 10,
    "1d_pcopy_1c": 11,
    "1d_pcopy_mc": 12,
    "1d_recolor_oe": 13,
    "1d_recolor_cnt": 14,
    "1d_recolor_cmp": 15,
    "1d_scale_dp": 16,
    "1d_padded_fill": 17,
}


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
        "owned_params": {},
    },
    "rnn": {
        "class": RNN,
        "owned_params": {},
    },
    "transformer": {
        "class": Transformer,
        "owned_params": {},
    },
}


def _task_category_metric_suffix(task_category: str) -> str:
    """Convert a task category string into a safe metric name suffix."""
    return re.sub(r"[^a-zA-Z0-9]", "_", task_category)


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


def _resolve_positive_int(value: object, config_key: str) -> int:
    """Validate a positive integer config value."""
    if not isinstance(value, int) or value < 1:
        msg = f"{config_key} must be a positive integer."
        raise ValueError(msg)
    return value


def _resolve_dropout(value: object, config_key: str) -> float:
    """Validate a dropout config value."""
    if not isinstance(value, int | float) or not 0.0 <= float(value) < 1.0:
        msg = f"{config_key} must be a float in [0.0, 1.0)."
        raise ValueError(msg)
    return float(value)


class HyperModelLightning(pl.LightningModule):
    """ARC1D training wrapper around a generic HyperModel."""

    supports_hard_val_examples = False
    supports_task_visualization = True

    def __init__(
        self,
        hyper_model: dict,
        target_model: dict,
        hyper_head: dict | None = None,
        task_encoding: dict | None = None,
        prediction_task: str = PREDICTION_TASK_BINARY,
        num_classes: int | None = None,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_name: str = "Adam",
        weight_decay: float = 0.01,
        lr_scheduler: dict | None = None,
        warmup_steps: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.prediction_task, self.num_classes = self.resolve_prediction_task(
            prediction_task, num_classes
        )
        self.target_output_dim = 1 if self.is_binary_task else self.num_classes
        self.padding_idx: int | None = kwargs.get("padding_idx")
        embedding_dim, value_vocab_size = self._resolve_embedding_params(task_encoding, kwargs)
        self.embedding_dim = embedding_dim
        self.shared_task_token_embedder = TaskTokenEmbedder(
            embedding_dim=embedding_dim,
            value_vocab_size=value_vocab_size,
        )
        hypernetwork, hyper_output_dim = self.build_hypernetwork(hyper_model, embedding_dim)
        target = self.build_target_model(target_model, embedding_dim)
        hyper_head_cfg = hyper_head or {}
        if not isinstance(hyper_head_cfg, Mapping):
            msg = "hyper_head must be a mapping."
            raise ValueError(msg)
        bottleneck_dim = hyper_head_cfg.get("bottleneck_dim")
        if bottleneck_dim is not None and (
            not isinstance(bottleneck_dim, int) or bottleneck_dim < 1
        ):
            msg = "hyper_head.bottleneck_dim must be a positive integer."
            raise ValueError(msg)
        projection_dims = hyper_head_cfg.get("projection_dims")
        num_tasks = hyper_head_cfg.get("num_tasks")
        low_rank_output = bool(hyper_head_cfg.get("low_rank_output", False))
        hyper_pooling = self.build_hyper_pooling(hyper_head_cfg, hyper_output_dim)
        self.hypermodel = HyperModel(
            hypernetwork=hypernetwork,
            target_model=target,
            hyper_output_dim=hyper_output_dim,
            bottleneck_dim=bottleneck_dim,
            projection_dims=projection_dims,
            hyper_pooling=hyper_pooling,
            num_tasks=num_tasks,
            low_rank_output=low_rank_output,
        )
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name or optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler
        self.warmup_steps = warmup_steps
        self.log_task_examples = kwargs.get("log_task_examples", False)
        self.log_task_examples_every_n_epochs = kwargs.get("log_task_examples_every_n_epochs", 25)
        self.num_periodic_train_task_examples = kwargs.get("num_periodic_train_task_examples", 1)
        self.num_periodic_val_task_examples = kwargs.get("num_periodic_val_task_examples", 1)
        self.selected_representative_task_ids: dict[str, list[int]] = {"train": [], "val": []}
        self._val_metric_totals: dict[str, float] = {}
        self._test_metric_totals: dict[str, float] = {}
        self._val_query_exact_match_totals_by_task: dict[str, float] = {}
        self._val_query_exact_match_counts_by_task: dict[str, int] = {}

    @property
    def is_binary_task(self) -> bool:
        return self.prediction_task == PREDICTION_TASK_BINARY

    def resolve_prediction_task(
        self,
        prediction_task: str,
        num_classes: int | None,
    ) -> tuple[str, int]:
        """Validate the prediction mode and return its owned output width."""
        if prediction_task == PREDICTION_TASK_BINARY:
            return prediction_task, 2
        if prediction_task == PREDICTION_TASK_MULTICLASS:
            if not isinstance(num_classes, int) or num_classes < 2:
                msg = "num_classes must be an integer >= 2 for multiclass prediction."
                raise ValueError(msg)
            return prediction_task, num_classes
        msg = (
            f"Unknown prediction_task {prediction_task!r}. Expected one of "
            f"{[PREDICTION_TASK_BINARY, PREDICTION_TASK_MULTICLASS]}."
        )
        raise ValueError(msg)

    def _resolve_embedding_params(
        self,
        task_encoding: Mapping | None,
        runtime_kwargs: Mapping,
    ) -> tuple[int, int]:
        """Return (embedding_dim, value_vocab_size) from config."""
        if task_encoding is None:
            task_encoding = {}
        if not isinstance(task_encoding, Mapping):
            msg = "task_encoding must be a mapping."
            raise ValueError(msg)

        embedding_dim = task_encoding.get("embedding_dim")
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            msg = "task_encoding.embedding_dim must be a positive integer."
            raise ValueError(msg)

        value_vocab_size = task_encoding.get("value_vocab_size", 2)
        if not isinstance(value_vocab_size, int) or value_vocab_size < 2:
            msg = "task_encoding.value_vocab_size must be an integer >= 2."
            raise ValueError(msg)

        return embedding_dim, value_vocab_size

    def build_hyper_pooling(
        self,
        hyper_head: Mapping,
        hyper_output_dim: int,
    ) -> torch.nn.Module:
        """Instantiate the configured pooling path for the hyper head."""
        pooling_name = hyper_head.get("pooling", "attention")
        if pooling_name == "attention":
            return AttentionPooler(hyper_output_dim)
        if pooling_name != "hierarchical":
            msg = (
                f"Unknown hyper_head.pooling {pooling_name!r}. "
                "Expected one of ['attention', 'hierarchical']."
            )
            raise ValueError(msg)

        interaction_num_heads = _resolve_positive_int(
            hyper_head.get("interaction_num_heads", 1),
            "hyper_head.interaction_num_heads",
        )
        if hyper_output_dim % interaction_num_heads != 0:
            msg = (
                "hyper_head.interaction_num_heads must divide hyper_output_dim, got "
                f"{interaction_num_heads} for hyper_output_dim={hyper_output_dim}."
            )
            raise ValueError(msg)

        segment_interaction_layers = _resolve_positive_int(
            hyper_head.get("segment_interaction_layers", 1),
            "hyper_head.segment_interaction_layers",
        )
        example_interaction_layers = _resolve_positive_int(
            hyper_head.get("example_interaction_layers", 1),
            "hyper_head.example_interaction_layers",
        )
        dropout = _resolve_dropout(
            hyper_head.get("dropout", 0.0),
            "hyper_head.dropout",
        )
        return HierarchicalPooler(
            hidden_dim=hyper_output_dim,
            num_heads=interaction_num_heads,
            segment_interaction_layers=segment_interaction_layers,
            example_interaction_layers=example_interaction_layers,
            dropout=dropout,
        )

    def build_hypernetwork(
        self,
        hyper_model: Mapping,
        embedding_dim: int,
    ) -> tuple[torch.nn.Module, int]:
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
            {"input_dim": embedding_dim},
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

    def build_target_model(self, target_model: Mapping, embedding_dim: int) -> torch.nn.Module:
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
            {
                "input_dim": embedding_dim,
                "output_dim": self.target_output_dim,
                **target_model_spec["owned_params"],
            },
            "target_model",
        )
        target_model_cls = target_model_spec["class"]
        return target_model_cls(**resolved_params)

    def build_task_context_token_ids(
        self,
        support_inputs: torch.Tensor,
        support_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build token ids for the serialized support-only hypernetwork input (6 segments)."""
        segments = [
            support_inputs[:, 0],
            support_outputs[:, 0],
            support_inputs[:, 1],
            support_outputs[:, 1],
            support_inputs[:, 2],
            support_outputs[:, 2],
        ]
        segment_example_ids = [0, 0, 1, 1, 2, 2]
        segment_role_ids = [0, 1, 0, 1, 0, 1]

        num_segments = len(segments)
        segment_values = torch.stack(segments, dim=1)
        batch_size, _, sequence_length = segment_values.shape
        flat_values = segment_values.reshape(batch_size, -1).long()

        position_ids = (
            torch.arange(sequence_length, device=segment_values.device)
            .repeat(num_segments)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        example_ids = (
            support_inputs.new_tensor(segment_example_ids, dtype=torch.long)
            .repeat_interleave(sequence_length)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        role_ids = (
            support_inputs.new_tensor(segment_role_ids, dtype=torch.long)
            .repeat_interleave(sequence_length)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        return flat_values, position_ids, example_ids, role_ids

    def build_target_token_ids(
        self,
        support_inputs: torch.Tensor,
        query_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build value and position token ids for all target-model input sequences."""
        all_inputs = torch.cat([support_inputs, query_input.unsqueeze(1)], dim=1)
        batch_size, num_examples, sequence_length = all_inputs.shape

        value_ids = all_inputs.long()
        position_ids = (
            torch.arange(sequence_length, device=all_inputs.device)
            .view(1, 1, -1)
            .expand(batch_size, num_examples, -1)
        )
        return value_ids, position_ids

    def prepare_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a task batch into hypernetwork features, target inputs, and targets."""
        support_inputs = batch["support_inputs"]
        support_outputs = batch["support_outputs"]
        query_input = batch["query_input"]
        query_output = batch["query_output"]

        # Hypernetwork sees support pairs only (6 segments): value + pos + example + role
        flat_values, position_ids, example_ids, role_ids = self.build_task_context_token_ids(
            support_inputs, support_outputs
        )
        task_features = self.shared_task_token_embedder(
            flat_values, position_ids, example_ids, role_ids
        )

        # Target model sees value + position only (no task-specific metadata)
        t_value_ids, t_pos_ids = self.build_target_token_ids(support_inputs, query_input)
        example_inputs = self.shared_task_token_embedder(t_value_ids, t_pos_ids)

        # Targets follow the same 4-example layout as the logits.
        example_targets = torch.cat([support_outputs, query_output.unsqueeze(1)], dim=1)

        return task_features, example_inputs, example_targets

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """batch -> (logits, targets) with shape (batch, 4, seq_len)."""
        task_features, example_inputs, example_targets = self.prepare_inputs(batch)
        canonical_ids = None
        if self.hypermodel.task_indicator_proj is not None:
            canonical_ids = torch.tensor(
                [TASK_CATEGORY_INDEX[c] for c in batch["task_category"]],
                device=self.device,
            )
        logits = self.hypermodel(task_features, example_inputs, task_ids=canonical_ids)
        return logits, example_targets

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode canonical logits to integer predictions."""
        if self.is_binary_task:
            return (torch.sigmoid(logits) >= 0.5).long()
        return logits.argmax(dim=-1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the task loss for backprop."""
        if self.is_binary_task:
            base_loss = F.binary_cross_entropy_with_logits(logits, targets)
        else:
            ignore_index = self.padding_idx if self.padding_idx is not None else -100
            base_loss = F.cross_entropy(
                logits.reshape(-1, self.num_classes),
                targets.long().reshape(-1),
                ignore_index=ignore_index,
            )
        return base_loss

    def _metric_sums(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw metric numerators and denominators for one task batch."""
        predictions = self.decode_logits(logits)
        targets_long = targets.long()
        if self.padding_idx is None:
            valid_mask = torch.ones_like(targets_long, dtype=torch.bool)
        else:
            valid_mask = targets_long != self.padding_idx

        matches = predictions == targets_long
        exact_matches = (matches | ~valid_mask).all(dim=2)

        support_valid = valid_mask[:, :NUM_SUPPORT_EXAMPLES]
        query_valid = valid_mask[:, NUM_SUPPORT_EXAMPLES:]
        support_matches = matches[:, :NUM_SUPPORT_EXAMPLES] & support_valid
        query_matches = matches[:, NUM_SUPPORT_EXAMPLES:] & query_valid

        metric_value = targets_long.new_tensor
        return {
            "support_correct": support_matches.sum().float(),
            "support_total": support_valid.sum().float(),
            "support_exact": exact_matches[:, :NUM_SUPPORT_EXAMPLES].sum().float(),
            "support_examples": metric_value(
                float(exact_matches[:, :NUM_SUPPORT_EXAMPLES].numel()),
                dtype=torch.float32,
            ),
            "query_correct": query_matches.sum().float(),
            "query_total": query_valid.sum().float(),
            "query_exact": exact_matches[:, NUM_SUPPORT_EXAMPLES:].sum().float(),
            "query_examples": metric_value(
                float(exact_matches[:, NUM_SUPPORT_EXAMPLES:].numel()),
                dtype=torch.float32,
            ),
            "all_examples_exact": exact_matches.all(dim=1).sum().float(),
            "task_count": metric_value(float(exact_matches.shape[0]), dtype=torch.float32),
        }

    def _metrics_from_sums(
        self,
        metric_sums: dict[str, torch.Tensor],
        prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Convert raw metric sums into logged ratios."""
        key = f"{prefix}_" if prefix else ""

        def ratio(numerator: str, denominator: str) -> torch.Tensor:
            return metric_sums[numerator] / metric_sums[denominator].clamp_min(1.0)

        metrics = {
            f"{key}support_accuracy": ratio("support_correct", "support_total"),
            f"{key}support_exact_match": ratio("support_exact", "support_examples"),
            f"{key}query_accuracy": ratio("query_correct", "query_total"),
            f"{key}query_exact_match": ratio("query_exact", "query_examples"),
        }
        if prefix == "train":
            metrics[f"{key}all_examples_exact_match"] = ratio(
                "all_examples_exact",
                "task_count",
            )
        return metrics

    def _reset_epoch_metric_totals(self, prefix: str) -> None:
        if prefix == "val":
            self._val_metric_totals = {
                "support_correct": 0.0,
                "support_total": 0.0,
                "support_exact": 0.0,
                "support_examples": 0.0,
                "query_correct": 0.0,
                "query_total": 0.0,
                "query_exact": 0.0,
                "query_examples": 0.0,
            }
        elif prefix == "test":
            self._test_metric_totals = {
                "support_correct": 0.0,
                "support_total": 0.0,
                "support_exact": 0.0,
                "support_examples": 0.0,
                "query_correct": 0.0,
                "query_total": 0.0,
                "query_exact": 0.0,
                "query_examples": 0.0,
            }

    def _accumulate_epoch_metric_totals(
        self,
        prefix: str,
        metric_sums: dict[str, torch.Tensor],
    ) -> None:
        if prefix not in {"val", "test"}:
            return

        totals = self._val_metric_totals if prefix == "val" else self._test_metric_totals
        for key in totals:
            totals[key] += float(metric_sums[key].detach().cpu().item())

    def _gather_epoch_metric_totals(self, prefix: str) -> dict[str, float]:
        totals = dict(self._val_metric_totals if prefix == "val" else self._test_metric_totals)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return totals

        gathered_payloads: list[dict[str, float] | None] = [
            None
        ] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_payloads, totals)

        merged_totals = {key: 0.0 for key in totals}
        for payload in gathered_payloads:
            if payload is None:
                continue
            for key, value in payload.items():
                merged_totals[key] += float(value)
        return merged_totals

    def _log_epoch_metrics(self, prefix: str) -> None:
        totals = self._gather_epoch_metric_totals(prefix)
        metric_names = {
            f"{prefix}_support_accuracy": ("support_correct", "support_total"),
            f"{prefix}_support_exact_match": ("support_exact", "support_examples"),
            f"{prefix}_query_accuracy": ("query_correct", "query_total"),
            f"{prefix}_query_exact_match": ("query_exact", "query_examples"),
        }

        for name, (numerator_key, denominator_key) in metric_names.items():
            denominator = totals[denominator_key]
            if denominator <= 0:
                continue
            self.log(
                name,
                totals[numerator_key] / denominator,
                on_step=False,
                on_epoch=True,
                prog_bar=name.endswith("query_exact_match"),
                batch_size=max(1, int(round(denominator))),
                sync_dist=False,
            )

    def _accumulate_query_exact_match_by_task_category(
        self,
        batch: dict,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate validation query exact-match totals grouped by task category."""
        predictions = self.decode_logits(logits)
        targets_long = targets.long()
        if self.padding_idx is not None:
            valid_mask = targets_long != self.padding_idx
            exact_matches = ((predictions == targets_long) | ~valid_mask).all(dim=2)
        else:
            exact_matches = (predictions == targets_long).all(dim=2)
        query_exact = exact_matches[:, NUM_SUPPORT_EXAMPLES].float()
        for task_category, val in zip(
            batch["task_category"],
            query_exact.detach().cpu().tolist(),
            strict=True,
        ):
            self._val_query_exact_match_totals_by_task[task_category] = (
                self._val_query_exact_match_totals_by_task.get(task_category, 0.0) + float(val)
            )
            self._val_query_exact_match_counts_by_task[task_category] = (
                self._val_query_exact_match_counts_by_task.get(task_category, 0) + 1
            )

    def _gather_query_exact_match_by_task_category(
        self,
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Gather per-task-category query exact-match totals across distributed ranks."""
        totals = dict(self._val_query_exact_match_totals_by_task)
        counts = dict(self._val_query_exact_match_counts_by_task)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return totals, counts
        gathered: list[dict | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, {"totals": totals, "counts": counts})
        merged_totals: dict[str, float] = {}
        merged_counts: dict[str, int] = {}
        for payload in gathered:
            if payload is None:
                continue
            for cat, v in payload["totals"].items():
                merged_totals[cat] = merged_totals.get(cat, 0.0) + float(v)
            for cat, v in payload["counts"].items():
                merged_counts[cat] = merged_counts.get(cat, 0) + int(v)
        return merged_totals, merged_counts

    def _actual_seq_len(self, sequence: torch.Tensor | list[int]) -> int:
        """Return the unpadded length of one serialized sequence."""
        if isinstance(sequence, torch.Tensor):
            values = sequence.detach().cpu().tolist()
        else:
            values = list(sequence)
        if self.padding_idx is None:
            return len(values)
        n = len(values)
        while n > 0 and int(values[n - 1]) == self.padding_idx:
            n -= 1
        return n if n > 0 else len(values)

    def _sequence_accuracy(
        self,
        target_sequence: list[int],
        prediction_sequence: list[int],
    ) -> float:
        """Return token accuracy over an already-trimmed sequence."""
        if not target_sequence:
            return 0.0
        correct = sum(
            int(prediction == target)
            for target, prediction in zip(target_sequence, prediction_sequence, strict=True)
        )
        return correct / len(target_sequence)

    def common_step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits, targets = self(batch)
        loss = self.compute_loss(logits, targets)
        metric_sums = self._metric_sums(logits, targets)
        metrics = self._metrics_from_sums(metric_sums, prefix=prefix)
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
        if prefix == "train":
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
        else:
            self._accumulate_epoch_metric_totals(prefix, metric_sums)
            if prefix == "val":
                self._accumulate_query_exact_match_by_task_category(batch, logits, targets)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        self.common_step(batch, prefix="test")

    def on_validation_epoch_start(self) -> None:
        self._reset_epoch_metric_totals("val")
        self._val_query_exact_match_totals_by_task = {}
        self._val_query_exact_match_counts_by_task = {}

    def on_validation_epoch_end(self) -> None:
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        self._log_epoch_metrics("val")
        category_totals, category_counts = self._gather_query_exact_match_by_task_category()
        for task_category, count in category_counts.items():
            if count < 1:
                continue
            self.log(
                f"val_query_exact_match_by_task_{_task_category_metric_suffix(task_category)}",
                category_totals[task_category] / count,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=count,
                sync_dist=sync_dist,
            )

    def on_test_epoch_start(self) -> None:
        self._reset_epoch_metric_totals("test")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def predict_batch(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, integer predictions, and targets for a task batch."""
        logits, targets = self(batch)
        predictions = self.decode_logits(logits)
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
        support_predictions = predictions[:, :NUM_SUPPORT_EXAMPLES].detach().cpu().long().tolist()
        query_inputs = batch["query_input"].detach().cpu().long().tolist()
        query_targets = targets[:, NUM_SUPPORT_EXAMPLES].detach().cpu().long().tolist()
        query_predictions = predictions[:, NUM_SUPPORT_EXAMPLES].detach().cpu().long().tolist()
        task_categories = list(batch["task_category"])
        raw_ids = batch["task_id"]
        task_ids = (
            raw_ids.detach().cpu().tolist() if isinstance(raw_ids, torch.Tensor) else list(raw_ids)
        )

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
            trimmed_support_inputs = []
            trimmed_support_targets = []
            trimmed_support_predictions = []
            for example_input, example_target, example_prediction in zip(
                support_input,
                support_target,
                support_prediction,
                strict=True,
            ):
                n = self._actual_seq_len(example_input)
                trimmed_support_inputs.append([int(value) for value in example_input[:n]])
                trimmed_support_targets.append([int(value) for value in example_target[:n]])
                trimmed_support_predictions.append(
                    [int(value) for value in example_prediction[:n]]
                )

            query_n = self._actual_seq_len(query_input)
            trimmed_query_input = [int(value) for value in query_input[:query_n]]
            trimmed_query_target = [int(value) for value in query_target[:query_n]]
            trimmed_query_prediction = [int(value) for value in query_prediction[:query_n]]
            query_exact_match = trimmed_query_prediction == trimmed_query_target
            query_accuracy = self._sequence_accuracy(
                trimmed_query_target,
                trimmed_query_prediction,
            )
            records.append(
                {
                    "support_inputs": trimmed_support_inputs,
                    "support_outputs": trimmed_support_targets,
                    "support_predictions": trimmed_support_predictions,
                    "query_input": trimmed_query_input,
                    "query_output": trimmed_query_target,
                    "query_prediction": trimmed_query_prediction,
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
    ) -> list[tuple[str, int]]:
        """Select at most one recurring visual-logging task per category."""
        if limit < 1:
            return []

        seen_categories = set()
        selected = []
        for task in dataset.tasks:
            task_category = task["task_category"]
            if task_category in seen_categories:
                continue
            seen_categories.add(task_category)
            selected.append((task_category, task["task_id"]))
            if len(selected) >= limit:
                break

        self.selected_representative_task_ids[split_name] = selected
        return selected

    def collect_task_records_from_dataset_by_task_ids(
        self,
        dataset,
        task_ids: list[tuple[str, int]],
    ) -> list[dict]:
        """Collect prediction records for a fixed set of (category, task_id) pairs."""
        if not task_ids:
            return []

        selected_records = []
        task_id_set = set(task_ids)
        for task in dataset.tasks:
            key = (task["task_category"], task["task_id"])
            if key not in task_id_set:
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

        selected_records.sort(
            key=lambda record: task_ids.index((record["task_category"], record["task_id"]))
        )
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
            scheduler = self._build_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return optimizer

    def _build_scheduler(self, optimizer):
        sched_cls = getattr(torch.optim.lr_scheduler, self.lr_scheduler_cfg["name"])
        params = dict(self.lr_scheduler_cfg.get("params", {}))
        if self.warmup_steps > 0:
            if "T_max" in params:
                params["T_max"] = max(1, params["T_max"] - self.warmup_steps)
            main_scheduler = sched_cls(optimizer, **params)
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_steps
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, main_scheduler], milestones=[self.warmup_steps]
            )
        return sched_cls(optimizer, **params)

    def on_fit_start(self) -> None:
        print(repr(self.hypermodel))
        n = sum(p.numel() for p in self.hypermodel.parameters() if p.requires_grad)
        print(f"Trainable params : {n:,}")

"""Lightning training wrapper for the simplified HyperModel path."""

import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from muon import SingleDeviceMuonWithAuxAdam
from torch.nn.utils import clip_grad_norm_

import wandb
from models.canon_transformer import CanonTransformer
from models.cnn import CNN
from models.hypermodel import AttentionPooler, HierarchicalPooler, HyperModel
from models.rnn import RNN
from models.rope_looped_transformer import (
    RoPECanonLoopedTransformer,
    RoPECanonTransformer,
    RoPECanonZhuTransformer,
)
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
    "canon_transformer": {
        "class": CanonTransformer,
        "output_dim_key": "output_dim",
    },
    "rope_canon_transformer": {
        "class": RoPECanonTransformer,
        "output_dim_key": "output_dim",
    },
    "rope_canon_zhu_transformer": {
        "class": RoPECanonZhuTransformer,
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
    "rope_canon_looped_transformer": {
        "class": RoPECanonLoopedTransformer,
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
    supports_embedding_visualization = True

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
        optimizer_name: str | None = None,
        weight_decay: float = 0.01,
        lr_scheduler: dict | None = None,
        warmup_steps: int = 0,
        N_supervision: int = 1,
        gradient_clip_val: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.N_supervision = N_supervision
        self.gradient_clip_val = gradient_clip_val
        # Loss-spike diagnostics: per-training-step JSONL of inner-loop loss/grad-norm
        # traces plus batch composition, written next to the run's other outputs.
        self._diagnostics_path = (
            os.path.join(kwargs["output_path"], "train_step_diagnostics.jsonl")
            if kwargs.get("output_path")
            else None
        )
        self.prediction_task, self.num_classes = self.resolve_prediction_task(
            prediction_task, num_classes
        )
        self.target_output_dim = 1 if self.is_binary_task else self.num_classes
        self.padding_idx: int | None = kwargs.get("padding_idx")
        embedding_dim, value_vocab_size, use_sinusoidal_pe = self._resolve_embedding_params(
            task_encoding, kwargs
        )
        self.embedding_dim = embedding_dim
        self.shared_task_token_embedder = TaskTokenEmbedder(
            embedding_dim=embedding_dim,
            value_vocab_size=value_vocab_size,
            use_sinusoidal_pe=use_sinusoidal_pe,
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
        low_rank_rank = int(hyper_head_cfg.get("low_rank_rank", 1))
        lora_adapter = bool(hyper_head_cfg.get("lora_adapter", False))
        lora_adapter_rank_cfg = hyper_head_cfg.get("lora_adapter_rank")
        lora_adapter_rank = 1 if lora_adapter_rank_cfg is None else int(lora_adapter_rank_cfg)
        lora_adapter_train_backbone = bool(
            hyper_head_cfg.get("lora_adapter_train_backbone", False)
        )
        lora_adapter_zero_backbone = bool(
            hyper_head_cfg.get("lora_adapter_zero_backbone", False)
        )
        freeze_task_indicator = bool(hyper_head_cfg.get("freeze_task_indicator", False))
        variational = bool(hyper_head_cfg.get("variational", False))
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
            low_rank_rank=low_rank_rank,
            lora_adapter=lora_adapter,
            lora_adapter_rank=lora_adapter_rank,
            lora_adapter_train_backbone=lora_adapter_train_backbone,
            lora_adapter_zero_backbone=lora_adapter_zero_backbone,
            freeze_task_indicator=freeze_task_indicator,
            variational=variational,
        )
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name or optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler
        self.warmup_steps = warmup_steps
        # Muon-specific: only used when optimizer_name == "Muon" (see configure_optimizers).
        # Keller Jordan's published defaults, not tuned for this repo.
        self.muon_lr = kwargs.get("muon_lr", 0.02)
        self.muon_momentum = kwargs.get("muon_momentum", 0.95)
        # If True, also route the LoRA head's A/B factor projections (lora_proj_a/lora_proj_b)
        # to the AdamW aux group instead of Muon -- see _build_muon_param_groups.
        self.muon_exclude_lora_heads = bool(kwargs.get("muon_exclude_lora_heads", False))
        # Beta-VAE KL multiplier; only used when hyper_head.variational is True. See
        # HyperModel._apply_variational_bottleneck for where the KL term itself is computed,
        # and training_step/_current_kl_beta for where/how it's applied.
        self.kl_beta = float(kwargs.get("kl_beta", 0.0))
        # "constant": kl_beta applied unchanged from step 0. "cosine": ramps from 0 up to
        # kl_beta over kl_beta_warmup_steps (a cosine ease-in, not the LR warmup_steps -- the
        # LR scheduler advances once per training_step/batch, while this ramp is keyed to
        # self.global_step, which -- like this repo's own max_steps -- advances once per
        # manual opt.step() call, i.e. N_supervision times per batch. Using self.global_step
        # here is intentional and matches that existing convention, not a separate counter).
        self.kl_beta_anneal = kwargs.get("kl_beta_anneal", "constant")
        self.kl_beta_warmup_steps = int(kwargs.get("kl_beta_warmup_steps", 0))
        self.log_task_examples = kwargs.get("log_task_examples", False)
        self.log_task_examples_every_n_epochs = kwargs.get("log_task_examples_every_n_epochs", 25)
        self.log_embedding_clusters = kwargs.get("log_embedding_clusters", False)
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
    ) -> tuple[int, int, bool]:
        """Return (embedding_dim, value_vocab_size, use_sinusoidal_pe) from config."""
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

        use_sinusoidal_pe = bool(task_encoding.get("use_sinusoidal_pe", True))

        return embedding_dim, value_vocab_size, use_sinusoidal_pe

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

    def _accumulate_query_accuracy_by_task_category(
        self,
        batch: dict,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate validation query token-accuracy totals grouped by task category.

        Companion to `_accumulate_query_exact_match_by_task_category`: same masked
        prediction/target setup, but averages correctness over token positions instead of
        requiring every position to match, so a category can show partial credit even when
        its exact-match rate is 0.
        """
        predictions = self.decode_logits(logits)
        targets_long = targets.long()
        if self.padding_idx is not None:
            valid_mask = targets_long != self.padding_idx
            correct = (predictions == targets_long) | ~valid_mask
        else:
            correct = predictions == targets_long
        query_accuracy = correct[:, NUM_SUPPORT_EXAMPLES].float().mean(dim=1)
        for task_category, val in zip(
            batch["task_category"],
            query_accuracy.detach().cpu().tolist(),
            strict=True,
        ):
            self._val_query_accuracy_totals_by_task[task_category] = (
                self._val_query_accuracy_totals_by_task.get(task_category, 0.0) + float(val)
            )
            self._val_query_accuracy_counts_by_task[task_category] = (
                self._val_query_accuracy_counts_by_task.get(task_category, 0) + 1
            )

    def _gather_query_accuracy_by_task_category(
        self,
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Gather per-task-category query token-accuracy totals across distributed ranks."""
        totals = dict(self._val_query_accuracy_totals_by_task)
        counts = dict(self._val_query_accuracy_counts_by_task)
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
        if self.hypermodel.variational:
            # Visibility only -- {prefix}_loss above stays reconstruction-only, so
            # primary_metric-based checkpoint selection is unaffected and stays comparable
            # across every beta in the sweep (and every other experiment in the repo).
            self.log(f"{prefix}_kl_loss", self.hypermodel._last_kl_loss, **log_kwargs)
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
                self._accumulate_query_accuracy_by_task_category(batch, logits, targets)
        return loss

    def _current_kl_beta(self) -> float:
        """Return the KL weight for the step about to be taken.

        "constant": kl_beta from step 0. "cosine": eases in from 0 to kl_beta over
        kl_beta_warmup_steps, keyed to self.global_step -- the count of completed manual
        opt.step() calls so far (see the comment on self.kl_beta_anneal in __init__ for why
        this, not a separate counter, is the right clock to use here).
        """
        if self.kl_beta_anneal != "cosine" or self.kl_beta_warmup_steps <= 0:
            return self.kl_beta
        step = self.global_step
        if step >= self.kl_beta_warmup_steps:
            return self.kl_beta
        progress = step / self.kl_beta_warmup_steps
        return self.kl_beta * 0.5 * (1 - math.cos(math.pi * progress))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        total_recon_loss = 0.0
        total_backward_loss = 0.0
        total_kl_loss = 0.0
        total_kl_beta_used = 0.0
        inner_losses = []
        inner_grad_norms = []
        inner_gen_weight_norms = []
        inner_boundary_grad_norms = []
        logits, targets = None, None
        for _ in range(self.N_supervision):
            logits, targets = self(batch)
            gen_weight_norm = self.hypermodel._last_generated_weight_norm
            recon_loss = self.compute_loss(logits, targets)
            if self.hypermodel.variational:
                kl_loss = self.hypermodel._last_kl_loss
                kl_beta_used = self._current_kl_beta()
                backward_loss = recon_loss + kl_beta_used * kl_loss
                total_kl_beta_used += kl_beta_used
            else:
                kl_loss = None
                backward_loss = recon_loss
            opt.zero_grad()
            self.manual_backward(backward_loss)
            # Always measure the pre-clip gradient norm, even when no clip value
            # is configured: clip_grad_norm_ with max_norm=inf never rescales
            # (the coefficient is always >= 1), so this is a no-op for existing
            # unclipped runs while making the norm observable for diagnostics.
            clip_value = self.gradient_clip_val if self.gradient_clip_val is not None else float("inf")
            grad_norm = clip_grad_norm_(self.parameters(), clip_value)
            if not torch.isfinite(grad_norm):
                # Stop immediately rather than continuing to train (and burn GPU time) on a
                # diverged run: once the gradient norm itself is NaN/Inf, every subsequent
                # step's weights, loss, and metrics are garbage too, and clipping cannot
                # protect against this for optimizers (e.g. Muon) whose update magnitude is
                # decoupled from the raw gradient norm.
                msg = (
                    f"Non-finite gradient norm ({grad_norm.item()}) at "
                    f"global_step={self.global_step}. Stopping to avoid wasting compute on a "
                    "diverged run."
                )
                raise RuntimeError(msg)
            opt.step()
            total_recon_loss += recon_loss.detach()
            total_backward_loss += backward_loss.detach()
            if kl_loss is not None:
                total_kl_loss += kl_loss.detach()
            inner_losses.append(recon_loss.detach().item())
            inner_grad_norms.append(grad_norm.item())
            inner_gen_weight_norms.append(gen_weight_norm.mean().item())
            boundary_grad_norm = self.hypermodel._last_boundary_grad_norm
            inner_boundary_grad_norms.append(
                boundary_grad_norm.mean().item() if boundary_grad_norm is not None else None
            )

        if sch is not None:
            sch.step()

        # Reconstruction-only, unchanged semantics: comparable across every optimizer/arm in
        # the repo regardless of whether this run has a VAE bottleneck or what kl_beta is.
        avg_loss = total_recon_loss / self.N_supervision
        metric_sums = self._metric_sums(logits, targets)
        metrics = self._metrics_from_sums(metric_sums, prefix="train")
        batch_size = batch["support_inputs"].shape[0]
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        log_kwargs = {
            "on_step": True,
            "on_epoch": True,
            "batch_size": batch_size,
            "sync_dist": sync_dist,
        }
        self.log("train_loss", avg_loss, prog_bar=True, **log_kwargs)
        if self.hypermodel.variational:
            self.log("train_kl_loss", total_kl_loss / self.N_supervision, **log_kwargs)
            self.log("train_elbo_loss", total_backward_loss / self.N_supervision, **log_kwargs)
            self.log("train_kl_beta", total_kl_beta_used / self.N_supervision, **log_kwargs)
        self.log("train_grad_norm_max", max(inner_grad_norms), **log_kwargs)
        self.log("train_grad_norm_mean", sum(inner_grad_norms) / len(inner_grad_norms), **log_kwargs)
        for name, value in metrics.items():
            self.log(
                name,
                value,
                prog_bar=name.endswith("query_exact_match"),
                **log_kwargs,
            )
        self._write_step_diagnostics(
            batch=batch,
            batch_idx=batch_idx,
            batch_size=batch_size,
            inner_losses=inner_losses,
            inner_grad_norms=inner_grad_norms,
            inner_gen_weight_norms=inner_gen_weight_norms,
            inner_boundary_grad_norms=inner_boundary_grad_norms,
        )
        return avg_loss

    def _write_step_diagnostics(
        self,
        batch,
        batch_idx,
        batch_size,
        inner_losses,
        inner_grad_norms,
        inner_gen_weight_norms,
        inner_boundary_grad_norms,
    ):
        """Append one JSONL record per training step for loss-spike diagnosis.

        Captures per-inner-supervision-step loss/grad-norm (otherwise averaged
        away into a single `train_loss` point) plus the batch's task-category
        and task-id composition, so spikes can be correlated post-hoc against
        gradient explosion, epoch position, and specific data items.

        `inner_gen_weight_norms`/`inner_boundary_grad_norms` sit at the boundary
        between the hypernetwork (encoder/pooling/projection) and the target
        model's vmapped forward: `gen_weight_norm` is the magnitude of the
        weights the hypernetwork generated for the target model that step, and
        `boundary_grad_norm` is the gradient arriving at that same point during
        backward. Comparing `boundary_grad_norm` against the final whole-model
        `inner_grad_norms` localizes an explosion: if they're already comparable
        at the boundary, the amplification happened inside the target-model
        loop; if the boundary is normal but the final norm is huge, it happened
        upstream in the hypernetwork encoder/projection instead.
        """
        if self._diagnostics_path is None:
            return
        if not (self.trainer is None or self.trainer.is_global_zero):
            return
        record = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "inner_losses": inner_losses,
            "inner_grad_norms": inner_grad_norms,
            "inner_gen_weight_norms": inner_gen_weight_norms,
            "inner_boundary_grad_norms": inner_boundary_grad_norms,
            "task_category_counts": dict(Counter(batch["task_category"])),
            "task_ids": list(batch["task_id"]),
        }
        with open(self._diagnostics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        self.common_step(batch, prefix="test")

    def on_validation_epoch_start(self) -> None:
        self._reset_epoch_metric_totals("val")
        self._val_query_exact_match_totals_by_task = {}
        self._val_query_exact_match_counts_by_task = {}
        self._val_query_accuracy_totals_by_task = {}
        self._val_query_accuracy_counts_by_task = {}

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
        accuracy_totals, accuracy_counts = self._gather_query_accuracy_by_task_category()
        for task_category, count in accuracy_counts.items():
            if count < 1:
                continue
            self.log(
                f"val_query_accuracy_by_task_{_task_category_metric_suffix(task_category)}",
                accuracy_totals[task_category] / count,
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

    def collect_embedding_records_from_dataloader(
        self,
        dataloader,
        limit: int | None = None,
    ) -> list[dict]:
        """Run inference and collect per-task pooled-embedding records.

        Only the pooled task latent that actually drove weight generation is captured
        (`hypermodel._last_task_representation`, stashed as a forward side effect), not
        predictions -- used for the end-of-run disentanglement cluster-map diagnostic.
        """
        self.eval()
        device = next(self.parameters()).device
        records = []

        with torch.no_grad():
            for batch in dataloader:
                tensor_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                self(tensor_batch)
                pooled = self.hypermodel._last_task_representation.cpu()
                task_categories = list(batch["task_category"])
                raw_ids = batch["task_id"]
                task_ids = (
                    raw_ids.detach().cpu().tolist()
                    if isinstance(raw_ids, torch.Tensor)
                    else list(raw_ids)
                )
                for index, (task_category, task_id) in enumerate(
                    zip(task_categories, task_ids, strict=True)
                ):
                    records.append(
                        {
                            "task_category": task_category,
                            "task_id": task_id,
                            "pooled_embedding": pooled[index],
                        }
                    )
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

    # Modules whose weight is an nn.Linear but is not a "hidden weight matrix" in the sense
    # Muon is designed for: input_projection/output_head are the hypernetwork's input
    # embedding and final output layer (Keller Jordan's own guidance excludes both from
    # Muon), task_indicator_proj is a one-hot task-identity embedding table implemented
    # as nn.Linear(num_tasks, hyper_output_dim, bias=False) -- structurally 2D but
    # semantically an embedding, not a matrix operator -- and vae_mu_head/vae_logvar_head
    # (beta-VAE bottleneck heads) produce/consume the pooled task representation itself,
    # not an internal hidden-to-hidden interaction matrix, same rationale as
    # task_indicator_proj. This last exclusion is a design call, not forced by Keller
    # Jordan's own guidance. Always excluded, regardless of muon_exclude_lora_heads.
    _MUON_EXCLUDED_MODULE_NAMES = (
        "input_projection",
        "output_head",
        "task_indicator_proj",
        "vae_mu_head",
        "vae_logvar_head",
    )
    # lora_proj_a/lora_proj_b (the LoRA head's low-rank A/B factor generators, see
    # models/hypermodel.py's HyperModel.__init__) are Muon-eligible by default -- Fabio's
    # original call was that they're most of where this model's trainable capacity sits.
    # muon_exclude_lora_heads=True instead routes them to AdamW, to test whether Muon's
    # orthogonalization is actually a poor fit for these particular projections.
    # lora_proj_other (generates the non-matrix target params) is never excluded by this
    # flag -- it isn't "the low rank vectors", it has no low-rank structure to speak of.
    _MUON_LORA_HEAD_MODULE_NAMES = ("lora_proj_a", "lora_proj_b")

    def _build_muon_param_groups(self) -> list[dict]:
        """Split trainable params into a Muon group (hidden nn.Linear weights) and an AdamW
        aux group (everything else: embeddings, norms, Canon conv weights, the attention
        pooler's query vector, biases, the excluded Linear layers above, and -- if
        muon_exclude_lora_heads -- the LoRA head's A/B projections)."""
        excluded_names = set(self._MUON_EXCLUDED_MODULE_NAMES)
        if self.muon_exclude_lora_heads:
            excluded_names |= set(self._MUON_LORA_HEAD_MODULE_NAMES)

        muon_params = []
        muon_param_ids = set()
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Match against every dotted path segment, not just the last one: lora_proj_a/
            # lora_proj_b are nn.ModuleLists, so a member's name ends in its list index (e.g.
            # "lora_proj_a.3"), not the attribute name -- checking only name.split(".")[-1]
            # would silently fail to exclude them.
            if set(name.split(".")) & excluded_names:
                continue
            if module.weight.requires_grad:
                muon_params.append(module.weight)
                muon_param_ids.add(id(module.weight))

        adam_params = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]

        return [
            dict(
                params=adam_params,
                use_muon=False,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            ),
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.muon_lr,
                momentum=self.muon_momentum,
                weight_decay=self.weight_decay,
            ),
        ]

    def configure_optimizers(self):
        if self.optimizer_name == "Muon":
            optimizer = SingleDeviceMuonWithAuxAdam(self._build_muon_param_groups())
        else:
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

"""Lightning training wrapper for training-recursion experiments (Condition C / D).

N_supervision backward passes are run on the same batch per training step,
with the same model and same input each time. No architectural changes to the
forward pass — the only difference from DirectSupervisedLightning is that
parameters are updated N_supervision times per batch instead of once.

This isolates training recursion from architectural recursion (weight sharing).
See models/recursive_transformer.py for the architectural counterpart.
"""

import re

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from metrics import accuracy, exact_match_accuracy
from models.canon_transformer import CanonRecursiveTransformer, CanonTransformer
from models.cnn import CNN
from models.recursive_transformer import RecursiveTransformer
from models.rnn import RNN
from models.task_token_embedder import TaskTokenEmbedder
from models.transformer import Transformer
from visualisation import figure_to_wandb_image, render_val_example_figure

PREDICTION_TASK_BINARY = "binary"
PREDICTION_TASK_MULTICLASS = "multiclass"


def _task_category_metric_suffix(task_category: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", task_category)


class LoopedSupervisedLightning(pl.LightningModule):
    """Training-recursion variant: N_supervision optimizer steps per batch.

    The model architecture is unchanged relative to DirectSupervisedLightning.
    The same forward pass is run N_supervision times on each batch, with a
    gradient update after each pass. This tests whether repeatedly adapting
    parameters to the same example improves performance.

    Pair with backbone_model.name=recursive_transformer for Condition D
    (combined architectural + training recursion).
    """

    def __init__(
        self,
        backbone_model: dict,
        task_encoding: dict,
        prediction_task: str = PREDICTION_TASK_MULTICLASS,
        num_classes: int | None = None,
        input_dim: int = 33,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_name: str = "Adam",
        weight_decay: float = 0.01,
        lr_scheduler: dict | None = None,
        warmup_steps: int = 0,
        non_background_loss_weight: float = 1.0,
        N_supervision: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False

        if (
            not isinstance(non_background_loss_weight, int | float)
            or non_background_loss_weight <= 0
        ):
            msg = "non_background_loss_weight must be a positive number."
            raise ValueError(msg)

        self.prediction_task = prediction_task
        self.num_classes = num_classes
        self.N_supervision = N_supervision
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name or optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler
        self.warmup_steps = warmup_steps
        self.non_background_loss_weight = float(non_background_loss_weight)
        self.gradient_clip_val: float | None = kwargs.get("gradient_clip_val")
        self.log_task_examples = kwargs.get("log_task_examples", False)
        self.log_task_examples_every_n_epochs = kwargs.get("log_task_examples_every_n_epochs", 100)
        self.supports_hard_val_examples = False

        embedding_dim: int = task_encoding["embedding_dim"]
        value_vocab_size: int = task_encoding.get("value_vocab_size", 2)
        self.padding_idx: int | None = kwargs.get("padding_idx")

        self.embedder = TaskTokenEmbedder(
            embedding_dim=embedding_dim,
            value_vocab_size=value_vocab_size,
            padding_idx=self.padding_idx,
        )

        self.backbone, self.hidden_dim = self._build_backbone(
            backbone_model, embedding_dim=embedding_dim, seq_len=input_dim
        )

        target_output_dim = 1 if prediction_task == PREDICTION_TASK_BINARY else num_classes
        self.head = nn.Linear(self.hidden_dim, target_output_dim, bias=False)

        self._val_query_exact_match_totals_by_task: dict[str, float] = {}
        self._val_query_exact_match_counts_by_task: dict[str, int] = {}
        self._val_examples: list[dict] = []

    def _build_backbone(
        self, backbone_model: dict, embedding_dim: int, seq_len: int
    ) -> tuple[nn.Module, int]:
        from models.mlp import MLP

        registry = {
            "rnn": RNN,
            "cnn": CNN,
            "transformer": Transformer,
            "mlp": MLP,
            "recursive_transformer": RecursiveTransformer,
            "canon_transformer": CanonTransformer,
            "canon_recursive_transformer": CanonRecursiveTransformer,
        }
        name = backbone_model["name"]
        if name not in registry:
            msg = f"Unknown backbone {name!r}. Choose from {sorted(registry)}."
            raise ValueError(msg)

        params = dict(backbone_model.get("params", {}))
        hidden_dim: int = params["hidden_dim"]
        params["input_dim"] = embedding_dim
        params["output_dim"] = hidden_dim
        if name == "mlp":
            params["seq_len"] = seq_len
        if name in ("transformer", "canon_transformer"):
            params["use_output_head"] = False
        # recursive_transformer / canon_recursive_transformer: no output_head; output_dim unused.

        return registry[name](**params), hidden_dim

    @property
    def is_binary_task(self) -> bool:
        return self.prediction_task == PREDICTION_TASK_BINARY

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard single forward pass — used for validation and inference."""
        value_ids = batch["input"].long()
        B, seq_len = value_ids.shape
        position_ids = torch.arange(seq_len, device=value_ids.device).unsqueeze(0).expand(B, -1)
        embedded = self.embedder(value_ids, position_ids)

        if self.padding_idx is not None:
            pad_mask = (value_ids == self.padding_idx).unsqueeze(-1)
            embedded = embedded.masked_fill(pad_mask, 0.0)

        if isinstance(self.backbone, (Transformer, RecursiveTransformer, CanonTransformer, CanonRecursiveTransformer)) and self.padding_idx is not None:
            padding_mask = value_ids == self.padding_idx
            hidden = self.backbone(embedded, src_key_padding_mask=padding_mask)
        elif isinstance(self.backbone, RNN) and self.padding_idx is not None:
            lengths = (value_ids != self.padding_idx).sum(dim=1)
            hidden = self.backbone(embedded, lengths=lengths)
        else:
            hidden = self.backbone(embedded)

        logits = self.head(hidden)
        if self.is_binary_task:
            logits = logits.squeeze(-1)
        return logits, batch["output"]

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.is_binary_task:
            return (logits > 0).long()
        return logits.argmax(dim=-1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.is_binary_task:
            targets_float = targets.float()
            losses = F.binary_cross_entropy_with_logits(logits, targets_float, reduction="none")
            weights = torch.ones_like(losses)
            weights = weights.masked_fill(targets_float != 0, self.non_background_loss_weight)
            return (losses * weights).mean()
        else:
            ignore = self.padding_idx if self.padding_idx is not None else -100
            targets_long = targets.long()
            losses = F.cross_entropy(
                logits.permute(0, 2, 1),
                targets_long,
                ignore_index=ignore,
                reduction="none",
            )
            valid_mask = targets_long != ignore
            weights = torch.ones_like(losses)
            weights = weights.masked_fill(
                valid_mask & (targets_long != 0),
                self.non_background_loss_weight,
            )
            return (losses * weights).sum() / valid_mask.sum().clamp_min(1)

    def _mask_padding(
        self, targets_long: torch.Tensor, predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.padding_idx is not None:
            is_pad = targets_long == self.padding_idx
            targets_long = targets_long.masked_fill(is_pad, 0)
            predictions = predictions.masked_fill(is_pad, 0)
        return targets_long, predictions

    def compute_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor, prefix: str
    ) -> dict[str, torch.Tensor]:
        predictions = self.decode_logits(logits)
        targets_long, predictions = self._mask_padding(targets.long(), predictions)
        key = f"{prefix}_" if prefix else ""
        return {
            f"{key}query_accuracy": accuracy(targets_long, predictions),
            f"{key}query_exact_match": exact_match_accuracy(targets_long, predictions),
        }

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        total_loss = 0.0
        logits, targets = None, None
        for _ in range(self.N_supervision):
            logits, targets = self(batch)
            loss = self.compute_loss(logits, targets)
            opt.zero_grad()
            self.manual_backward(loss)
            if self.gradient_clip_val is not None:
                clip_grad_norm_(self.parameters(), self.gradient_clip_val)
            opt.step()
            total_loss += loss.detach()

        if sch is not None:
            sch.step()

        avg_loss = total_loss / self.N_supervision
        B = batch["input"].shape[0]
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        log_kwargs = {"on_step": True, "on_epoch": True, "batch_size": B, "sync_dist": sync_dist}
        self.log("train_loss", avg_loss, prog_bar=True, **log_kwargs)
        metrics = self.compute_metrics(logits, targets, prefix="train")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=name.endswith("query_exact_match"), **log_kwargs)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        logits, targets = self(batch)
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets, prefix="val")
        B = batch["input"].shape[0]
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        log_kwargs = {"on_step": False, "on_epoch": True, "batch_size": B, "sync_dist": sync_dist}
        self.log("val_loss", loss, prog_bar=True, **log_kwargs)
        for name, value in metrics.items():
            self.log(name, value, prog_bar=name.endswith("query_exact_match"), **log_kwargs)
        self.log(
            "val_all_examples_exact_match",
            metrics["val_query_exact_match"],
            on_step=False, on_epoch=True, batch_size=B, sync_dist=sync_dist,
        )
        self.accumulate_query_exact_match_by_task_category(batch, logits, targets)
        if self.log_task_examples and len(self._val_examples) < 2:
            self._accumulate_val_examples(batch, logits, targets)

    def test_step(self, batch, batch_idx):
        logits, targets = self(batch)
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets, prefix="test")
        B = batch["input"].shape[0]
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        log_kwargs = {"on_step": False, "on_epoch": True, "batch_size": B, "sync_dist": sync_dist}
        self.log("test_loss", loss, **log_kwargs)
        for name, value in metrics.items():
            self.log(name, value, **log_kwargs)

    def _actual_seq_len(self, inp: torch.Tensor) -> int:
        if self.padding_idx is None:
            return inp.shape[0]
        values = inp.detach().cpu().tolist()
        n = len(values)
        while n > 0 and int(values[n - 1]) == self.padding_idx:
            n -= 1
        return n if n > 0 else len(values)

    def _accumulate_val_examples(
        self, batch: dict, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        predictions = self.decode_logits(logits)
        targets_long = targets.long()
        for inp, tgt, pred, cat in zip(
            batch["input"], targets_long, predictions, batch["task_category"], strict=True
        ):
            if len(self._val_examples) >= 2:
                break
            n = self._actual_seq_len(inp)
            inp_vals = [int(v) for v in inp[:n].detach().cpu().tolist()]
            tgt_vals = [int(v) for v in tgt[:n].detach().cpu().tolist()]
            pred_vals = [int(v) for v in pred[:n].detach().cpu().tolist()]
            self._val_examples.append(
                {
                    "task_category": cat,
                    "input": " ".join(str(v) for v in inp_vals),
                    "target": " ".join(str(v) for v in tgt_vals),
                    "prediction": " ".join(str(v) for v in pred_vals),
                    "correct": pred_vals == tgt_vals,
                }
            )

    def accumulate_query_exact_match_by_task_category(
        self, batch: dict, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        predictions = self.decode_logits(logits)
        targets_long, predictions = self._mask_padding(targets.long(), predictions)
        query_exact_matches = (predictions == targets_long).all(dim=1).float()
        for task_category, query_exact_match in zip(
            batch["task_category"], query_exact_matches.detach().cpu().tolist(), strict=True
        ):
            self._val_query_exact_match_totals_by_task[task_category] = (
                self._val_query_exact_match_totals_by_task.get(task_category, 0.0)
                + float(query_exact_match)
            )
            self._val_query_exact_match_counts_by_task[task_category] = (
                self._val_query_exact_match_counts_by_task.get(task_category, 0) + 1
            )

    def gather_query_exact_match_by_task_category(
        self,
    ) -> tuple[dict[str, float], dict[str, int]]:
        totals = dict(self._val_query_exact_match_totals_by_task)
        counts = dict(self._val_query_exact_match_counts_by_task)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return totals, counts
        gathered_payloads: list[dict | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            gathered_payloads, {"totals": totals, "counts": counts}
        )
        merged_totals: dict[str, float] = {}
        merged_counts: dict[str, int] = {}
        for payload in gathered_payloads:
            if payload is None:
                continue
            for task_category, total in payload["totals"].items():
                merged_totals[task_category] = merged_totals.get(task_category, 0.0) + float(total)
            for task_category, count in payload["counts"].items():
                merged_counts[task_category] = merged_counts.get(task_category, 0) + int(count)
        return merged_totals, merged_counts

    def on_validation_epoch_start(self) -> None:
        self._val_query_exact_match_totals_by_task = {}
        self._val_query_exact_match_counts_by_task = {}
        self._val_examples = []

    def on_validation_epoch_end(self) -> None:
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        category_totals, category_counts = self.gather_query_exact_match_by_task_category()
        for task_category, count in category_counts.items():
            if count < 1:
                continue
            metric_name = (
                f"val_query_exact_match_by_task_{_task_category_metric_suffix(task_category)}"
            )
            self.log(
                metric_name,
                category_totals[task_category] / count,
                on_step=False, on_epoch=True, prog_bar=False,
                batch_size=count, sync_dist=sync_dist,
            )
        epoch = self.trainer.current_epoch + 1
        if (
            self.log_task_examples
            and self._val_examples
            and epoch % self.log_task_examples_every_n_epochs == 0
        ):
            self._log_val_examples_to_wandb()

    def _log_val_examples_to_wandb(self) -> None:
        if self.trainer is not None and not self.trainer.is_global_zero:
            return
        experiment = getattr(getattr(self, "logger", None), "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        payload = {}
        for i, row in enumerate(self._val_examples):
            correct = row["correct"]
            title = f"{row['task_category']} | correct={correct}"
            fig = render_val_example_figure(
                input_sequence=[int(float(v)) for v in row["input"].split()],
                target_sequence=[int(float(v)) for v in row["target"].split()],
                prediction_sequence=[int(float(v)) for v in row["prediction"].split()],
                title=title,
            )
            payload[f"val_example_{i}"] = figure_to_wandb_image(fig, caption=title)
        experiment.log(payload)

    def configure_optimizers(self):
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_cls(
            (p for p in self.parameters() if p.requires_grad),
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
        print(repr(self.backbone))
        n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"N_supervision: {self.N_supervision}")
        print(f"Trainable params (total)   : {n_total:,}")
        print(f"Trainable params (backbone): {n_backbone:,}")

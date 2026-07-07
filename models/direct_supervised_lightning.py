"""Lightning training wrapper for direct supervised target-model experiments."""

import re

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import accuracy, exact_match_accuracy
from models.canon_transformer import CanonRecursiveTransformer, CanonTransformer
from models.looped_transformer import CanonLoopedTransformer, LoopedTransformer
from models.rope_looped_transformer import RoPECanonLoopedTransformer
from models.cnn import CNN
from models.recursive_transformer import RecursiveTransformer
from models.rnn import RNN
from models.task_token_embedder import TaskTokenEmbedder
from models.transformer import Transformer
from visualisation import figure_to_wandb_image, render_val_example_figure

PREDICTION_TASK_BINARY = "binary"
PREDICTION_TASK_MULTICLASS = "multiclass"


def _task_category_metric_suffix(task_category: str) -> str:
    """Convert a task category string into a safe metric name suffix."""
    return re.sub(r"[^a-zA-Z0-9]", "_", task_category)


class DirectSupervisedLightning(pl.LightningModule):
    """Direct supervised training of a target model on ARC-1D tasks.

    The model receives a single (input, output) pair per forward pass and
    learns to map input sequences to output sequences within a task category.
    Training pools support examples from all tasks in the category; validation
    uses query examples from held-out tasks in the same category.

    Architectures supported: RNN, CNN, Transformer, MLP (dispatched via
    backbone_model["name"]).
    """

    def __init__(
        self,
        backbone_model: dict,
        task_encoding: dict,
        prediction_task: str = PREDICTION_TASK_BINARY,
        num_classes: int | None = None,
        input_dim: int = 33,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_name: str = "Adam",
        weight_decay: float = 0.01,
        lr_scheduler: dict | None = None,
        warmup_steps: int = 0,
        non_background_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if (
            not isinstance(non_background_loss_weight, int | float)
            or non_background_loss_weight <= 0
        ):
            msg = "non_background_loss_weight must be a positive number."
            raise ValueError(msg)
        self.prediction_task = prediction_task
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name or optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler
        self.warmup_steps = warmup_steps
        self.non_background_loss_weight = float(non_background_loss_weight)
        self.log_task_examples = kwargs.get("log_task_examples", False)
        self.log_task_examples_every_n_epochs = kwargs.get("log_task_examples_every_n_epochs", 100)
        self.supports_hard_val_examples = False

        embedding_dim: int = task_encoding["embedding_dim"]
        value_vocab_size: int = task_encoding.get("value_vocab_size", 2)
        use_sinusoidal_pe: bool = task_encoding.get("use_sinusoidal_pe", True)
        self.padding_idx: int | None = kwargs.get("padding_idx")

        self.embedder = TaskTokenEmbedder(
            embedding_dim=embedding_dim,
            value_vocab_size=value_vocab_size,
            padding_idx=self.padding_idx,
            use_sinusoidal_pe=use_sinusoidal_pe,
        )

        self.backbone, hidden_dim = self._build_backbone(
            backbone_model, embedding_dim=embedding_dim, seq_len=input_dim
        )

        target_output_dim = 1 if prediction_task == PREDICTION_TASK_BINARY else num_classes
        self.head = nn.Linear(hidden_dim, target_output_dim, bias=False)

        # Per-category tracking state (reset each validation epoch)
        self._val_query_exact_match_totals_by_task: dict[str, float] = {}
        self._val_query_exact_match_counts_by_task: dict[str, int] = {}
        self._val_examples: list[dict] = []

    def _build_backbone(
        self, backbone_model: dict, embedding_dim: int, seq_len: int
    ) -> tuple[nn.Module, int]:
        """Instantiate the backbone and return (module, hidden_dim)."""
        from models.mlp import MLP  # explicit import to avoid collision with transformer.MLP

        registry = {
            "rnn": RNN,
            "cnn": CNN,
            "transformer": Transformer,
            "mlp": MLP,
            "recursive_transformer": RecursiveTransformer,
            "canon_transformer": CanonTransformer,
            "canon_recursive_transformer": CanonRecursiveTransformer,
            "looped_transformer": LoopedTransformer,
            "canon_looped_transformer": CanonLoopedTransformer,
            "rope_canon_looped_transformer": RoPECanonLoopedTransformer,
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
            # These backbones have an internal output_head that is redundant when DSL
            # adds its own head on top. Disable it so the backbone returns hidden states
            # directly, matching the contract of all other backbones.
            params["use_output_head"] = False
        # recursive_transformer / canon_recursive_transformer: no output_head; output_dim unused.

        return registry[name](**params), hidden_dim

    @property
    def is_binary_task(self) -> bool:
        return self.prediction_task == PREDICTION_TASK_BINARY

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: (batch, seq_len) for binary, (batch, seq_len, num_classes) for multiclass
            targets: (batch, seq_len) matching dtype of batch["output"]
        """
        value_ids = batch["input"].long()  # (B, seq_len)
        B, seq_len = value_ids.shape
        position_ids = torch.arange(seq_len, device=value_ids.device).unsqueeze(0).expand(B, -1)
        embedded = self.embedder(value_ids, position_ids)  # (B, seq_len, emb_dim)

        # Zero out padding positions so backbones see neutral vectors for PAD tokens.
        # value_embedding already zeros the value component (padding_idx), but sinusoidal
        # PE is still added, giving padding tokens a non-zero positional signal.
        if self.padding_idx is not None:
            pad_mask = (value_ids == self.padding_idx).unsqueeze(-1)  # (B, seq_len, 1)
            embedded = embedded.masked_fill(pad_mask, 0.0)

        if isinstance(self.backbone, (Transformer, CanonTransformer, CanonRecursiveTransformer, LoopedTransformer, CanonLoopedTransformer, RoPECanonLoopedTransformer)) and self.padding_idx is not None:
            padding_mask = value_ids == self.padding_idx  # (B, seq_len), True = pad
            hidden = self.backbone(embedded, src_key_padding_mask=padding_mask)
        elif isinstance(self.backbone, RNN) and self.padding_idx is not None:
            lengths = (value_ids != self.padding_idx).sum(dim=1)
            hidden = self.backbone(embedded, lengths=lengths)
        else:
            hidden = self.backbone(embedded)  # (B, seq_len, hidden_dim)
        logits = self.head(hidden)  # (B, seq_len, out_dim)
        if self.is_binary_task:
            logits = logits.squeeze(-1)  # (B, seq_len)
        return logits, batch["output"]

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to integer predictions."""
        if self.is_binary_task:
            return (logits > 0).long()
        return logits.argmax(dim=-1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.is_binary_task:
            targets_float = targets.float()
            losses = F.binary_cross_entropy_with_logits(
                logits,
                targets_float,
                reduction="none",
            )
            weights = torch.ones_like(losses)
            weights = weights.masked_fill(targets_float != 0, self.non_background_loss_weight)
            base_loss = (losses * weights).mean()
        else:
            # logits: (B, seq_len, num_classes) → CrossEntropyLoss expects (B, C, seq_len)
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
            weighted_losses = losses * weights
            base_loss = weighted_losses.sum() / valid_mask.sum().clamp_min(1)

        return base_loss

    def _mask_padding(
        self, targets_long: torch.Tensor, predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (targets, predictions) with padding positions zeroed out so
        metrics treat them as trivially correct (same value on both sides)."""
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

    def common_step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits, targets = self(batch)
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets, prefix=prefix)
        batch_size = batch["input"].shape[0]
        log_on_step = prefix == "train"
        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        log_kwargs = {
            "on_step": log_on_step,
            "on_epoch": True,
            "batch_size": batch_size,
            "sync_dist": sync_dist,
        }
        self.log(f"{prefix}_loss", loss, prog_bar=True, **log_kwargs)
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
        if prefix == "val":
            # StopOnMetricThreshold monitors this exact key
            self.log(
                "val_all_examples_exact_match",
                metrics["val_query_exact_match"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
            self.accumulate_query_exact_match_by_task_category(batch, logits, targets)
            if self.log_task_examples and len(self._val_examples) < 2:
                self._accumulate_val_examples(batch, logits, targets)
        return loss

    def _actual_seq_len(self, inp: torch.Tensor) -> int:
        """Return the unpadded length of a sequence tensor.

        Finds the last position that is NOT the padding sentinel so that
        trailing padding tokens are excluded from visualisation.
        """
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
        """Store up to 2 (input, target, prediction) rows for W&B logging.

        Sequences are sliced to their actual (unpadded) length so that padding
        tokens never appear in the rendered visualisation.
        """
        predictions = self.decode_logits(logits)
        targets_long = targets.long()
        for inp, tgt, pred, cat in zip(
            batch["input"],
            targets_long,
            predictions,
            batch["task_category"],
            strict=True,
        ):
            if len(self._val_examples) >= 2:
                break
            n = self._actual_seq_len(inp)
            inp_vals = [int(v) for v in inp[:n].detach().cpu().tolist()]
            tgt_vals = [int(v) for v in tgt[:n].detach().cpu().tolist()]
            pred_vals = [int(v) for v in pred[:n].detach().cpu().tolist()]
            exact = pred_vals == tgt_vals
            self._val_examples.append(
                {
                    "task_category": cat,
                    "input": " ".join(str(v) for v in inp_vals),
                    "target": " ".join(str(v) for v in tgt_vals),
                    "prediction": " ".join(str(v) for v in pred_vals),
                    "correct": exact,
                }
            )

    def accumulate_query_exact_match_by_task_category(
        self,
        batch: dict,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate validation query exact-match totals grouped by task category."""
        predictions = self.decode_logits(logits)
        targets_long, predictions = self._mask_padding(targets.long(), predictions)
        query_exact_matches = (predictions == targets_long).all(dim=1).float()
        for task_category, query_exact_match in zip(
            batch["task_category"],
            query_exact_matches.detach().cpu().tolist(),
            strict=True,
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
        """Gather validation task-category query exact-match totals across ranks."""
        totals = dict(self._val_query_exact_match_totals_by_task)
        counts = dict(self._val_query_exact_match_counts_by_task)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return totals, counts

        gathered_payloads: list[dict | None] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            gathered_payloads,
            {"totals": totals, "counts": counts},
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
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=count,
                sync_dist=sync_dist,
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

    def export_hard_examples(
        self,
        datamodule,
        output_path: str,
        wandb_logger=None,
        num_examples: int = 3,
        key_prefix: str = "val_hard_example",
        split: str = "val",
    ) -> None:
        """Find the hardest failures on split and log them to W&B and disk.

        A failure is any example where the predicted sequence does not exactly
        match the target (over valid, non-padding positions). Examples are ranked
        by position accuracy (lowest first). Sequences are stripped of trailing
        padding tokens before rendering.
        """
        import os

        from training.logging import _log_wandb_payload

        self.eval()
        device = next(self.parameters()).device
        wrong_examples: list[dict] = []

        dataloader = (
            datamodule.test_dataloader() if split == "test" else datamodule.val_dataloader()
        )
        with torch.no_grad():
            for batch in dataloader:
                batch_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                }
                logits, targets = self(batch_device)
                predictions = self.decode_logits(logits)
                targets_long = targets.long()

                for inp, tgt, pred, cat, tid in zip(
                    batch_device["input"],
                    targets_long,
                    predictions,
                    batch["task_category"],
                    batch["task_id"],
                    strict=True,
                ):
                    n = self._actual_seq_len(inp)
                    inp_vals = [int(v) for v in inp[:n].cpu().tolist()]
                    tgt_vals = [int(v) for v in tgt[:n].cpu().tolist()]
                    pred_vals = [int(v) for v in pred[:n].cpu().tolist()]

                    if pred_vals == tgt_vals:
                        continue  # exact match — not a hard example

                    num_correct = sum(p == t for p, t in zip(pred_vals, tgt_vals, strict=True))
                    pos_acc = num_correct / n if n > 0 else 0.0
                    wrong_examples.append(
                        {
                            "input": inp_vals,
                            "target": tgt_vals,
                            "prediction": pred_vals,
                            "position_accuracy": pos_acc,
                            "task_category": cat,
                            "task_id": int(tid) if isinstance(tid, torch.Tensor) else int(tid),
                        }
                    )

        if not wrong_examples:
            print(f"No wrong {split} examples found — skipping hard example logging.")
            return

        wrong_examples.sort(key=lambda ex: ex["position_accuracy"])
        hard = wrong_examples[:num_examples]

        hard_dir = os.path.join(output_path, f"hard_examples_{split}")
        os.makedirs(hard_dir, exist_ok=True)
        wandb_payload = {}

        for i, ex in enumerate(hard):
            n = len(ex["target"])
            num_wrong = sum(p != t for p, t in zip(ex["prediction"], ex["target"], strict=True))
            caption = (
                f"{ex['task_category']}:{ex['task_id']} | "
                f"pos_acc={ex['position_accuracy']:.2f} | "
                f"{num_wrong}/{n} wrong"
            )
            fig = render_val_example_figure(
                input_sequence=ex["input"],
                target_sequence=ex["target"],
                prediction_sequence=ex["prediction"],
                title=caption,
            )
            filename = (
                f"hard_{i}_{ex['task_category']}_{ex['task_id']}"
                f"_acc{ex['position_accuracy']:.2f}.png"
            )
            fig.savefig(os.path.join(hard_dir, filename), dpi=150, bbox_inches="tight")
            wandb_payload[f"{key_prefix}_{i}"] = figure_to_wandb_image(fig, caption=caption)

        _log_wandb_payload(wandb_logger, wandb_payload)
        print(
            f"Logged {len(hard)} hard {split} examples "
            f"({len(wrong_examples)} total failures) to {hard_dir}"
        )

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, prefix="val")

    def test_step(self, batch, batch_idx):
        self.common_step(batch, prefix="test")

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
        print(f"Trainable params (total)   : {n_total:,}")
        print(f"Trainable params (backbone): {n_backbone:,}")

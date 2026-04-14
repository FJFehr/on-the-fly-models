"""Lightning training wrapper for direct supervised target-model experiments."""

import re

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import accuracy, exact_match_accuracy
from models.cnn import CNN
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
        **kwargs,
    ):
        super().__init__()
        self.prediction_task = prediction_task
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name or optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler
        self.log_task_examples = kwargs.get("log_task_examples", False)
        self.supports_hard_val_examples = False

        embedding_dim: int = task_encoding["embedding_dim"]
        value_vocab_size: int = task_encoding.get("value_vocab_size", 2)

        self.embedder = TaskTokenEmbedder(
            embedding_dim=embedding_dim,
            position_vocab_size=input_dim,
            value_vocab_size=value_vocab_size,
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

        registry = {"rnn": RNN, "cnn": CNN, "transformer": Transformer, "mlp": MLP}
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
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        # logits: (B, seq_len, num_classes) → CrossEntropyLoss expects (B, C, seq_len)
        return F.cross_entropy(logits.permute(0, 2, 1), targets.long())

    def compute_metrics(
        self, logits: torch.Tensor, targets: torch.Tensor, prefix: str
    ) -> dict[str, torch.Tensor]:
        predictions = self.decode_logits(logits)
        targets_long = targets.long()
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

    def _seq_to_str(self, seq: torch.Tensor) -> str:
        return " ".join(str(v) for v in seq.detach().cpu().tolist())

    def _accumulate_val_examples(
        self, batch: dict, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Store up to 2 (input, target, prediction) rows for W&B logging."""
        predictions = self.decode_logits(logits)
        needed = 2 - len(self._val_examples)
        for inp, tgt, pred, cat in zip(
            batch["input"],
            targets,
            predictions,
            batch["task_category"],
            strict=True,
        ):
            if len(self._val_examples) >= 2:
                break
            exact = bool((pred == tgt.long()).all().item())
            self._val_examples.append(
                {
                    "task_category": cat,
                    "input": self._seq_to_str(inp),
                    "target": self._seq_to_str(tgt.long()),
                    "prediction": self._seq_to_str(pred),
                    "correct": exact,
                }
            )
            needed -= 1
            if needed == 0:
                break

    def accumulate_query_exact_match_by_task_category(
        self,
        batch: dict,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate validation query exact-match totals grouped by task category."""
        predictions = self.decode_logits(logits)
        query_exact_matches = (predictions == targets.long()).all(dim=1).float()
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
                sync_dist=False,
            )

        if self.log_task_examples and self._val_examples:
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
            sched_cls = getattr(torch.optim.lr_scheduler, self.lr_scheduler_cfg["name"])
            scheduler = sched_cls(optimizer, **self.lr_scheduler_cfg.get("params", {}))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return optimizer

    def on_fit_start(self) -> None:
        print(repr(self.backbone))
        n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Trainable params (total)   : {n_total:,}")
        print(f"Trainable params (backbone): {n_backbone:,}")

"""Training logic shared by the direct and hypernetwork Lightning modules.

Covers the optimiser (Muon, AdamW or RAdam), the warmup + cosine learning-rate schedule,
and per-task-category validation metrics.
"""

import re

import lightning as pl
import torch
import torch.nn as nn
from muon import SingleDeviceMuonWithAuxAdam

OPTIMIZERS = ("Muon", "AdamW", "RAdam")


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def sum_across_ranks(totals: dict[str, float]) -> dict[str, float]:
    """Sum a {key: number} dict over all distributed ranks (no-op on a single device)."""
    if not is_distributed():
        return dict(totals)
    gathered: list[dict | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, totals)
    summed: dict[str, float] = {}
    for rank_totals in gathered:
        for key, value in (rank_totals or {}).items():
            summed[key] = summed.get(key, 0.0) + float(value)
    return summed


def metric_suffix(task_category: str) -> str:
    """'1d_move_1p' -> '1d_move_1p'; any non-alphanumeric character becomes '_'."""
    return re.sub(r"[^a-zA-Z0-9]", "_", task_category)


class ArcLightningModule(pl.LightningModule):
    """Base class: optimiser, LR schedule, and per-category validation metrics.

    Subclasses set `muon_excluded_modules`: names of nn.Linear modules whose weights go to
    the AdamW group instead of Muon (input/output layers and embedding-like projections,
    following Keller Jordan's guidance for Muon).
    """

    muon_excluded_modules: tuple[str, ...] = ()

    def __init__(
        self,
        num_classes: int,
        padding_idx: int,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        lr_scheduler: dict | None = None,
        warmup_steps: int = 0,
        log_task_examples: bool = False,
        log_task_examples_every_n_epochs: int = 100,
        **_other_config,  # the whole run config is passed in; the rest is for data/trainer
    ):
        super().__init__()
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}.")
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.lr_scheduler_cfg = lr_scheduler
        self.warmup_steps = warmup_steps
        self.log_task_examples = log_task_examples
        self.log_task_examples_every_n_epochs = log_task_examples_every_n_epochs
        self._category_totals: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Optimiser and learning-rate schedule
    # ------------------------------------------------------------------

    def muon_param_groups(self) -> list[dict]:
        """Hidden nn.Linear weights go to Muon; everything else goes to AdamW."""
        muon_params = [
            module.weight
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
            and not set(name.split(".")) & set(self.muon_excluded_modules)
            and module.weight.requires_grad
        ]
        muon_ids = {id(p) for p in muon_params}
        adam_params = [p for p in self.parameters() if p.requires_grad and id(p) not in muon_ids]
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
            optimizer = SingleDeviceMuonWithAuxAdam(self.muon_param_groups())
        else:
            optimizer = getattr(torch.optim, self.optimizer_name)(
                (p for p in self.parameters() if p.requires_grad),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if not self.lr_scheduler_cfg:
            return optimizer
        scheduler = {
            "scheduler": self._build_scheduler(optimizer),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_scheduler(self, optimizer):
        """The configured scheduler, preceded by `warmup_steps` of linear warmup.

        With warmup, a cosine schedule's T_max is shortened by warmup_steps so the whole
        schedule still ends at T_max.
        """
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.lr_scheduler_cfg["name"])
        params = dict(self.lr_scheduler_cfg.get("params", {}))
        if self.warmup_steps <= 0:
            return scheduler_cls(optimizer, **params)
        if "T_max" in params:
            params["T_max"] = max(1, params["T_max"] - self.warmup_steps)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, scheduler_cls(optimizer, **params)],
            milestones=[self.warmup_steps],
        )

    # ------------------------------------------------------------------
    # Per-task-category validation metrics
    # ------------------------------------------------------------------

    def reset_category_metrics(self) -> None:
        self._category_totals = {}

    def add_category_values(
        self, metric: str, categories: list[str], values: torch.Tensor
    ) -> None:
        """Record one value per example for `metric`, grouped by task category."""
        totals = self._category_totals.setdefault(metric, {})
        for category, value in zip(categories, values.detach().cpu().tolist(), strict=True):
            totals[f"sum/{category}"] = totals.get(f"sum/{category}", 0.0) + float(value)
            totals[f"count/{category}"] = totals.get(f"count/{category}", 0.0) + 1.0

    def log_category_means(self, prefix: str) -> None:
        """Log `{prefix}_{metric}_by_task_{category}` = mean over that category's examples."""
        for metric, totals in self._category_totals.items():
            totals = sum_across_ranks(totals)
            for key, count in totals.items():
                if not key.startswith("count/") or count < 1:
                    continue
                category = key.removeprefix("count/")
                self.log(
                    f"{prefix}_{metric}_by_task_{metric_suffix(category)}",
                    totals[f"sum/{category}"] / count,
                    on_step=False,
                    on_epoch=True,
                    batch_size=int(count),
                    sync_dist=is_distributed(),
                )

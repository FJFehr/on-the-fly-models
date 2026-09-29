"""Hypernetwork training: generate a Transformer's weights from each task's support set.

Each batch item is one task: 3 support (input, output) pairs and 1 query pair. The
hypernetwork reads the support pairs, generates a target Transformer's weights, and that
target then predicts the outputs of all 4 inputs (support and query). The loss covers all
4; "query" metrics measure generalisation to the unseen query input.

Used in experiments 02 to 06. With `hyper_head.num_tasks` set, a one-hot task identity is
added to the task representation ("Task ID" arms); otherwise the support set is the only
signal ("w/o Task ID" arms).
"""

import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

from lightning_modules.common import ArcLightningModule, is_distributed, sum_across_ranks
from models.embedding import TASK_CATEGORY_INDEX, TokenEmbedder
from models.hypernetwork import Hypernetwork
from models.transformer import Transformer
from training.logging import log_wandb_payload
from visualisation import figure_to_wandb_image, render_task_prediction_figure

NUM_SUPPORT = 3
METRIC_TOTAL_KEYS = (
    "support_correct",
    "support_total",
    "support_exact",
    "support_examples",
    "query_correct",
    "query_total",
    "query_exact",
    "query_examples",
)


class HypernetworkLightning(ArcLightningModule):
    """Token embedder shared by both sides, plus a Hypernetwork writing a target Transformer."""

    # Input/output layers and the one-hot task projection stay on AdamW (see ArcLightningModule).
    muon_excluded_modules = ("input_projection", "output_head", "task_indicator_proj")

    # Hooks used by training/trainer.py and training/logging.py.
    supports_task_visualization = True
    supports_embedding_visualization = True

    def __init__(
        self,
        encoder: dict,
        target: dict,
        hyper_head: dict,
        task_encoding: dict,
        gradient_clip_val: float | None = None,
        log_embedding_clusters: bool = False,
        num_periodic_train_task_examples: int = 1,
        num_periodic_val_task_examples: int = 1,
        **config,
    ):
        super().__init__(**config)
        # One optimiser step per batch, with the gradient norm checked before each step.
        self.automatic_optimization = False
        self.gradient_clip_val = gradient_clip_val
        self.log_embedding_clusters = log_embedding_clusters
        self.num_periodic_train_task_examples = num_periodic_train_task_examples
        self.num_periodic_val_task_examples = num_periodic_val_task_examples
        self.selected_representative_task_ids: dict[str, list] = {"train": [], "val": []}
        self._metric_totals: dict[str, float] = {}

        embedding_dim = task_encoding["embedding_dim"]
        # Unlike the direct model, this embedder has no padding_idx and padded tokens are not
        # zeroed: the padding token has a learned embedding. The paper results used this setup.
        self.embedder = TokenEmbedder(
            dim=embedding_dim,
            vocab_size=self.num_classes + 1,  # the colours plus the padding token
            use_sinusoidal_pe=task_encoding.get("use_sinusoidal_pe", False),
        )
        self.hypernetwork = Hypernetwork(
            encoder=Transformer(input_dim=embedding_dim, **encoder),
            target=Transformer(input_dim=embedding_dim, output_dim=self.num_classes, **target),
            encoder_dim=encoder["output_dim"],
            bottleneck_dim=hyper_head["bottleneck_dim"],
            num_tasks=hyper_head.get("num_tasks"),
            freeze_task_indicator=hyper_head.get("freeze_task_indicator", False),
            task_indicator_placement=hyper_head.get("placement", "latent"),
        )
        # Categories scored with a zero task vector (no task identity), e.g. the category
        # held out of training in a leave-one-out run.
        self.zero_task_categories = set(hyper_head.get("zero_task_categories") or [])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def embed_context(
        self, support_inputs: torch.Tensor, support_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Serialise the 3 support pairs as in0, out0, in1, out1, in2, out2 and embed them.

        Each token also gets its support example index (0-2) and role (0 input, 1 output).
        Returns (batch, 6 * seq_len, embedding_dim).
        """
        batch, _, seq_len = support_inputs.shape
        segments = torch.stack(
            [s for i in range(NUM_SUPPORT) for s in (support_inputs[:, i], support_outputs[:, i])],
            dim=1,
        )
        num_segments = segments.shape[1]

        def per_segment(ids: list[int]) -> torch.Tensor:
            ids = torch.tensor(ids, device=segments.device).repeat_interleave(seq_len)
            return ids.unsqueeze(0).expand(batch, -1)

        positions = torch.arange(seq_len, device=segments.device).repeat(num_segments)
        return self.embedder(
            segments.reshape(batch, -1).long(),
            positions.unsqueeze(0).expand(batch, -1),
            example_ids=per_segment([0, 0, 1, 1, 2, 2]),
            role_ids=per_segment([0, 1, 0, 1, 0, 1]),
        )

    def prepare_inputs(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Turn a batch of tasks into the hypernetwork's and target's inputs.

        Returns:
            context: embedded support pairs, (batch, 6 * seq_len, embedding_dim)
            target_inputs: embedded inputs of the 3 support pairs then the query,
                (batch, 4, seq_len, embedding_dim)
            targets: the matching outputs, (batch, 4, seq_len)
            task_ids: task category indices when the model uses task identity, else None;
                a float task vector (zero rows for `zero_task_categories`) when any are set
        """
        context = self.embed_context(batch["support_inputs"], batch["support_outputs"])

        inputs = torch.cat(
            [batch["support_inputs"], batch["query_input"].unsqueeze(1)], dim=1
        ).long()
        positions = torch.arange(inputs.shape[-1], device=inputs.device).expand_as(inputs)
        target_inputs = self.embedder(inputs, positions)
        targets = torch.cat([batch["support_outputs"], batch["query_output"].unsqueeze(1)], dim=1)

        task_ids = None
        if self.hypernetwork.task_indicator_proj is not None:
            task_ids = torch.tensor(
                [TASK_CATEGORY_INDEX[c] for c in batch["task_category"]], device=self.device
            )
            if self.zero_task_categories:
                keep = torch.tensor(
                    [c not in self.zero_task_categories for c in batch["task_category"]],
                    device=self.device,
                )
                num_tasks = self.hypernetwork.num_tasks
                task_ids = F.one_hot(task_ids, num_tasks).float() * keep.unsqueeze(1)
        return context, target_inputs, targets, task_ids

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits (batch, 4, seq_len, num_classes) and targets (batch, 4, seq_len).

        Example index 0-2 are the support pairs and 3 is the query pair.
        """
        context, target_inputs, targets, task_ids = self.prepare_inputs(batch)
        return self.hypernetwork(context, target_inputs, task_ids), targets

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over the non-padding positions of all 4 examples."""
        return F.cross_entropy(
            logits.reshape(-1, self.num_classes),
            targets.long().reshape(-1),
            ignore_index=self.padding_idx,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metric_totals(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Correct/total counts for support and query examples, ignoring padding."""
        targets = targets.long()
        valid = targets != self.padding_idx
        matches = logits.argmax(dim=-1) == targets
        exact = (matches | ~valid).all(dim=2)
        totals = {}
        for name, part in (
            ("support", slice(None, NUM_SUPPORT)),
            ("query", slice(NUM_SUPPORT, None)),
        ):
            totals[f"{name}_correct"] = (matches[:, part] & valid[:, part]).sum().float()
            totals[f"{name}_total"] = valid[:, part].sum().float()
            totals[f"{name}_exact"] = exact[:, part].sum().float()
            totals[f"{name}_examples"] = torch.tensor(
                float(exact[:, part].numel()), device=targets.device
            )
        totals["all_examples_exact"] = exact.all(dim=1).sum().float()
        totals["task_count"] = torch.tensor(float(exact.shape[0]), device=targets.device)
        return totals

    @staticmethod
    def ratios(totals: dict, prefix: str) -> dict[str, tuple]:
        """{metric name: (numerator, denominator)} for the logged accuracy metrics."""
        pairs = {
            "support_accuracy": ("support_correct", "support_total"),
            "support_exact_match": ("support_exact", "support_examples"),
            "query_accuracy": ("query_correct", "query_total"),
            "query_exact_match": ("query_exact", "query_examples"),
        }
        return {
            f"{prefix}_{name}": (totals[num], totals[den]) for name, (num, den) in pairs.items()
        }

    def add_query_category_values(
        self, batch: dict, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Per-category query exact match and token accuracy.

        Here padded positions count as correct in the token accuracy (unlike the overall
        val_query_accuracy); the paper's per-category numbers were computed this way.
        """
        targets = targets.long()
        correct = (logits.argmax(dim=-1) == targets) | (targets == self.padding_idx)
        query_correct = correct[:, NUM_SUPPORT]
        categories = batch["task_category"]
        self.add_category_values("query_exact_match", categories, query_correct.all(dim=1).float())
        self.add_category_values("query_accuracy", categories, query_correct.float().mean(dim=1))

    # ------------------------------------------------------------------
    # Training, validation and test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        logits, targets = self(batch)
        loss = self.compute_loss(logits, targets)
        optimizer.zero_grad()
        self.manual_backward(loss)
        # With no clip value, max_norm=inf only measures the gradient norm.
        grad_norm = clip_grad_norm_(self.parameters(), self.gradient_clip_val or float("inf"))
        if not torch.isfinite(grad_norm):
            # A diverged run cannot recover, so stop rather than waste GPU time.
            raise RuntimeError(
                f"Non-finite gradient norm {grad_norm.item()} at step {self.global_step}."
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        totals = self.metric_totals(logits, targets)
        metrics = {
            name: num / den.clamp_min(1.0)
            for name, (num, den) in self.ratios(totals, "train").items()
        }
        metrics["train_all_examples_exact_match"] = (
            totals["all_examples_exact"] / totals["task_count"]
        )
        metrics["train_grad_norm"] = grad_norm
        log_kwargs = dict(
            on_step=True, on_epoch=True, batch_size=len(logits), sync_dist=is_distributed()
        )
        self.log("train_loss", loss.detach(), prog_bar=True, **log_kwargs)
        for name, value in metrics.items():
            self.log(name, value, prog_bar=name.endswith("query_exact_match"), **log_kwargs)
        return loss.detach()

    def eval_step(self, batch: dict, prefix: str) -> None:
        logits, targets = self(batch)
        self.log(
            f"{prefix}_loss",
            self.compute_loss(logits, targets),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(logits),
            sync_dist=is_distributed(),
        )
        for key, value in self.metric_totals(logits, targets).items():
            if key in METRIC_TOTAL_KEYS:
                self._metric_totals[key] = self._metric_totals.get(key, 0.0) + float(value)
        if prefix == "val":
            self.add_query_category_values(batch, logits, targets)

    def log_epoch_metrics(self, prefix: str) -> None:
        """Log accuracies over the whole epoch (total correct / total count across batches)."""
        totals = sum_across_ranks(self._metric_totals)
        for name, (numerator, denominator) in self.ratios(totals, prefix).items():
            if denominator > 0:
                self.log(
                    name,
                    numerator / denominator,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=name.endswith("query_exact_match"),
                    batch_size=max(1, round(denominator)),
                )

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._metric_totals = {}
        self.reset_category_metrics()

    def on_validation_epoch_end(self) -> None:
        self.log_epoch_metrics("val")
        self.log_category_means("val")

    def on_test_epoch_start(self) -> None:
        self._metric_totals = {}

    def on_test_epoch_end(self) -> None:
        self.log_epoch_metrics("test")

    # ------------------------------------------------------------------
    # Task records for figures, galleries and embedding plots
    # ------------------------------------------------------------------

    def predict_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, predicted colours and targets for a batch of tasks."""
        logits, targets = self(batch)
        return logits, logits.argmax(dim=-1), targets.long()

    def unpadded(self, sequence: list[int], length_from: list[int]) -> list[int]:
        """`sequence` cut to the unpadded length of `length_from`."""
        length = len(length_from)
        while length > 0 and length_from[length - 1] == self.padding_idx:
            length -= 1
        return [int(v) for v in sequence[: length or len(length_from)]]

    def build_task_records(
        self, batch: dict, predictions: torch.Tensor, targets: torch.Tensor
    ) -> list[dict]:
        """One dict per task with its unpadded support/query inputs, targets and predictions."""
        support_inputs = batch["support_inputs"].cpu().long().tolist()
        query_inputs = batch["query_input"].cpu().long().tolist()
        predictions = predictions.cpu().tolist()
        targets = targets.cpu().tolist()
        task_ids = batch["task_id"]
        task_ids = task_ids.tolist() if isinstance(task_ids, torch.Tensor) else list(task_ids)

        records = []
        for i, category in enumerate(batch["task_category"]):
            support_in = support_inputs[i]
            query_in = query_inputs[i]
            query_target = self.unpadded(targets[i][NUM_SUPPORT], query_in)
            query_prediction = self.unpadded(predictions[i][NUM_SUPPORT], query_in)
            num_correct = sum(p == t for p, t in zip(query_prediction, query_target, strict=True))
            records.append(
                {
                    "support_inputs": [self.unpadded(x, x) for x in support_in],
                    "support_outputs": [
                        self.unpadded(targets[i][j], support_in[j]) for j in range(NUM_SUPPORT)
                    ],
                    "support_predictions": [
                        self.unpadded(predictions[i][j], support_in[j]) for j in range(NUM_SUPPORT)
                    ],
                    "query_input": self.unpadded(query_in, query_in),
                    "query_output": query_target,
                    "query_prediction": query_prediction,
                    "task_category": category,
                    "task_id": task_ids[i],
                    "query_exact_match": query_prediction == query_target,
                    "query_accuracy": num_correct / len(query_target) if query_target else 0.0,
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

    def _batches_on_device(self, dataloader):
        for batch in dataloader:
            yield {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    @torch.no_grad()
    def collect_task_records_from_dataloader(
        self, dataloader, limit: int | None = None
    ) -> list[dict]:
        self.eval()
        records = []
        for batch in self._batches_on_device(dataloader):
            _, predictions, targets = self.predict_batch(batch)
            records.extend(self.build_task_records(batch, predictions, targets))
            if limit is not None and len(records) >= limit:
                return records[:limit]
        return records

    @torch.no_grad()
    def collect_embedding_records_from_dataloader(
        self, dataloader, limit: int | None = None
    ) -> list[dict]:
        """The task representation each task's weights were generated from, for cluster plots."""
        self.eval()
        records = []
        for batch in self._batches_on_device(dataloader):
            self(batch)
            representations = self.hypernetwork.last_task_representation.cpu()
            task_ids = batch["task_id"]
            task_ids = task_ids.tolist() if isinstance(task_ids, torch.Tensor) else list(task_ids)
            for category, task_id, representation in zip(
                batch["task_category"], task_ids, representations, strict=True
            ):
                records.append(
                    {
                        "task_category": category,
                        "task_id": task_id,
                        "pooled_embedding": representation,
                    }
                )
                if limit is not None and len(records) >= limit:
                    return records
        return records

    def select_representative_task_records_from_dataset(
        self, dataset, split_name: str, limit: int
    ) -> list:
        """The first task of each category (up to `limit`), reused for every gallery snapshot."""
        selected, seen = [], set()
        for task in dataset.tasks:
            if len(selected) >= limit:
                break
            if task["task_category"] not in seen:
                seen.add(task["task_category"])
                selected.append((task["task_category"], task["task_id"]))
        self.selected_representative_task_ids[split_name] = selected
        return selected

    @torch.no_grad()
    def collect_task_records_from_dataset_by_task_ids(self, dataset, task_ids: list) -> list[dict]:
        """Prediction records for the given (category, task_id) pairs, in that order."""
        order = {key: i for i, key in enumerate(task_ids)}
        records = []
        for task in dataset.tasks:
            if (task["task_category"], task["task_id"]) not in order:
                continue
            batch = {
                key: torch.tensor([task[key]], device=self.device)
                for key in (
                    "support_inputs",
                    "support_outputs",
                    "query_input",
                    "query_output",
                    "task_id",
                )
            }
            batch["task_category"] = [task["task_category"]]
            _, predictions, targets = self.predict_batch(batch)
            records.extend(self.build_task_records(batch, predictions, targets))
        return sorted(records, key=lambda r: order[(r["task_category"], r["task_id"])])

    def log_task_gallery(
        self,
        split_name: str,
        records: list[dict],
        output_path: str,
        wandb_logger=None,
        key_prefix: str | None = None,
    ) -> None:
        """Save a prediction figure per task record, and log them to W&B."""
        if not records or (self._trainer is not None and not self.trainer.is_global_zero):
            return
        prefix = key_prefix or split_name
        split_dir = os.path.join(output_path, f"{split_name}_task_examples")
        os.makedirs(split_dir, exist_ok=True)
        payload = {}
        for i, record in enumerate(records):
            figure = self.build_task_visualization_figure(record)
            name = f"{record['task_category']}_{record['task_id']}"
            figure.savefig(
                os.path.join(
                    split_dir, f"{prefix}_{i}_{name}_query{int(record['query_exact_match'])}.png"
                ),
                dpi=150,
                bbox_inches="tight",
            )
            caption = (
                f"{prefix} | {record['task_category']}:{record['task_id']} | "
                f"query_exact_match={record['query_exact_match']} | "
                f"query_acc={record['query_accuracy']:.2f}"
            )
            payload[f"{prefix}_{name}"] = figure_to_wandb_image(figure, caption=caption)
            plt.close(figure)
        log_wandb_payload(wandb_logger, payload)

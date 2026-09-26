"""Direct training: one Transformer learns input -> output for ARC-1D examples.

Used for the "individual" (one model per task category) and "joint" (one model for all
categories) conditions in experiments 01 and 06. In the joint condition, the optional task
embedding gives the model the task category's identity ("Task ID").
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_modules.common import ArcLightningModule, is_distributed
from models.embedding import TASK_CATEGORY_INDEX, TokenEmbedder
from models.transformer import Transformer
from training.logging import log_wandb_payload
from visualisation import figure_to_wandb_image, render_val_example_figure

NUM_LOGGED_VAL_EXAMPLES = 2


class DirectLightning(ArcLightningModule):
    """Embed tokens -> Transformer backbone -> linear head over the colour classes."""

    # The final output layer stays on AdamW (see ArcLightningModule).
    muon_excluded_modules = ("head",)

    def __init__(self, backbone: dict, task_encoding: dict, **config):
        super().__init__(**config)
        embedding_dim = task_encoding["embedding_dim"]
        self.embedder = TokenEmbedder(
            dim=embedding_dim,
            vocab_size=self.num_classes + 1,  # the colours plus the padding token
            padding_idx=self.padding_idx,
            use_sinusoidal_pe=task_encoding.get("use_sinusoidal_pe", False),
        )
        self.task_embedding = (
            nn.Embedding(len(TASK_CATEGORY_INDEX), embedding_dim)
            if task_encoding.get("use_task_embedding", False)
            else None
        )
        self.backbone = Transformer(input_dim=embedding_dim, **backbone)
        self.head = nn.Linear(backbone["hidden_dim"], self.num_classes, bias=False)
        self._val_examples: list[dict] = []

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits (batch, seq_len, num_classes), targets (batch, seq_len))."""
        value_ids = batch["input"].long()
        position_ids = torch.arange(value_ids.shape[1], device=self.device).expand_as(value_ids)
        embedded = self.embedder(value_ids, position_ids)
        if self.task_embedding is not None:
            task_ids = torch.tensor(
                [TASK_CATEGORY_INDEX[c] for c in batch["task_category"]], device=self.device
            )
            embedded = embedded + self.task_embedding(task_ids).unsqueeze(1)
        # Padded positions enter the backbone as zero vectors.
        embedded = embedded.masked_fill((value_ids == self.padding_idx).unsqueeze(-1), 0.0)
        return self.head(self.backbone(embedded)), batch["output"]

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy averaged over non-padding positions."""
        return F.cross_entropy(
            logits.permute(0, 2, 1), targets.long(), ignore_index=self.padding_idx
        )

    def predict(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted colours and targets, with padded positions set to 0 in both.

        Padded positions therefore count as correct. That inflates token accuracy slightly
        (not exact match), and is how every direct-training result in the paper was computed.
        """
        predictions = logits.argmax(dim=-1)
        targets = targets.long()
        is_pad = targets == self.padding_idx
        return predictions.masked_fill(is_pad, 0), targets.masked_fill(is_pad, 0)

    def common_step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits, targets = self(batch)
        loss = self.compute_loss(logits, targets)
        predictions, targets = self.predict(logits, targets)
        exact_match = (predictions == targets).all(dim=1).float()
        metrics = {
            f"{prefix}_loss": loss,
            f"{prefix}_query_accuracy": (predictions == targets).float().mean(),
            f"{prefix}_query_exact_match": exact_match.mean(),
        }
        if prefix == "val":
            # StopOnMetricThreshold monitors this key.
            metrics["val_all_examples_exact_match"] = metrics["val_query_exact_match"]
            self.add_category_values("query_exact_match", batch["task_category"], exact_match)
            self._keep_val_examples(batch, predictions, targets)

        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=name.endswith(("_loss", "query_exact_match")),
                batch_size=batch["input"].shape[0],
                sync_dist=is_distributed(),
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.common_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self.reset_category_metrics()
        self._val_examples = []

    def on_validation_epoch_end(self) -> None:
        self.log_category_means("val")
        epoch = self.trainer.current_epoch + 1
        if self.log_task_examples and epoch % self.log_task_examples_every_n_epochs == 0:
            self._log_val_examples_to_wandb()

    # ------------------------------------------------------------------
    # Example figures
    # ------------------------------------------------------------------

    def unpadded(self, sequence: torch.Tensor, length_from: torch.Tensor) -> list[int]:
        """`sequence` cut to the unpadded length of `length_from`, as a list of ints."""
        values = length_from.tolist()
        length = len(values)
        while length > 0 and values[length - 1] == self.padding_idx:
            length -= 1
        return [int(v) for v in sequence[: length or len(values)].tolist()]

    def _keep_val_examples(
        self, batch: dict, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Keep the first few validation examples of the epoch for W&B figures."""
        rows = zip(batch["input"], targets, predictions, batch["task_category"], strict=True)
        for inputs, target, prediction, category in rows:
            if not self.log_task_examples or len(self._val_examples) >= NUM_LOGGED_VAL_EXAMPLES:
                return
            inputs = inputs.cpu()
            self._val_examples.append(
                {
                    "task_category": category,
                    "input": self.unpadded(inputs, inputs),
                    "target": self.unpadded(target.cpu(), inputs),
                    "prediction": self.unpadded(prediction.cpu(), inputs),
                }
            )

    def _log_val_examples_to_wandb(self) -> None:
        if not self._val_examples or not self.trainer.is_global_zero:
            return
        payload = {}
        for i, example in enumerate(self._val_examples):
            correct = example["prediction"] == example["target"]
            title = f"{example['task_category']} | correct={correct}"
            figure = render_val_example_figure(
                input_sequence=example["input"],
                target_sequence=example["target"],
                prediction_sequence=example["prediction"],
                title=title,
            )
            payload[f"val_example_{i}"] = figure_to_wandb_image(figure, caption=title)
        log_wandb_payload(self.logger, payload)

    @torch.no_grad()
    def export_hard_examples(
        self,
        datamodule,
        output_path: str,
        wandb_logger=None,
        num_examples: int = 3,
        key_prefix: str = "val_hard_example",
        split: str = "val",
    ) -> None:
        """Save figures of the worst-predicted examples (lowest token accuracy) in `split`."""
        self.eval()
        dataloader = (
            datamodule.test_dataloader() if split == "test" else datamodule.val_dataloader()
        )
        failures = []
        for batch in dataloader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits, targets = self(batch)
            predictions = logits.argmax(dim=-1).cpu()
            targets = targets.long().cpu()
            inputs = batch["input"].cpu()
            for i in range(len(inputs)):
                example = {
                    "input": self.unpadded(inputs[i], inputs[i]),
                    "target": self.unpadded(targets[i], inputs[i]),
                    "prediction": self.unpadded(predictions[i], inputs[i]),
                    "task_category": batch["task_category"][i],
                    "task_id": int(batch["task_id"][i]),
                }
                if example["prediction"] == example["target"]:
                    continue
                correct = sum(
                    p == t for p, t in zip(example["prediction"], example["target"], strict=True)
                )
                example["accuracy"] = correct / len(example["target"])
                failures.append(example)

        if not failures:
            print(f"No wrong {split} examples found, skipping hard example export.")
            return

        failures.sort(key=lambda example: example["accuracy"])
        hard_dir = os.path.join(output_path, f"hard_examples_{split}")
        os.makedirs(hard_dir, exist_ok=True)
        payload = {}
        for i, example in enumerate(failures[:num_examples]):
            num_wrong = sum(
                p != t for p, t in zip(example["prediction"], example["target"], strict=True)
            )
            caption = (
                f"{example['task_category']}:{example['task_id']} | "
                f"pos_acc={example['accuracy']:.2f} | {num_wrong}/{len(example['target'])} wrong"
            )
            figure = render_val_example_figure(
                input_sequence=example["input"],
                target_sequence=example["target"],
                prediction_sequence=example["prediction"],
                title=caption,
            )
            name = f"{example['task_category']}_{example['task_id']}"
            filename = f"hard_{i}_{name}_acc{example['accuracy']:.2f}.png"
            figure.savefig(os.path.join(hard_dir, filename), dpi=150, bbox_inches="tight")
            payload[f"{key_prefix}_{i}"] = figure_to_wandb_image(figure, caption=caption)

        log_wandb_payload(wandb_logger, payload)
        num_saved = min(num_examples, len(failures))
        print(f"Saved {num_saved} hard {split} examples ({len(failures)} failures) to {hard_dir}")

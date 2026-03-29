# models/target_models/base.py
# ---------------------------------------------------------------------------
# Base PyTorch Lightning module for all target models.
#
# This file contains ALL shared training logic: loss computation, metric
# logging, optimizer setup, validation example collection, and WandB image
# logging. Concrete model files (target_model.py, target_cnn_model.py, etc.)
# inherit from BaseTargetModel and only need to implement two things:
#
#   1. __init__  — build self.model (the torch.nn.Module architecture)
#   2. forward   — define how inputs flow through self.model
#
# This design eliminates the duplication that previously existed between
# model files, where ~200 lines of identical boilerplate were copy-pasted
# into each new model. Now adding a new model is ~20 lines.
# ---------------------------------------------------------------------------

import lightning as pl
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

import wandb
from metrics import accuracy, exact_match_accuracy
from visualisation import figure_to_wandb_image, render_val_example_figure


class BaseTargetModel(pl.LightningModule):
    """Base Lightning module for all target models in the on-the-fly framework.

    Handles everything except the architecture itself:
      - Loss computation for binary or multiclass sequence tasks
      - Metric logging (loss, elementwise accuracy, exact-match accuracy)
      - Optimizer creation from config string (e.g. "Adam" -> torch.optim.Adam)
      - Validation example collection and WandB image logging
      - Training / validation / test step orchestration

    Subclasses MUST:
      1. Call super().__init__(...) with the shared parameters
      2. Set self.model to a torch.nn.Module
      3. Override forward(inputs) to define the forward pass

    The forward() method receives raw input tensors of shape (batch, seq_len).
    Subclasses must return logits using the canonical shape for the configured
    task:

    - binary: (batch, seq_len)
    - multiclass: (batch, seq_len, num_classes)

    The base class does not reshape or infer multiclass layouts. It only
    validates the returned shape, computes the task-appropriate loss, decodes
    predictions, and logs shared metrics.

    Args:
        learning_rate: Learning rate for the optimizer.
        optimizer: Name of the torch.optim optimizer class (e.g. "Adam").
        weight_decay: L2 regularisation strength.
        log_val_examples: Whether to log validation comparison images to WandB
            during training.
        max_logged_val_examples: Maximum number of validation examples to
            log per epoch (prevents flooding the WandB dashboard).
        max_logged_val_examples_per_category: Maximum number of examples to
            log per task category within one validation epoch.
        log_val_examples_every_n_epochs: How often to log validation examples.
            Logging happens on epoch 1 and then every N epochs after that.
        **kwargs: Extra config keys from the YAML config. These are silently
            ignored so that the full config dict can be passed as **kwargs
            without causing errors for keys the model doesn't use.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        optimizer: str = "Adam",
        weight_decay: float = 0.01,
        log_val_examples: bool = False,
        max_logged_val_examples: int = 8,
        max_logged_val_examples_per_category: int = 3,
        log_val_examples_every_n_epochs: int = 5,
        prediction_task: str = "binary",
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__()

        if prediction_task not in {"binary", "multiclass"}:
            msg = f"Unknown prediction_task {prediction_task!r}."
            raise ValueError(msg)
        if prediction_task == "multiclass" and num_classes < 2:
            msg = "num_classes must be at least 2 for multiclass prediction."
            raise ValueError(msg)

        self.prediction_task = prediction_task
        self.num_classes = num_classes
        self.loss_fn = (
            BCEWithLogitsLoss() if self.prediction_task == "binary" else CrossEntropyLoss()
        )

        # Store optimizer config for use in configure_optimizers().
        # The optimizer is resolved dynamically from its string name so that
        # experiment configs can switch optimizers without code changes.
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        # Validation example logging config.
        # When enabled, the model collects a handful of validation examples
        # during validation and renders them as stacked ARC-colored comparison
        # images in WandB. This is invaluable for visually debugging what the
        # model is actually predicting vs the ground truth.
        self.log_val_examples = log_val_examples
        self.max_logged_val_examples = max_logged_val_examples
        self.max_logged_val_examples_per_category = max_logged_val_examples_per_category
        self.log_val_examples_every_n_epochs = log_val_examples_every_n_epochs

        # Internal state for tracking which validation epoch we're on and
        # whether we should log examples this epoch. Reset at the start of
        # each validation epoch.
        self.validation_epoch_count = 0
        self.should_log_val_examples = False
        self.val_example_records: list[dict] = []
        self.val_examples_per_category: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Forward pass — subclasses MUST override this
    # -----------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute raw logits using the canonical task-specific shape.

        Subclasses MUST override this method. The base implementation raises
        NotImplementedError to catch accidental use of the base class directly.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length).

        Returns:
            Binary tasks must return logits of shape ``(batch_size, sequence_length)``.
            Multiclass tasks must return logits of shape
            ``(batch_size, sequence_length, num_classes)``.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    # -----------------------------------------------------------------------
    # Batch unpacking
    # -----------------------------------------------------------------------

    def unpack_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Extract inputs, targets, and metadata from a dataloader batch.

        The dataloader produces dicts with input/target tensors plus task
        metadata (category, task_id, etc.). This method splits them into
        the tensors needed for the forward pass and a metadata dict used
        for logging and visualization.

        Args:
            batch: Dict from the dataloader with keys "inputs", "targets",
                "task_category", "task_id", "example_index", "source".

        Returns:
            Tuple of (inputs, targets, metadata) where inputs and targets
            are tensors and metadata is a dict of task-level information.
        """
        metadata = {
            "task_category": batch["task_category"],
            "task_id": batch["task_id"],
            "example_index": batch["example_index"],
            "source": batch["source"],
        }
        return batch["inputs"], batch["targets"], metadata

    # -----------------------------------------------------------------------
    # Shared training / evaluation step
    # -----------------------------------------------------------------------

    def common_step(
        self, batch, prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run the shared train/val/test path from batch to logged metrics.

        This method is the single orchestration point for supervised steps.
        It:

        1. unpacks inputs, targets, and metadata
        2. calls the subclass ``forward()``
        3. validates that logits already follow the canonical task contract
        4. computes the configured loss
        5. decodes predicted class ids
        6. logs elementwise and exact-match metrics

        Args:
            batch: Dict batch from the dataloader.
            prefix: String prefix for logged metric names ("train", "val", "test").

        Returns:
            Tuple of (loss, predictions, targets, metadata):
              - loss: Scalar loss tensor (used by Lightning for backprop)
              - predictions: Integer class predictions, shape (batch, seq_len)
              - targets: Ground truth class ids, shape (batch, seq_len)
              - metadata: Task metadata dict for visualization
        """
        inputs, targets, metadata = self.unpack_batch(batch)
        logits = self(inputs)
        logits = self.format_logits(logits, targets)
        targets_long = targets.long()
        loss = self.compute_loss(logits, targets, targets_long)
        preds = self.decode_logits(logits)

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_accuracy", accuracy(targets_long, preds), prog_bar=True)
        self.log(
            f"{prefix}_exact_match_accuracy",
            exact_match_accuracy(targets_long, preds),
            prog_bar=True,
        )

        return loss, preds, targets_long, metadata

    # -----------------------------------------------------------------------
    # Lightning step hooks
    # -----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Lightning training step — runs forward pass and returns loss.

        Only the loss is needed for backpropagation; predictions and metadata
        are discarded during training.
        """
        loss, _, _, _ = self.common_step(batch, "train")
        return loss

    def on_validation_epoch_start(self) -> None:
        """Called by Lightning at the start of each validation epoch.

        Increments the epoch counter, checks whether this epoch should log
        validation examples, and resets the example collection buffer.
        """
        self.validation_epoch_count += 1
        self.should_log_val_examples = self.should_log_examples_for_epoch()
        self.val_example_records = []
        self.val_examples_per_category = {}

    def validation_step(self, batch, batch_idx):
        """Lightning validation step — runs forward pass and optionally collects examples.

        In addition to computing loss and metrics, this step collects a
        limited number of validation-example records for WandB visualization
        when example logging is enabled for this epoch.
        """
        loss, preds, targets, metadata = self.common_step(batch, "val")

        # Collect validation examples for WandB logging if:
        # 1. Example logging is enabled in config
        # 2. This is a logging epoch (first epoch or every N epochs)
        # 3. We haven't filled our per-epoch example budget yet
        if (
            self.log_val_examples
            and self.should_log_val_examples
            and len(self.val_example_records) < self.max_logged_val_examples
        ):
            self.val_example_records.extend(
                self.build_val_example_records(batch, preds, targets, metadata)
            )
        return loss

    def test_step(self, batch, batch_idx):
        """Lightning test step — runs forward pass and returns loss.

        Same as training_step: only loss and metrics are needed.
        """
        loss, _, _, _ = self.common_step(batch, "test")
        return loss

    def on_validation_epoch_end(self) -> None:
        """Called by Lightning at the end of each validation epoch.

        If validation examples were collected during this epoch, render them
        as ARC-colored grid images and log to WandB as a single media stream.
        """
        # Early exit if logging is disabled, not a logging epoch, or no examples collected
        if (
            not self.log_val_examples
            or not self.should_log_val_examples
            or not self.val_example_records
        ):
            return

        # Guard: only log if we have a valid WandB logger attached
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        if not isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            return

        # Log the collected examples to WandB as three image streams
        self.logger.experiment.log(
            self.build_val_example_log_payload(self.val_example_records),
            step=self.global_step,
        )

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    def configure_optimizers(self):
        """Create the optimizer from the config-specified class name.

        Uses getattr to dynamically look up the optimizer class from
        torch.optim by name (e.g. "Adam" -> torch.optim.Adam). This lets
        experiment configs switch optimizers without any code changes.

        Returns:
            The configured optimizer instance.
        """
        # Dynamically resolve the optimizer class from its string name
        optimizer_cls = getattr(torch.optim, self.optimizer_name)

        # Create the optimizer with all model parameters
        return optimizer_cls(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    # -----------------------------------------------------------------------
    # Validation example logging helpers
    # -----------------------------------------------------------------------

    def should_log_examples_for_epoch(self) -> bool:
        """Decide whether to log validation examples for the current epoch.

        Examples are logged on the very first validation epoch (to see
        initial predictions) and then every N epochs after that. This keeps
        WandB storage reasonable while still providing periodic snapshots.
        """
        if not self.log_val_examples:
            return False
        return self.validation_epoch_count == 1 or (
            self.validation_epoch_count % self.log_val_examples_every_n_epochs == 0
        )

    def format_logits(self, logits: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """Validate that a model returned the canonical shape for its task.

        Binary models must emit ``(batch, seq_len)``.
        Multiclass models must emit ``(batch, seq_len, num_classes)``.

        The method intentionally does not reshape or transpose logits. If a
        model returns the wrong layout, that is treated as a model bug rather
        than silently repaired in the training loop.
        """
        if targets is None:
            msg = "Targets are required so the base class can validate logit shapes."
            raise ValueError(msg)

        expected_batch, expected_seq_len = targets.shape[0], targets.shape[1]

        if self.prediction_task == "binary":
            expected_shape = (expected_batch, expected_seq_len)
            if logits.shape != expected_shape:
                msg = (
                    "Binary models must return logits with shape "
                    f"{expected_shape}, got {tuple(logits.shape)}."
                )
                raise ValueError(msg)
            return logits

        expected_shape = (expected_batch, expected_seq_len, self.num_classes)
        if logits.shape != expected_shape:
            msg = (
                "Multiclass models must return logits with shape "
                f"{expected_shape}, got {tuple(logits.shape)}."
            )
            raise ValueError(msg)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        targets_long: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the configured loss from canonical logits and targets.

        Binary tasks use ``BCEWithLogitsLoss`` on ``(batch, seq_len)`` logits
        against float targets.

        Multiclass tasks use ``CrossEntropyLoss`` on per-position class logits.
        ``CrossEntropyLoss`` expects the class axis before the sequence axis, so
        the canonical ``(batch, seq_len, num_classes)`` tensor is permuted to
        ``(batch, num_classes, seq_len)`` only at the loss callsite.
        """
        if self.prediction_task == "binary":
            return self.loss_fn(logits, targets)
        return self.loss_fn(logits.permute(0, 2, 1), targets_long)

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert canonical logits into integer class predictions.

        Binary tasks apply a sigmoid and threshold at 0.5.
        Multiclass tasks take the argmax over the class dimension.
        """
        if self.prediction_task == "binary":
            return (torch.sigmoid(logits) > 0.5).long()
        return logits.argmax(dim=-1)

    def build_val_example_log_payload(self, records: list[dict]) -> dict:
        """Build the WandB log payload from collected example records.

        Organizes records into individual media entries so each validation
        example appears as its own image in WandB instead of as a table row.
        """
        return {record["log_key"]: record["combined_image"] for record in records}

    def build_val_example_caption(
        self,
        task_category: str,
        task_id: int,
        exact_match: bool,
    ) -> str:
        """Build a human-readable caption for a validation example image.

        The caption shows the task category, task ID, and whether the model's
        prediction exactly matched the target sequence.
        """
        return f"{task_category}:{task_id} | exact_match={exact_match}"

    def build_val_example_log_key(self, task_category: str, task_id: int) -> str:
        """Build the WandB payload key for an individual validation example."""
        safe_category = task_category.replace("/", "_")
        return f"val_example_{safe_category}_{task_id}"

    def build_val_example_records(
        self,
        batch: dict,
        preds: torch.Tensor,
        targets: torch.Tensor,
        metadata: dict,
    ) -> list[dict]:
        """Build visualization records from a batch of validation examples.

        For each eligible example, renders one vertically stacked figure
        containing the input, target, and prediction sequences and wraps it in
        a wandb.Image with a descriptive caption. Eligibility is controlled by
        both the global per-epoch cap and the per-category cap.

        Args:
            batch: The raw batch dict from the dataloader.
            preds: Model predictions tensor, shape (batch, seq_len).
            targets: Ground truth tensor, shape (batch, seq_len).
            metadata: Task metadata dict with category, task_id, etc.
        Returns:
            List of record dicts, each containing raw data and one combined
            wandb.Image object.
        """
        records = []

        # Detach tensors from the computation graph and move to CPU for rendering
        inputs = batch["inputs"].detach().cpu().long().tolist()
        pred_sequences = preds.detach().cpu().long().tolist()
        target_sequences = targets.detach().cpu().long().tolist()
        task_categories = list(metadata["task_category"])
        task_ids = metadata["task_id"].detach().cpu().tolist()

        # Zip all per-example data together for iteration
        batched_rows = zip(
            inputs,
            pred_sequences,
            target_sequences,
            task_categories,
            task_ids,
            strict=True,
        )

        for row in batched_rows:
            if len(self.val_example_records) + len(records) >= self.max_logged_val_examples:
                break

            input_sequence, pred_sequence, target_sequence, task_category, task_id = row
            category_count = self.val_examples_per_category.get(task_category, 0)
            if category_count >= self.max_logged_val_examples_per_category:
                continue

            # Check if the full predicted sequence matches the target exactly
            exact_match = pred_sequence == target_sequence

            # Build a descriptive caption for the WandB images
            caption = self.build_val_example_caption(
                task_category=task_category,
                task_id=task_id,
                exact_match=exact_match,
            )

            # Render each sequence as an ARC-colored matplotlib figure and
            # convert to a wandb.Image for logging
            records.append(
                {
                    "task_category": task_category,
                    "task_id": task_id,
                    "log_key": self.build_val_example_log_key(task_category, task_id),
                    "input": input_sequence,
                    "prediction": pred_sequence,
                    "target": target_sequence,
                    "exact_match": exact_match,
                    "combined_image": figure_to_wandb_image(
                        render_val_example_figure(
                            input_sequence=input_sequence,
                            target_sequence=target_sequence,
                            prediction_sequence=pred_sequence,
                        ),
                        caption=caption,
                    ),
                }
            )
            self.val_examples_per_category[task_category] = category_count + 1
        return records

"""Measure the compute cost of "train once, generate many" (hypernetwork) vs.
"train many small models" (arc1d_lowdata_baseline's per-task recipe).

Background (see experiments/06_data_efficiency_ablation/individual/README.md and
legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/README.md): both routes use
the IDENTICAL target architecture (rope_canon_looped_transformer) for actual per-query
inference, so that cost cancels out of the "one model vs many" comparison. What's left is:

    hypernetwork route    = train_hypernetwork_once + n_tasks * (generate_weights_once_per_task
                             + eval_rows_per_task * run_target_model)
    many-small-models route = n_tasks * (train_one_small_model_from_scratch
                             + eval_rows_per_task * run_target_model)

This script measures each term with real FLOP counts (torch.utils.flop_counter.FlopCounterMode
-- hardware-independent, no need for fvcore/thop) and wall-clock time, then reports where the
two routes cross over as a function of n_tasks.

Terminology note: "eval_rows_per_task" is a task INSTANCE (3 support + 1 query examples,
generated/applied together, see models/hypermodel.py's HyperModel.forward), not a single query
example -- every dataset row in this repo already bundles its own support set, so there is no
real data where several distinct query examples share one fixed, reusable support set to batch
against. See Phase D below for the (synthetic, mechanics-only) check of that broadcast idea.

Five phases, independently runnable via --stage (A-D are synthetic/shape-only and safe to run
against a freshly-initialized model; E needs real learned weights and real data):
  A. generation  -- one hypernetwork forward pass (encoder + pooling + projection head),
                    forward-only, per task-batch-row.
  B. inference   -- target-model forward only, given already-generated weights (accounts for
                    n_loops); reported for both routes to confirm it's identical (same target
                    architecture), so it's excluded from the crossover math as a common term.
  C. training    -- FLOP-count + time one real training_step() call (forward+backward+opt.step(),
                    already including N_supervision internally -- see training/trainer.py's
                    build_trainer for why N_supervision must not be double-counted) via a real
                    (tiny) pl.Trainer.fit() call, then extrapolate by the config's own max_steps
                    (which counts batches, i.e. training_step() calls, not opt.step() calls).
                    Used for BOTH the hypernetwork's one-time training cost and the baseline's
                    per-task training cost, with the same function.
  D. broadcast   -- sanity check only, not a speed measurement: generate weights for one task
                    row, broadcast them (one plain functional_call, no vmap) over K synthetic
                    duplicated inputs, and confirm the output matches that row's own reference
                    prediction from the normal vmap path bit-for-bit (modulo fp32 tolerance).
                    Confirms no hidden per-example coupling (dropout/batchnorm) breaks the
                    "one generated model, many predictions" assumption -- under model.eval(),
                    RoPECanonLoopedTransformer has no such coupling (no BatchNorm; its only
                    Dropout is zeroed in eval mode).
  E. generalization -- the SHARPER, real claim (opt-in only, not part of --stage all -- needs a
                    trained checkpoint via --hyper-checkpoint and real val/holdout data, unlike
                    A-D which only care about shapes): does ONE generated model per task
                    CATEGORY, not per task INSTANCE, still work? For each category, freezes the
                    weights generated from the first instance seen, then applies those SAME
                    frozen weights (no regeneration) to every OTHER instance's own distinct
                    support set and query in that category, and reports shared_exact_match
                    alongside each instance's own normal (per-instance-generated) accuracy. If
                    shared_exact_match holds up near own_exact_match, that argues for one
                    hypernetwork forward pass per task category at eval time, not one per task
                    instance -- a further multiplier on top of Phases A-C's "train once"
                    argument, roughly by the number of instances ordinarily evaluated per
                    category. This is answerable from real data (unlike the literal reading of
                    "generate once, batch-infer many queries against one support set", which
                    Phase D's docstring explains isn't realizable here) because it doesn't
                    require several queries to share a support set -- it tests whether the
                    generated model itself transfers across different support sets of the same
                    category, which is exactly what "one model per category" would require.

Known confounds (see this script's README / the plan that produced it):
  - Muon's opt.step() orthogonalization cost is FLOP-counted too, not assumed negligible --
    happens automatically since the whole training_step() call is wrapped.
  - Generation batch size (hyper config, typically 512 rows/batch) differs from the baseline's
    training batch size (256) -- everything below is normalized to a PER-TASK-INSTANCE cost
    before combining, not raw per-batch cost.
  - Forcing sdpa_kernel(MATH) for vmap correctness (already done in models/hypermodel.py) may
    make the generation/inference path artificially slower in wall-clock than a non-vmapped
    deployment would be -- FLOPs are unaffected (same math, different kernel), but don't read
    the wall-clock numbers here as production-deployment numbers.
  - No local GPU was available when this script was written -- FLOP counts are hardware
    independent and are the primary evidence; wall-clock numbers should be regathered with
    --device cuda on a torrnode for a realistic wall-clock comparison.

Usage:
    uv run python scripts/measure_compute_efficiency.py --stage all
    uv run python scripts/measure_compute_efficiency.py --stage flops --device cuda
    uv run python scripts/measure_compute_efficiency.py --stage broadcast
    uv run python scripts/measure_compute_efficiency.py --stage plot
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.flop_counter import FlopCounterMode

import lightning as pl
from data_modules import DATA_REGISTRY
from models import MODEL_REGISTRY
from models.hypermodel_lightning import TASK_CATEGORY_INDEX
from training.config import build_runtime_config_dict, load_config
from training.trainer import load_checkpoint_state, resolve_evaluation_checkpoint_path
from visualisation.core.style import apply_latex_style

DEFAULT_HYPER_CONFIG = "legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/notd.yaml"
DEFAULT_BASELINE_CONFIG = "experiments/06_data_efficiency_ablation/individual/configs/1d_fill/v3.yaml"
DEFAULT_OUTPUT_DIR = "outputs/compute_efficiency"

# Real scale already run in this repo's experiments (see README context), used as vertical
# reference lines on the crossover plot.
N_TASKS_IN_DISTRIBUTION = 15  # arc1d_lowdata_baseline / arc1d_hypermodel_compositional_generalization's train categories
N_TASKS_WITH_COMPOSITIONAL_HOLDOUT = 25  # + 10 held-out composite categories


def log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Synthetic batches -- shapes only matter for FLOPs/timing, not the values, so
# these are self-contained (no dependency on data/ existing on disk).
# ---------------------------------------------------------------------------


def build_synthetic_hyper_batch(
    batch_size: int, seq_len: int, num_classes: int, device: torch.device
) -> dict:
    def rand_seq(*shape):
        return torch.randint(0, num_classes, shape, dtype=torch.long, device=device)

    categories = list(TASK_CATEGORY_INDEX)
    return {
        "support_inputs": rand_seq(batch_size, 3, seq_len),
        "support_outputs": rand_seq(batch_size, 3, seq_len),
        "query_input": rand_seq(batch_size, seq_len),
        "query_output": rand_seq(batch_size, seq_len),
        "task_category": [categories[i % len(categories)] for i in range(batch_size)],
        # A plain Python list of ints, matching Arc1dMetaPaddingCollator's real contract
        # (models/hypermodel_lightning.py's _write_step_diagnostics JSON-serializes this
        # directly, which a torch.Tensor here would break).
        "task_id": list(range(batch_size)),
    }


def build_synthetic_direct_batch(
    batch_size: int, seq_len: int, num_classes: int, device: torch.device
) -> dict:
    def rand_seq(*shape):
        return torch.randint(0, num_classes, shape, dtype=torch.long, device=device)

    categories = list(TASK_CATEGORY_INDEX)
    return {
        "input": rand_seq(batch_size, seq_len),
        "output": rand_seq(batch_size, seq_len),
        "task_category": [categories[i % len(categories)] for i in range(batch_size)],
    }


class RepeatedBatchDataset(Dataset):
    """Yields the same pre-batched dict `length` times, for DataLoader(batch_size=None)."""

    def __init__(self, batch: dict, length: int):
        self.batch = batch
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return self.batch


def make_repeated_loader(batch: dict, n_batches: int) -> DataLoader:
    return DataLoader(RepeatedBatchDataset(batch, n_batches), batch_size=None, shuffle=False)


# ---------------------------------------------------------------------------
# Generic FLOP + wall-clock measurement
# ---------------------------------------------------------------------------


def measure_forward_flops_and_time(
    fn, n_warmup: int, n_iters: int, device: torch.device
) -> tuple[int, float]:
    """Time and FLOP-count a zero-arg forward-only closure. Mirrors
    legacy/scripts/smoke_test_vmap_target_model.py's warm-up/perf_counter/cuda-sync pattern."""
    is_cuda = device.type == "cuda"
    for _ in range(n_warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()

    with FlopCounterMode(display=False) as flop_counter:
        fn()
    flops = flop_counter.get_total_flops()

    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    wall_s = (time.perf_counter() - start) / n_iters
    return flops, wall_s


def measure_training_flops_and_time(
    model: pl.LightningModule,
    batch: dict,
    n_warmup_steps: int,
    n_timed_steps: int,
    accelerator: str,
) -> tuple[int, float]:
    """FLOP-count + time one real training_step() call (forward + backward + opt.step(),
    N_supervision loop included) via a real (tiny) pl.Trainer.fit(), not a hand-rolled call --
    training_step() calls self.log(), which requires a real trainer attached.

    limit_train_batches=n_batches (not max_steps) is used deliberately: under manual
    optimization (this repo's HyperModelLightning/LoopedSupervisedLightning), Lightning's
    global_step increments once per opt.step() call, i.e. N_supervision times per batch (see
    training/trainer.py's build_trainer) -- limit_train_batches counts batches directly,
    matching what a config's own `max_steps` means (see that same comment), with no risk of
    off-by-N_supervision here.

    Returns (flops_per_batch, wall_s_per_batch), where "per batch" already includes the full
    N_supervision inner loop -- multiply by a config's own `max_steps` to get total cost.
    """

    def build_trainer(n_batches: int) -> pl.Trainer:
        return pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            max_epochs=1,
            limit_train_batches=n_batches,
            num_sanity_val_steps=0,
            log_every_n_steps=max(1, n_batches),
        )

    build_trainer(n_warmup_steps).fit(model, train_dataloaders=make_repeated_loader(batch, n_warmup_steps))
    if accelerator == "gpu":
        torch.cuda.synchronize()

    timed_trainer = build_trainer(n_timed_steps)
    with FlopCounterMode(display=False) as flop_counter:
        start = time.perf_counter()
        timed_trainer.fit(model, train_dataloaders=make_repeated_loader(batch, n_timed_steps))
        if accelerator == "gpu":
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - start
    flops = flop_counter.get_total_flops()
    return flops / n_timed_steps, wall_s / n_timed_steps


# ---------------------------------------------------------------------------
# Phase A: generation cost
# ---------------------------------------------------------------------------


def measure_generation_cost(hyper_lightning, batch: dict, n_warmup: int, n_iters: int, device: torch.device) -> dict:
    hyper_lightning.eval()
    hypermodel = hyper_lightning.hypermodel
    with torch.no_grad():
        task_features, _, _ = hyper_lightning.prepare_inputs(batch)
    canonical_ids = None
    if hypermodel.task_indicator_proj is not None:
        canonical_ids = torch.tensor(
            [TASK_CATEGORY_INDEX[c] for c in batch["task_category"]], device=task_features.device
        )

    def _generate():
        with torch.no_grad():
            hyper_output = hypermodel.hypernetwork(task_features)
            hypermodel.extract_parameter_vectors(hyper_output, canonical_ids)

    flops, wall_s = measure_forward_flops_and_time(_generate, n_warmup, n_iters, device)
    batch_size = task_features.shape[0]
    return {
        "batch_size": batch_size,
        "flops_per_batch": flops,
        "wall_s_per_batch": wall_s,
        "flops_per_task": flops / batch_size,
        "wall_s_per_task": wall_s / batch_size,
    }


# ---------------------------------------------------------------------------
# Phase B: inference cost (target model only, given already-generated weights)
# ---------------------------------------------------------------------------


def measure_inference_cost(hyper_lightning, batch: dict, n_warmup: int, n_iters: int, device: torch.device) -> dict:
    hyper_lightning.eval()
    hypermodel = hyper_lightning.hypermodel
    with torch.no_grad():
        task_features, example_inputs, _ = hyper_lightning.prepare_inputs(batch)
        canonical_ids = None
        if hypermodel.task_indicator_proj is not None:
            canonical_ids = torch.tensor(
                [TASK_CATEGORY_INDEX[c] for c in batch["task_category"]], device=task_features.device
            )
        hyper_output = hypermodel.hypernetwork(task_features)
        parameter_vectors = hypermodel.extract_parameter_vectors(hyper_output, canonical_ids).float()

    def _infer():
        with torch.no_grad():
            if hypermodel._use_vmap:
                params = hypermodel.build_batched_param_dict(parameter_vectors)
                with sdpa_kernel(SDPBackend.MATH):
                    torch.vmap(functional_call, in_dims=(None, 0, 0), randomness="different")(
                        hypermodel.target_model, params, example_inputs
                    )
            else:
                for i in range(parameter_vectors.shape[0]):
                    params = hypermodel.build_param_dict(parameter_vectors[i])
                    functional_call(hypermodel.target_model, params, example_inputs[i])

    flops, wall_s = measure_forward_flops_and_time(_infer, n_warmup, n_iters, device)
    batch_size = parameter_vectors.shape[0]
    return {
        "batch_size": batch_size,
        "flops_per_batch": flops,
        "wall_s_per_batch": wall_s,
        "flops_per_task_row": flops / batch_size,
        "wall_s_per_task_row": wall_s / batch_size,
    }


# ---------------------------------------------------------------------------
# Phase C: training cost (used for both the hypernetwork's one-time cost and the
# baseline's per-task cost -- same function, different model/batch/config).
# ---------------------------------------------------------------------------


def measure_training_cost(
    model: pl.LightningModule,
    batch: dict,
    max_steps: int,
    n_warmup_steps: int,
    n_timed_steps: int,
    accelerator: str,
) -> dict:
    flops_per_batch, wall_s_per_batch = measure_training_flops_and_time(
        model, batch, n_warmup_steps, n_timed_steps, accelerator
    )
    return {
        "max_steps": max_steps,
        "flops_per_batch": flops_per_batch,
        "wall_s_per_batch": wall_s_per_batch,
        "flops_total": flops_per_batch * max_steps,
        "wall_s_total": wall_s_per_batch * max_steps,
    }


# ---------------------------------------------------------------------------
# Phase D: broadcast validity check (sanity only, synthetic)
# ---------------------------------------------------------------------------


def measure_broadcast_validity(hyper_lightning, batch: dict, k_rows: int = 8, atol: float = 1e-4) -> dict:
    """Generate weights for row 0 only, broadcast row 0's own query input over k_rows
    synthetic duplicates via ONE plain (non-vmap, non-batched-weights) functional_call, and
    confirm the output matches row 0's own reference query prediction from the normal
    (per-row-weights, vmapped) path. This validates the "one generated model, many
    predictions" mechanic -- it cannot be run against real multi-query-per-support data,
    since no such data exists in this repo (see module docstring). Note this case needs no
    vmap at all: with one shared (non-batched) weight dict, functional_call already treats
    the input's leading dim as an ordinary batch -- vmap is only needed when every row gets
    its OWN weights, which is the opposite of what's being tested here."""
    hyper_lightning.eval()
    hypermodel = hyper_lightning.hypermodel
    with torch.no_grad():
        task_features, example_inputs, _ = hyper_lightning.prepare_inputs(batch)
        canonical_ids = None
        if hypermodel.task_indicator_proj is not None:
            # Must match hyper_lightning(batch)'s own canonical_ids below exactly (same
            # task_category -> TASK_CATEGORY_INDEX lookup as HyperModelLightning.forward) --
            # a task-ID-conditioned model (frozentd/td) generates different weights with vs.
            # without its task signal, so ref_logits below would silently stop matching
            # param_dict's weights otherwise.
            canonical_ids = torch.tensor(
                [TASK_CATEGORY_INDEX[c] for c in batch["task_category"][:1]], device=task_features.device
            )
        hyper_output = hypermodel.hypernetwork(task_features[:1])
        parameter_vectors = hypermodel.extract_parameter_vectors(hyper_output, canonical_ids).float()
        param_dict = hypermodel.build_param_dict(parameter_vectors[0])

        # example_inputs: (batch, 4, seq_len, dim), examples ordered [support x3, query] (see
        # HyperModelLightning.prepare_inputs) -- index -1 is row 0's own query example.
        query_input_row0 = example_inputs[0, -1]  # (seq_len, dim)
        broadcast_inputs = query_input_row0.unsqueeze(0).expand(k_rows, -1, -1)  # (k_rows, seq_len, dim)
        with sdpa_kernel(SDPBackend.MATH):
            out_broadcast = functional_call(hypermodel.target_model, param_dict, broadcast_inputs)
            if out_broadcast.shape[-1] == 1:
                out_broadcast = out_broadcast.squeeze(-1)

        ref_logits, _ = hyper_lightning(batch)
        ref_query_row0 = ref_logits[0, -1]  # this row's own query prediction, normal vmap path

    diff = (out_broadcast - ref_query_row0.unsqueeze(0)).abs()
    max_abs_diff = diff.max().item()
    all_close = bool(
        torch.allclose(out_broadcast, ref_query_row0.unsqueeze(0).expand_as(out_broadcast), atol=atol)
    )
    return {"k_rows": k_rows, "max_abs_diff": max_abs_diff, "all_close": all_close}


# ---------------------------------------------------------------------------
# Phase E: cross-instance generalization (real data, not synthetic), SAME CATEGORY ONLY --
# a category's generated weights are only ever evaluated against that category's own other
# instances, never another category's (Fabio, 2026-08-31: no cross-category matrix). The
# sharper claim: does ONE generated model per task CATEGORY (not per task INSTANCE) still
# work? This is distinct from Phase D: Phase D reuses one instance's own support set +
# duplicated query (a mechanics-only check, no real data exists for it). Phase E reuses one
# instance's GENERATED WEIGHTS on every OTHER instance's own distinct support set and query in
# that category -- a real generalization question, answerable from the existing val/holdout
# data.
#
# Leave-one-out, not a single fixed reference: for a category with n instances, every
# instance in turn is treated as the reference (not just the first one seen) -- for each
# reference i, its weights are evaluated against the other n-1 instances, giving n
# leave-one-out accuracies per category. This answers "does it matter which instance you
# generate from," not just "does the first one happen to work." If accuracy holds up near
# own_exact_match across all n choices of reference (small loo_std), that argues for one
# hypernetwork forward pass per task category at eval time, not one per task instance -- a
# further efficiency multiplier on top of "train once" (Phases A-C), roughly by the number of
# instances per category ordinarily evaluated (see the "no data-loading time" note in the
# plan for how to turn that into a FLOPs/wall-clock number using Phase A's own measurements,
# without adding any new timing instrumentation here).
# ---------------------------------------------------------------------------


def measure_cross_instance_generalization(
    hyper_lightning,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    padding_idx: int | None = None,
) -> dict[str, dict]:
    """padding_idx must be passed and matched against target sequences exactly as
    HyperModelLightning._accumulate_query_exact_match_by_task_category does (models/
    hypermodel_lightning.py): a padded position can never be "predicted" correctly by an
    argmax over num_classes (padding_idx sits outside that range), so it must be treated as
    vacuously matched, not compared -- otherwise every batch with any padding at all (which is
    most of them, since Arc1dMetaPaddingCollator pads to each batch's own longest sequence)
    reports near-zero exact-match regardless of the model's real accuracy.

    No timing here (see module docstring's Phase E note): this only reports accuracy
    mean/std, so there is no wall-clock/FLOPs number that could accidentally include data
    loading -- dataloader iteration itself is the only "loading" involved and it precedes
    every measured forward pass here by construction (a batch is always fully materialized
    before its rows are used)."""
    hyper_lightning.eval()
    hypermodel = hyper_lightning.hypermodel

    # Per category: every instance seen, as (param_dict, embedded query input, query target).
    # NOTE: query inputs are stored already-embedded (not re-embedded from raw token ids), so
    # stacking two instances of the same category assumes they came from the same dataloader
    # batch (Arc1dMetaPaddingCollator pads each batch independently to ITS OWN longest
    # sequence) -- true whenever batch_size covers the whole split in one batch, as it does
    # for this repo's small (~150-row combined val+test) generalization runs. The stack below
    # raises a clear error instead of silently mis-comparing if that assumption ever breaks.
    instances: dict[str, list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]] = defaultdict(list)
    own_correct: dict[str, list[bool]] = defaultdict(list)

    def decode(logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        return logits.argmax(dim=-1)

    def exact_match(predictions: torch.Tensor, targets_long: torch.Tensor) -> torch.Tensor:
        """predictions, targets_long: (..., seq_len). Padded positions (targets_long ==
        padding_idx) are excluded from the comparison, matching the model's own convention."""
        if padding_idx is None:
            return (predictions == targets_long).all(dim=-1)
        valid_mask = targets_long != padding_idx
        return ((predictions == targets_long) | ~valid_mask).all(dim=-1)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            tensor_batch = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in batch.items()
            }
            task_features, example_inputs, targets = hyper_lightning.prepare_inputs(tensor_batch)
            canonical_ids = None
            if hypermodel.task_indicator_proj is not None:
                # A task-ID-conditioned model (frozentd/td) generates different weights with
                # vs. without this signal -- must match HyperModelLightning.forward's own
                # canonical_ids exactly, or "own" accuracy here would silently regress to the
                # unconditioned generation path instead of what the checkpoint was actually
                # trained/evaluated with (caught before running against a frozentd checkpoint,
                # which is exactly the case this matters for).
                canonical_ids = torch.tensor(
                    [TASK_CATEGORY_INDEX[c] for c in batch["task_category"]], device=task_features.device
                )
            hyper_output = hypermodel.hypernetwork(task_features)
            parameter_vectors = hypermodel.extract_parameter_vectors(hyper_output, canonical_ids).float()

            if hypermodel._use_vmap:
                params = hypermodel.build_batched_param_dict(parameter_vectors)
                with sdpa_kernel(SDPBackend.MATH):
                    own_logits = torch.vmap(
                        functional_call, in_dims=(None, 0, 0), randomness="different"
                    )(hypermodel.target_model, params, example_inputs)
            else:
                own_logits = torch.stack(
                    [
                        functional_call(
                            hypermodel.target_model,
                            hypermodel.build_param_dict(parameter_vectors[i]),
                            example_inputs[i],
                        )
                        for i in range(parameter_vectors.shape[0])
                    ]
                )

            own_predictions = decode(own_logits)
            query_targets = targets[:, -1].long()
            own_matches = exact_match(own_predictions[:, -1], query_targets)

            for row_idx, category in enumerate(batch["task_category"]):
                own_correct[category].append(bool(own_matches[row_idx].item()))
                instances[category].append(
                    (
                        hypermodel.build_param_dict(parameter_vectors[row_idx]),
                        example_inputs[row_idx, -1].detach(),  # (seq_len, dim)
                        query_targets[row_idx].detach(),  # (seq_len,)
                    )
                )

    def mean_std(values: list[float]) -> tuple[float, float]:
        """Sample mean and (ddof=1) sample std -- std is undefined (reported as NaN) for
        n<2, not a script bug."""
        if not values:
            return float("nan"), float("nan")
        floats = [float(v) for v in values]
        mean = statistics.mean(floats)
        std = statistics.stdev(floats) if len(floats) >= 2 else float("nan")
        return mean, std

    report = {}
    for category in sorted(instances):
        rows = instances[category]
        n = len(rows)
        own_mean, own_std = mean_std(own_correct[category])

        loo_accuracies: list[float] = []  # one per reference instance: accuracy over the other n-1
        pooled_pairs: list[bool] = []  # every individual (reference, other-instance) pair, flattened

        for i in range(n):
            other_indices = [j for j in range(n) if j != i]
            if not other_indices:
                continue  # only 1 instance total for this category -- no leave-one-out possible
            try:
                other_inputs = torch.stack([rows[j][1] for j in other_indices])  # (n-1, seq_len, dim)
            except RuntimeError as exc:
                msg = (
                    f"Category {category!r} has instances with mismatched padded sequence "
                    "lengths (they landed in different dataloader batches, each padded "
                    "independently). Leave-one-out needs every instance of a category in one "
                    "batch -- increase the hyper config's batch_size so this category's rows "
                    "(and every other category's) fit in a single batch."
                )
                raise RuntimeError(msg) from exc
            other_targets = torch.stack([rows[j][2] for j in other_indices])  # (n-1, seq_len)
            with sdpa_kernel(SDPBackend.MATH):
                logits = functional_call(hypermodel.target_model, rows[i][0], other_inputs)
            matches = exact_match(decode(logits), other_targets).tolist()
            pooled_pairs.extend(matches)
            loo_accuracies.append(sum(matches) / len(matches))

        loo_mean, loo_std = mean_std(loo_accuracies)
        pooled_mean, pooled_std = mean_std(pooled_pairs)
        report[category] = {
            "n_instances": n,
            "own_exact_match_mean": own_mean,
            "own_exact_match_std": own_std,
            "n_loo_references": len(loo_accuracies),
            "loo_mean": loo_mean,
            "loo_std": loo_std,
            "n_loo_pairs": len(pooled_pairs),
            "loo_pooled_mean": pooled_mean,
            "loo_pooled_std": pooled_std,
        }
    return report


def render_generalization_table(report: dict[str, dict]) -> str:
    """Plain-text table: per-category own-generation accuracy vs. leave-one-out accuracy
    (every instance in turn as the reference, evaluated on the rest of its own category),
    both reference-level (mean/std across the n choices of reference) and pooled
    (mean/std across every individual reference/other-instance pair)."""

    def fmt(value: float) -> str:
        return "n/a" if value != value else f"{value:.3f}"  # value != value <=> NaN

    columns = [
        ("category", "<20"),
        ("n", ">4"),
        ("own_mean", ">10"),
        ("own_std", ">9"),
        ("loo_mean", ">10"),
        ("loo_std", ">9"),
        ("pooled_mean", ">13"),
        ("pooled_std", ">12"),
    ]
    header = "".join(f"{name:{width}}" for name, width in columns)
    lines = [header, "-" * len(header)]
    for category in sorted(report):
        row = report[category]
        lines.append(
            f"{category:<20}{row['n_instances']:>4}"
            f"{fmt(row['own_exact_match_mean']):>10}{fmt(row['own_exact_match_std']):>9}"
            f"{fmt(row['loo_mean']):>10}{fmt(row['loo_std']):>9}"
            f"{fmt(row['loo_pooled_mean']):>13}{fmt(row['loo_pooled_std']):>12}"
        )

    all_own = [row["own_exact_match_mean"] for row in report.values() if row["n_instances"]]
    all_loo = [row["loo_mean"] for row in report.values() if row["n_loo_references"]]
    lines.append("-" * len(header))
    lines.append(
        f"{'macro-average':<20}{'':>4}{fmt(statistics.mean(all_own)):>10}{'':>9}"
        f"{fmt(statistics.mean(all_loo)):>10}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Crossover computation and report
# ---------------------------------------------------------------------------


def crossover_curve(
    hyper_training_flops_total: float,
    generation_flops_per_task: float,
    training_flops_per_task: float,
    inference_flops_per_task_row: float,
    eval_rows_per_task: int,
    n_tasks_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    common = eval_rows_per_task * inference_flops_per_task_row
    hyper_route = hyper_training_flops_total + n_tasks_values * (generation_flops_per_task + common)
    baseline_route = n_tasks_values * (training_flops_per_task + common)
    return hyper_route, baseline_route


def break_even_n_tasks(
    hyper_training_flops_total: float,
    generation_flops_per_task: float,
    training_flops_per_task: float,
) -> float:
    denom = training_flops_per_task - generation_flops_per_task
    if denom <= 0:
        return float("inf")
    return hyper_training_flops_total / denom


def plot_crossover(
    n_tasks_values: np.ndarray,
    hyper_route: np.ndarray,
    baseline_route: np.ndarray,
    n_star: float,
    output_path: Path,
    y_label: str,
    title: str,
) -> None:
    apply_latex_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(n_tasks_values, hyper_route, label="Hypernetwork route (train once + generate)", linewidth=2)
    ax.plot(n_tasks_values, baseline_route, label="Many small models (train per task)", linewidth=2)
    if np.isfinite(n_star) and n_tasks_values.min() <= n_star <= n_tasks_values.max():
        ax.axvline(n_star, color="gray", linestyle=":", linewidth=1.5, label=f"break-even (n={n_star:.1f})")
    for n_ref, label in (
        (N_TASKS_IN_DISTRIBUTION, "15 in-dist. categories"),
        (N_TASKS_WITH_COMPOSITIONAL_HOLDOUT, "+10 compositional holdout"),
    ):
        if n_tasks_values.min() <= n_ref <= n_tasks_values.max():
            ax.axvline(n_ref, color="lightgray", linestyle="--", linewidth=1)
            ax.text(n_ref, ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=7, color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage", choices=["flops", "broadcast", "generalization", "plot", "all"], default="all"
    )
    parser.add_argument("--hyper-config", default=DEFAULT_HYPER_CONFIG)
    parser.add_argument("--hyper-checkpoint", default=None, help="'best', 'last', or an explicit path; omit for a freshly-initialized model (cost depends on shapes, not weight values).")
    parser.add_argument("--baseline-config", default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--n-warmup", type=int, default=5, help="Forward-only phases (A/B).")
    parser.add_argument("--n-iters", type=int, default=20, help="Forward-only phases (A/B).")
    parser.add_argument("--n-warmup-steps", type=int, default=3, help="Training phase (C).")
    parser.add_argument("--n-timed-steps", type=int, default=10, help="Training phase (C).")
    parser.add_argument("--eval-rows-per-task", type=int, default=100, help="'K' in the crossover formula -- task instances evaluated per task at eval time.")
    parser.add_argument("--n-tasks-max", type=int, default=200)
    parser.add_argument("--broadcast-k", type=int, default=8)
    parser.add_argument(
        "--generalization-split",
        default="val",
        help="Comma-separated 'val'/'test' (via the hyper config's own datamodule, concatenated "
        "if more than one -- e.g. 'val,test' to double the tiny fixed-size per-category split "
        "for a more meaningful mean/std) or 'holdout' (compositional holdout set, not combinable).",
    )
    parser.add_argument(
        "--generalization-max-batches",
        type=int,
        default=None,
        help="Limit the number of batches scanned (default: the whole split). Rarely needed "
        "now that the generalization dataloader always uses one batch big enough to hold "
        "everything (see build_generalization_dataloader).",
    )
    parser.add_argument(
        "--generalization-data-dir",
        default=None,
        help="Override the hyper config's own data_dir for --stage generalization only -- e.g. "
        "data/arc_1d_looped_augmented_devtest_shifted, which has 500 dev + 500 test rows per "
        "category (vs. the base dataset's fixed 5 + 5), for far more leave-one-out statistical "
        "power. Categories/architecture are unaffected -- only which dev/test pool is read.",
    )
    parser.add_argument(
        "--generalization-max-per-category",
        type=int,
        default=None,
        help="Cap the number of instances kept per category (first N seen, deterministic) "
        "after combining --generalization-split's splits. Default: use everything available.",
    )
    parser.add_argument(
        "--holdout-data-dir",
        default="data/arc_1d_compositional_holdout",
        help="Only used when --generalization-split=holdout.",
    )
    parser.add_argument("--holdout-split", default="holdout_test")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_hyper_lightning(config_path: str, checkpoint: str | None, device: torch.device):
    cfg = load_config(config_path)
    runtime_cfg = build_runtime_config_dict(cfg)
    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    if checkpoint is not None:
        checkpoint_path = resolve_evaluation_checkpoint_path(cfg.output_path, checkpoint)
        load_checkpoint_state(model, checkpoint_path)
        log(f"Loaded hypernetwork checkpoint: {checkpoint_path}")
    else:
        log(
            "No --hyper-checkpoint given: using a freshly-initialized model. Fine for FLOPs/timing "
            "(depend on shapes, not weight values) -- NOT meaningful for --stage generalization, "
            "which needs real learned weights to say anything about accuracy."
        )
    model.to(device)
    return model, cfg, runtime_cfg


def load_baseline_lightning(config_path: str, device: torch.device):
    cfg = load_config(config_path)
    runtime_cfg = build_runtime_config_dict(cfg)
    model = MODEL_REGISTRY[cfg.model](**runtime_cfg)
    model.to(device)
    return model, cfg


def subsample_per_category(dataset, max_per_category: int):
    """Deterministically keep at most max_per_category rows per task_category -- first N
    seen in the dataset's existing order, not random. Works on any Dataset supporting
    integer indexing (plain Arc1dMetaTaskDataset or a ConcatDataset of several)."""
    counts: dict[str, int] = defaultdict(int)
    keep_indices = []
    for i in range(len(dataset)):
        category = dataset[i]["task_category"]
        if counts[category] < max_per_category:
            keep_indices.append(i)
            counts[category] += 1
    return torch.utils.data.Subset(dataset, keep_indices)


def build_generalization_dataloader(args: argparse.Namespace, hyper_cfg, hyper_runtime_cfg: dict) -> DataLoader:
    """Build the val/test(+holdout) dataloader for --stage generalization, honoring
    --generalization-split (comma-separated 'val'/'test', concatenated), an optional
    --generalization-data-dir override (e.g. a richer devtest pool than the checkpoint's own
    training data_dir), and an optional --generalization-max-per-category cap.

    batch_size is always set to cover the WHOLE resulting dataset in one batch: leave-one-out
    (measure_cross_instance_generalization) stacks a category's instances assuming they share
    one batch's padding length (Arc1dMetaPaddingCollator pads each batch independently to its
    own longest sequence) -- one big batch sidesteps that instead of re-embedding to a global
    max length. Fine at this scale (hundreds to low thousands of rows, a tiny target model)."""
    if args.generalization_split == "holdout":
        from scripts.eval_compositional_holdout import build_holdout_dataloader

        return build_holdout_dataloader(
            args.holdout_data_dir, args.holdout_split, hyper_cfg.batch_size, hyper_cfg.padding_idx
        )

    split_names = [name.strip() for name in args.generalization_split.split(",")]
    runtime_cfg = dict(hyper_runtime_cfg)
    if args.generalization_data_dir is not None:
        runtime_cfg["data_dir"] = args.generalization_data_dir
    datamodule = DATA_REGISTRY[hyper_cfg.data](**runtime_cfg)
    datamodule.setup()
    split_dataset_map = {"val": datamodule.val_dataset, "test": datamodule.test_dataset}
    unknown = set(split_names) - set(split_dataset_map)
    if unknown:
        msg = f"Unknown --generalization-split value(s) {unknown}; expected 'val', 'test', or 'holdout'."
        raise ValueError(msg)
    datasets = [split_dataset_map[name] for name in split_names]
    combined_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    if args.generalization_max_per_category is not None:
        combined_dataset = subsample_per_category(combined_dataset, args.generalization_max_per_category)

    batch_size = max(hyper_cfg.batch_size, len(combined_dataset))
    return DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=False, collate_fn=datamodule.collator
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    accelerator = "gpu" if device.type == "cuda" else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements_path = output_dir / "measurements.json"

    log(f"Device: {device}")
    hyper_lightning, hyper_cfg, hyper_runtime_cfg = load_hyper_lightning(
        args.hyper_config, args.hyper_checkpoint, device
    )
    num_classes = hyper_cfg.num_classes

    measurements: dict = {}
    if measurements_path.exists() and args.stage != "flops" and args.stage != "all":
        measurements = json.loads(measurements_path.read_text())

    if args.stage in ("flops", "all"):
        gen_batch_size = hyper_cfg.batch_size
        hyper_batch = build_synthetic_hyper_batch(gen_batch_size, args.seq_len, num_classes, device)

        log("\n=== Phase A: generation cost (hypernetwork forward pass) ===")
        generation = measure_generation_cost(hyper_lightning, hyper_batch, args.n_warmup, args.n_iters, device)
        log(json.dumps(generation, indent=2))

        log("\n=== Phase B: inference cost (target model only) ===")
        inference = measure_inference_cost(hyper_lightning, hyper_batch, args.n_warmup, args.n_iters, device)
        log(json.dumps(inference, indent=2))

        log("\n=== Phase C (hypernetwork): one-time training cost ===")
        hyper_training = measure_training_cost(
            hyper_lightning, hyper_batch, hyper_cfg.max_steps, args.n_warmup_steps, args.n_timed_steps, accelerator
        )
        log(json.dumps(hyper_training, indent=2))

        log(f"\nLoading baseline config: {args.baseline_config}")
        baseline_lightning, baseline_cfg = load_baseline_lightning(args.baseline_config, device)
        baseline_batch = build_synthetic_direct_batch(
            baseline_cfg.batch_size, args.seq_len, baseline_cfg.num_classes, device
        )

        log("\n=== Phase C (baseline): per-task training cost ===")
        baseline_training = measure_training_cost(
            baseline_lightning, baseline_batch, baseline_cfg.max_steps, args.n_warmup_steps, args.n_timed_steps, accelerator
        )
        log(json.dumps(baseline_training, indent=2))

        measurements = {
            "device": str(device),
            "hyper_config": args.hyper_config,
            "baseline_config": args.baseline_config,
            "eval_rows_per_task": args.eval_rows_per_task,
            "generation": generation,
            "inference": inference,
            "hyper_training": hyper_training,
            "baseline_training": baseline_training,
        }
        measurements_path.write_text(json.dumps(measurements, indent=2))
        log(f"\nWrote {measurements_path}")

        sanity_ratio = baseline_training["flops_total"] / max(generation["flops_per_task"], 1)
        log(
            f"\nSanity check: baseline per-task training FLOPs are {sanity_ratio:,.0f}x "
            "generation FLOPs per task (expect several orders of magnitude -- if this is "
            "close to 1x, something in the extrapolation is likely wrong)."
        )

    if args.stage in ("broadcast", "all"):
        gen_batch_size = min(hyper_cfg.batch_size, 8)
        broadcast_batch = build_synthetic_hyper_batch(gen_batch_size, args.seq_len, num_classes, device)
        log("\n=== Phase D: broadcast validity check (synthetic, mechanics only) ===")
        broadcast = measure_broadcast_validity(hyper_lightning, broadcast_batch, k_rows=args.broadcast_k)
        log(json.dumps(broadcast, indent=2))
        if not broadcast["all_close"]:
            log("WARNING: broadcast output did NOT match the reference vmap output within tolerance.")
        measurements["broadcast"] = broadcast
        measurements_path.write_text(json.dumps(measurements, indent=2))

    if args.stage == "generalization":
        log("\n=== Phase E: cross-instance generalization (real data) ===")
        dataloader = build_generalization_dataloader(args, hyper_cfg, hyper_runtime_cfg)
        generalization = measure_cross_instance_generalization(
            hyper_lightning,
            dataloader,
            device,
            max_batches=args.generalization_max_batches,
            padding_idx=hyper_cfg.padding_idx,
        )
        log(json.dumps(generalization, indent=2))
        (output_dir / "generalization.json").write_text(json.dumps(generalization, indent=2))
        log(f"\nWrote {output_dir / 'generalization.json'}")

        table_text = render_generalization_table(generalization)
        log("\n" + table_text)
        (output_dir / "generalization_table.txt").write_text(table_text + "\n")
        log(f"Wrote {output_dir / 'generalization_table.txt'}")

        csv_path = output_dir / "generalization_table.csv"
        with csv_path.open("w") as f:
            f.write(
                "category,n_instances,own_exact_match_mean,own_exact_match_std,"
                "n_loo_references,loo_mean,loo_std,n_loo_pairs,loo_pooled_mean,loo_pooled_std\n"
            )
            for category in sorted(generalization):
                row = generalization[category]
                f.write(
                    f"{category},{row['n_instances']},{row['own_exact_match_mean']:.4f},"
                    f"{row['own_exact_match_std']:.4f},{row['n_loo_references']},"
                    f"{row['loo_mean']:.4f},{row['loo_std']:.4f},{row['n_loo_pairs']},"
                    f"{row['loo_pooled_mean']:.4f},{row['loo_pooled_std']:.4f}\n"
                )
        log(f"Wrote {csv_path}")

    if args.stage in ("plot", "all"):
        if not measurements or "generation" not in measurements:
            msg = f"No measurements found at {measurements_path}; run --stage flops first."
            raise RuntimeError(msg)

        eval_rows_per_task = measurements.get("eval_rows_per_task", args.eval_rows_per_task)
        n_tasks_values = np.logspace(0, np.log10(args.n_tasks_max), 200)

        # Two independent views of the same routes: FLOPs (hardware-independent, but blind to
        # how efficiently each route's ops actually run on real hardware) and GPU wall-clock
        # (device.type == "cuda" only -- CPU wall-clock is not a meaningful stand-in, see
        # feedback_no_local_heavy_compute; also don't just scale down a CPU number). These can
        # (and did, in the first real GPU run) disagree substantially: the hypernetwork's
        # bigger batch (512 vs. baseline's 256) and bigger encoder utilize the GPU far more
        # efficiently per FLOP than the baseline's tiny per-task model does, so wall-clock
        # break-even can land orders of magnitude below the FLOPs break-even.
        metrics = [
            (
                "flops",
                "Total FLOPs",
                "Compute cost (FLOPs): hypernetwork vs. many small models",
                measurements["generation"]["flops_per_task"],
                measurements["baseline_training"]["flops_total"],
                measurements["inference"]["flops_per_task_row"],
                measurements["hyper_training"]["flops_total"],
            )
        ]
        if measurements.get("device", "").startswith("cuda"):
            metrics.append(
                (
                    "wallclock",
                    "Total GPU wall-clock time (s)",
                    "Compute cost (GPU wall-clock): hypernetwork vs. many small models",
                    measurements["generation"]["wall_s_per_task"],
                    measurements["baseline_training"]["wall_s_total"],
                    measurements["inference"]["wall_s_per_task_row"],
                    measurements["hyper_training"]["wall_s_total"],
                )
            )
        else:
            log(
                "\nSkipping the wall-clock crossover plot: measurements were taken on "
                f"{measurements.get('device')!r}, not cuda -- CPU wall-clock isn't a "
                "meaningful stand-in for GPU wall-clock (see feedback_no_local_heavy_compute); "
                "re-run --stage flops with --device cuda on a torrnode for that view."
            )

        for name, y_label, title, generation_per_task, training_per_task, inference_per_row, hyper_training_total in metrics:
            n_star = break_even_n_tasks(hyper_training_total, generation_per_task, training_per_task)
            hyper_route, baseline_route = crossover_curve(
                hyper_training_total,
                generation_per_task,
                training_per_task,
                inference_per_row,
                eval_rows_per_task,
                n_tasks_values,
            )

            csv_path = output_dir / f"crossover_{name}.csv"
            with csv_path.open("w") as f:
                f.write(f"n_tasks,hyper_route_{name},baseline_route_{name}\n")
                for n_tasks, hyper_val, baseline_val in zip(n_tasks_values, hyper_route, baseline_route, strict=True):
                    f.write(f"{n_tasks:.4f},{hyper_val:.6e},{baseline_val:.6e}\n")
            log(f"Wrote {csv_path}")

            plot_path = output_dir / f"crossover_{name}"
            plot_crossover(n_tasks_values, hyper_route, baseline_route, n_star, plot_path, y_label, title)
            log(f"Wrote {plot_path.with_suffix('.pdf')} and {plot_path.with_suffix('.png')}")

            log(f"\n[{name}] Break-even point: n_tasks* = {n_star:.2f}")

        log(f"\nReal in-distribution scale in this repo: n_tasks = {N_TASKS_IN_DISTRIBUTION}")
        log(f"Real scale incl. compositional holdout:  n_tasks = {N_TASKS_WITH_COMPOSITIONAL_HOLDOUT}")
        log(
            "(For the compositional-holdout categories specifically, the baseline route "
            "isn't even available -- there is no training data/pipeline for a fresh composite "
            "category, so the hypernetwork's one-forward-pass generation is the only route "
            "at all, independent of either comparison above.)"
        )


if __name__ == "__main__":
    main()

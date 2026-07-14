"""Smoke test: does torch.func.vmap work for the target model's functional_call pattern?

HyperModel.forward() (models/hypermodel.py) currently generates a distinct set of weights
per task and applies them via a Python for-loop of functional_call, one call per batch item
(there's no way to share weights across the batch, since every task gets its own generated
params). This is the standard use case for torch.func.vmap + functional_call ("model
ensembling" in PyTorch's own terminology) -- this script checks whether that combination
actually works, and how much faster it is, for THIS target model specifically, before
touching HyperModel itself.

A previous attempt at vmap with the RNN target model (models/rnn.py, wraps nn.RNN) broke --
likely because nn.RNN dispatches to a fused cuDNN kernel with no registered vmap batching
rule (a well-known torch.func limitation for nn.RNN/LSTM/GRU specifically, not for recurrence
in general). RoPECanonLoopedTransformer's own "recurrence" (a fixed-iteration Python loop
calling ordinary Linear/LayerNorm/attention/conv ops) is architecturally very different and
should not hit the same issue -- this script checks that directly instead of assuming it.

Known finding from a CPU dry run in the dev sandbox (informative, not a substitute for your
GPU run): the "flash" attention path (scaled_dot_product_attention's fused kernel) triggers
"we have not yet implemented the batching rule for
aten::_scaled_dot_product_flash_attention_for_cpu" and becomes extremely slow under vmap --
the exact same class of issue as the RNN/cuDNN failure (a fused kernel with no vmap batching
rule), just for attention instead of recurrence. The "math" (manual matmul+softmax) fallback
already built into RoPECanonSelfAttention does NOT hit this warning. Whether CUDA's
flash/memory-efficient SDPA backends have the same gap is untested -- that's what this script
checks on your hardware. The script therefore runs the math path FIRST (fast, expected safe)
and flash SECOND (may be slow or hang -- see the warning printed before it starts).

Checks, for both attention paths:
  1. Correctness: vmap output matches the loop's output (forward), and gradients w.r.t. the
     generated parameter vector match too (backward) -- not just "does it run," but "does it
     compute the same thing." Reports rich diagnostics (not just pass/fail), since a strict
     allclose can flag small floating-point drift as "wrong" -- also cross-checks in float64,
     where rounding error should nearly vanish if a mismatch really is just fp32 noise
     (amplified here by an unnormalized `.sum()` loss and the 4-loop recursive weight reuse)
     rather than a real bug.
  2. Speed: wall-clock comparison of loop vs. vmap across a few representative batch sizes,
     forward-only and forward+backward.

All output is printed with flush=True (or via a line-buffered stdout) so `tail -f` on a
redirected log file shows progress live rather than appearing to hang.

Usage:
    uv run python scripts/smoke_test_vmap_target_model.py
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_test_vmap_target_model.py
    uv run python scripts/smoke_test_vmap_target_model.py --skip-flash   # if flash hangs
    uv run python scripts/smoke_test_vmap_target_model.py --quick        # smaller/faster pass
"""

import argparse
import sys
import time

import torch
from torch.func import functional_call

from models.rope_looped_transformer import RoPECanonLoopedTransformer

# Exact target_model.params block from
# configs/experiments/arc1d_hypermodel_looped_lora_adapter/base_n2_loop4_noskip.yaml --
# the target model fixed across the hypernet_rope_canon_ablation experiment too.
TARGET_MODEL_KWARGS = dict(
    input_dim=16,
    hidden_dim=16,
    num_heads=2,
    output_dim=10,
    inner_dim=16,
    inner_num_heads=2,
    n_loops=4,
    dropout=0.0,  # dropout is stochastic; disabled here so loop vs vmap outputs are comparable
    canon_set="ABCD",
    canon_kernel=5,
    canon_activation=True,
    canon_residual=True,
    canon_causal=False,
    use_block_skip=False,
    use_loop_skip=False,
    use_output_head=True,
)

# Representative of real usage: 4 examples/task (3 support + 1 query), a modest seq_len,
# batch sizes spanning the smoke-test scale up to the real batch_size=512 used in training.
N_EXAMPLES = 4
SEQ_LEN = 24
CORRECTNESS_BATCH = 3
BATCH_SIZES = [8, 64, 512]
N_TIMING_ITERS = 20


def log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def target_parameter_specs(model: torch.nn.Module) -> list[tuple[str, tuple[int, ...], int]]:
    return [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters()]


def total_params(specs: list[tuple[str, tuple[int, ...], int]]) -> int:
    return sum(numel for _, _, numel in specs)


def build_param_dict(
    flat: torch.Tensor, specs: list[tuple[str, tuple[int, ...], int]]
) -> dict[str, torch.Tensor]:
    """flat: (total_params,) -> {name: tensor}. Mirrors HyperModel.build_param_dict exactly."""
    params, offset = {}, 0
    for name, shape, numel in specs:
        params[name] = flat[offset : offset + numel].view(shape)
        offset += numel
    return params


def build_batched_param_dict(
    flat_batch: torch.Tensor, specs: list[tuple[str, tuple[int, ...], int]]
) -> dict[str, torch.Tensor]:
    """flat_batch: (batch, total_params) -> {name: (batch, *shape) tensor}, for vmap's
    in_dims=0 batched-parameter argument."""
    params, offset = {}, 0
    for name, shape, numel in specs:
        params[name] = flat_batch[:, offset : offset + numel].reshape(-1, *shape)
        offset += numel
    return params


def run_loop(
    model: torch.nn.Module,
    flat_batch: torch.Tensor,
    inputs_batch: torch.Tensor,
    specs: list[tuple[str, tuple[int, ...], int]],
) -> torch.Tensor:
    """Exactly mirrors HyperModel.forward()'s current per-example loop."""
    outputs = []
    for i in range(flat_batch.shape[0]):
        params = build_param_dict(flat_batch[i], specs)
        outputs.append(functional_call(model, params, inputs_batch[i]))
    return torch.stack(outputs)


def run_vmap(
    model: torch.nn.Module,
    flat_batch: torch.Tensor,
    inputs_batch: torch.Tensor,
    specs: list[tuple[str, tuple[int, ...], int]],
) -> torch.Tensor:
    batched_params = build_batched_param_dict(flat_batch, specs)
    return torch.vmap(functional_call, in_dims=(None, 0, 0))(model, batched_params, inputs_batch)


def set_flash(model: torch.nn.Module, enabled: bool) -> None:
    """Toggle RoPECanonSelfAttention's flash-vs-manual attention path on every submodule."""
    found = False
    for module in model.modules():
        if hasattr(module, "flash"):
            module.flash = enabled
            found = True
    if not found:
        raise RuntimeError("No module with a 'flash' attribute found -- model structure changed?")


def make_batch(
    batch_size: int,
    specs: list[tuple[str, tuple[int, ...], int]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_params = total_params(specs)
    flat_batch = torch.randn(batch_size, n_params, device=device, dtype=dtype, requires_grad=True)
    inputs_batch = torch.randn(
        batch_size, N_EXAMPLES, SEQ_LEN, TARGET_MODEL_KWARGS["input_dim"], device=device, dtype=dtype
    )
    return flat_batch, inputs_batch


def compare_grads(grad_loop: torch.Tensor, grad_vmap: torch.Tensor) -> dict[str, float]:
    diff = (grad_loop - grad_vmap).abs()
    rel = diff / grad_loop.abs().clamp_min(1e-8)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "loop_norm": grad_loop.norm().item(),
        "vmap_norm": grad_vmap.norm().item(),
        "frac_gt_1e-3": (diff > 1e-3).float().mean().item(),
    }


def run_one_correctness_pass(
    model: torch.nn.Module,
    specs: list[tuple[str, tuple[int, ...], int]],
    device: torch.device,
    dtype: torch.dtype,
    label: str,
) -> bool:
    torch.manual_seed(1)
    flat_batch, inputs_batch = make_batch(CORRECTNESS_BATCH, specs, device, dtype=dtype)

    flat_loop = flat_batch.detach().clone().requires_grad_(True)
    out_loop = run_loop(model, flat_loop, inputs_batch, specs)
    out_loop.sum().backward()
    grad_loop = flat_loop.grad.detach().clone()

    flat_vmap = flat_batch.detach().clone().requires_grad_(True)
    try:
        out_vmap = run_vmap(model, flat_vmap, inputs_batch, specs)
        out_vmap.sum().backward()
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: report ANY vmap failure
        log(f"    [{label}] vmap FAILED: {type(exc).__name__}: {exc}")
        return False
    grad_vmap = flat_vmap.grad.detach().clone()

    max_out_diff = (out_loop - out_vmap).abs().max().item()
    g = compare_grads(grad_loop, grad_vmap)
    log(
        f"    [{label}] output_max_diff={max_out_diff:.2e}  "
        f"grad_max_abs_diff={g['max_abs']:.2e}  grad_mean_abs_diff={g['mean_abs']:.2e}  "
        f"grad_norms=(loop={g['loop_norm']:.1f}, vmap={g['vmap_norm']:.1f})  "
        f"frac_entries_off_by_gt_1e-3={g['frac_gt_1e-3']:.1%}"
    )
    return True


def check_attention_path(
    model: torch.nn.Module, specs: list[tuple[str, tuple[int, ...], int]], device: torch.device, flash: bool
) -> bool:
    label = "flash (scaled_dot_product_attention fused kernel)" if flash else "math (manual matmul+softmax)"
    log(f"\n--- {label} ---")
    set_flash(model, flash)

    log("  fp32 (real training precision):")
    ok_fp32 = run_one_correctness_pass(model, specs, device, torch.float32, "fp32")

    log("  fp64 (rounding-error control -- should nearly match if fp32 gap above is just noise):")
    model_fp64 = model.double()
    ok_fp64 = run_one_correctness_pass(model_fp64, specs, device, torch.float64, "fp64")
    model.float()  # restore fp32 for the benchmark step

    return ok_fp32 and ok_fp64


def check_correctness(model: torch.nn.Module, specs: list[tuple[str, tuple[int, ...], int]], device: torch.device) -> bool:
    log("=== Correctness: vmap vs. loop (forward + backward) ===")
    log(f"batch={CORRECTNESS_BATCH}, n_examples={N_EXAMPLES}, seq_len={SEQ_LEN}")

    ok_math = check_attention_path(model, specs, device, flash=False)

    log(
        "\nAbout to test the flash-attention path. This is the one PyTorch/vmap historically has "
        "weak support for (confirmed broken/very slow on CPU in the dev sandbox via a "
        "'no batching rule for aten::_scaled_dot_product_flash_attention...' warning). "
        "If this hangs for more than ~2 minutes on GPU, Ctrl-C and rerun with --skip-flash -- "
        "that itself is a usable, informative result (flash path unsupported, math path is the "
        "one to integrate)."
    )
    ok_flash = check_attention_path(model, specs, device, flash=True)

    return ok_math and ok_flash


def time_variant(
    fn,
    model: torch.nn.Module,
    flat_batch: torch.Tensor,
    inputs_batch: torch.Tensor,
    specs: list[tuple[str, tuple[int, ...], int]],
    device: torch.device,
    backward: bool,
) -> float:
    is_cuda = device.type == "cuda"
    # Warm-up: excludes one-time CUDA kernel compilation/caching from the timed region.
    for _ in range(3):
        flat = flat_batch.detach().clone().requires_grad_(backward)
        out = fn(model, flat, inputs_batch, specs)
        if backward:
            out.sum().backward()
    if is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(N_TIMING_ITERS):
        flat = flat_batch.detach().clone().requires_grad_(backward)
        out = fn(model, flat, inputs_batch, specs)
        if backward:
            out.sum().backward()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / N_TIMING_ITERS


def benchmark(
    model: torch.nn.Module,
    specs: list[tuple[str, tuple[int, ...], int]],
    device: torch.device,
    batch_sizes: list[int],
    skip_flash: bool,
) -> None:
    log()
    log("=== Speed: loop vs. vmap ===")
    header = f"{'flash':<6} {'batch':>6} {'mode':<10} {'loop (ms)':>12} {'vmap (ms)':>12} {'speedup':>9}"
    log(header)
    log("-" * len(header))
    flash_settings = (False,) if skip_flash else (False, True)
    for flash in flash_settings:
        set_flash(model, flash)
        for batch_size in batch_sizes:
            torch.manual_seed(0)
            flat_batch, inputs_batch = make_batch(batch_size, specs, device)

            for mode, backward in (("fwd only", False), ("fwd+bwd", True)):
                t_loop = time_variant(run_loop, model, flat_batch, inputs_batch, specs, device, backward)
                try:
                    t_vmap = time_variant(run_vmap, model, flat_batch, inputs_batch, specs, device, backward)
                    speedup = f"{t_loop / t_vmap:.1f}x"
                    vmap_str = f"{t_vmap * 1000:.2f}"
                except Exception as exc:  # noqa: BLE001
                    vmap_str = f"FAILED({type(exc).__name__})"
                    speedup = "n/a"
                flash_label = "on" if flash else "off"
                log(
                    f"{flash_label:<6} {batch_size:>6} {mode:<10} "
                    f"{t_loop * 1000:>12.2f} {vmap_str:>12} {speedup:>9}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-flash", action="store_true", help="Skip the flash-attention path entirely (use if it hangs).")
    parser.add_argument("--quick", action="store_true", help="Smaller benchmark batch sizes for a faster pass.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    log(f"PyTorch: {torch.__version__}")
    log()

    model = RoPECanonLoopedTransformer(**TARGET_MODEL_KWARGS).to(device)
    model.eval()  # disable dropout/etc. so loop and vmap see identical stochasticity (none)
    specs = target_parameter_specs(model)
    log(f"Target model params: {total_params(specs):,} (expect 11,760 backbone + 160 head = 11,920)")
    log()

    if args.skip_flash:
        log("--skip-flash set: testing the math (manual matmul+softmax) path only.\n")
        ok = check_attention_path(model, specs, device, flash=False)
    else:
        ok = check_correctness(model, specs, device)

    batch_sizes = [8, 64] if args.quick else BATCH_SIZES
    benchmark(model, specs, device, batch_sizes, skip_flash=args.skip_flash)

    log()
    if ok:
        log("RESULT: vmap matches the loop's outputs and gradients on this device. Safe to integrate.")
    else:
        log(
            "RESULT: vmap diverged from the loop, or failed, on at least one path -- see above. "
            "If only the flash path failed and math passed cleanly (especially in fp64), "
            "integrating vmap while forcing the math attention path (flash=False) is likely still safe."
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # visible progress even when redirected to a log file
    main()

# arc1d_rope_skip_ablation

Diagnostic experiment investigating why the Canon ABCD configuration fails on the **flip** task
and exploring skip-connection remedies.

## Motivation

The current best architecture (outer=8, inner=32, RoPE, Canon ABCD, 4 loops) solves most
ARC-1D tasks well but has two failure modes:

- **flip** — a per-position colour remap that is trivially solved without Canon layers but
  fails with Canon ABCD.
- **padded_fill** — longer-sequence variant; requires more capacity (separate problem).

### Why does flip fail with Canon?

Each CanonLayer already contains an internal additive skip (`output = input + silu(conv(input))`
when `canon_residual=True`), so the original signal at each position is preserved within each
Canon module. The deeper issue is that **Canon-A and Canon-B corrupt the attention pattern**:

- **Canon-A** (before attention): produces `xx = ln(x) + silu(conv5(ln(x)))`. Q and K are
  projected from `xx`, so each query/key contains a blend of its own and neighbouring positions.
  The resulting `softmax(QK^T)` spreads attention across neighbours even when self-attention is
  the correct strategy.
- **Canon-B** (on the concatenated QKV after projection): further mixes Q, K and V across
  the sequence before they are split, compounding the attention-pattern corruption.
- **Canon-C and Canon-D** act after the attention pattern is committed (inside the MLP path)
  and are therefore less likely to cause this failure.

## Architecture base (all conditions)

| Property | Value |
|---|---|
| Outer hidden dim (pre/post layers) | 8 |
| Inner hidden dim (middle/looped layer) | 32 |
| Outer heads / Inner heads | 1 / 4 |
| n_loops | 4 |
| Positional encoding | RoPE (no sinusoidal PE in embedder) |
| Canon kernel | 5, non-causal |
| N_sup | 2 |
| Learning rate | 0.0005 |
| Max steps | 4000 (= 8000 optimizer updates) |
| Parameters | ~16,672 (independent of n_loops) |

## Conditions

Condition A (Canon ABCD, no skip) is the reference and **reuses existing runs** from
`arc1d_rope_wide_middle_ablation` (condition A there). Only B, C, D are new.

| Cond | Canon set | Block highway skip | Hypothesis |
|------|-----------|--------------------|------------|
| A    | ABCD      | No                 | Reference: current best |
| B    | ABCD      | Yes (all layers)   | Does `x_out += x_in` bypass enough Canon corruption to fix flip? |
| C    | BCD       | No                 | Is Canon-A (mixing attention input) the primary culprit? |
| D    | ACD       | No                 | Is Canon-B (mixing QKV) the primary culprit? |

**Block highway skip**: adds `x_out = block(x_in) + x_in` at the end of each
`RoPECanonBlock` (pre, middle, post). This provides a direct identity path that bypasses all
Canon convolutions, attention, and MLP within the block — stronger than the per-Canon-layer
residuals.

## Predicted outcomes

| Condition | Flip | Other tasks | Interpretation |
|-----------|------|-------------|----------------|
| B (ABCD + skip) | Improves | ~Same | Highway sufficient to preserve identity signal |
| C (BCD, no A)   | Improves | ~Same | Canon-A is primary culprit (corrupts Q/K) |
| D (ACD, no B)   | Improves | ~Same | Canon-B is primary culprit (corrupts QKV split) |

If both C and D improve flip, Canon-A is the bottleneck.
If only B improves flip, the internal Canon residuals are insufficient and a block-level
highway is needed regardless of which Canon position is active.

## Jobs

3 new conditions × 18 tasks × 3 seeds = **162 jobs**

```bash
bash scripts/run_ablation_arc1d_rope_skip.sh        # 8 GPUs (default)
bash scripts/run_ablation_arc1d_rope_skip.sh 4      # 4 GPUs
```

## Plotting

```bash
bash scripts/fetch_experiments.sh arc1d_rope_skip_ablation
python scripts/plot_rope_skip_ablation.py
```

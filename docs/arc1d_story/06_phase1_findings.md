# 06 - Phase 1 findings: the minimal recipe

This consolidates everything Phase 1 (`docs/arc1d_story/05_round2_plan.md`'s
several "Update" sections, plus the published artifact) established, into
one clean summary. Phase 1's question was "what is the smallest model that
solves individual ARC-1D tasks, trained directly, one model per task
category, no cross-task sharing." All of it ran under **Muon from the very
first job** - no AdamW/RAdam meander this round, unlike the historical
chain.

## The headline result

**A ~5,000-parameter RoPE+Canon transformer, trained with a single forward
pass (no extra supervision loop) and no recursion, solves 14 of the 15
ARC-1D task categories** (`1d_recolor_cmp` excluded - see below) at
essentially ceiling accuracy (0.986, with several configurations reaching
1.000).

The story is deliberately presented with **RoPE as part of the fixed
baseline**, not as something being ablated - RoPE stays in the architecture
going forward regardless of what Phase 1 found, since every downstream
Round 2 phase already commits to it. Canon is the one addition Phase 1
actually needed to discover:

| Step | Architecture | 14-task mean |
|---|---|---:|
| **Baseline (R2)** | RoPE only, flat (n_loops=1), single pass (N_sup=1), sin-PE off | 0.881 |
| **+ Canon (RC1)** | RoPE + Canon, still flat, still single pass | **0.986** |
| + N_sup=2 (T4) | same architecture, extra supervision loop | 0.991 |
| + looped, n_loops=4 (T5) | same, now recurrent | 0.986 |
| + n_loops=8/16/32 (L8/L16/L32) | more recursive iterations | 0.995 / 0.991 / 1.000 |
| + skip connections (B4/B8/P4/P8/S4/S8) | block-only, loop-only, or both, at n_loops in {4,8} | 0.991 - 1.000 |

Everything after RC1 sits within 1-2 examples of RC1 itself (out of 42) -
noise, not a trend. **Canon is the whole story.**

## What was ruled out, and how

- **Does model width matter beyond ~5k params?** dim=16 (11,760 params) was
  tried first and turned out uninformative - already flat at ~0.92-0.93 from
  T1 onward, no separation between steps. dim=10 (5,130 params, the closest
  even-head_dim fit to ~5k) is what actually shows the progression above.
- **Does step order matter?** An ordering ablation tested "RoPE before
  N_sup", "RoPE before Canon", and "RoPE, then looped, then N_sup, Canon
  last". None beat the reference order; "Canon last" was the worst option,
  staying stuck at ~0.83-0.84 for three steps until Canon finally landed.
- **Does more training-time supervision help (N_sup=4)?** Tested across the
  full T2-T5 sequence. A small bump at the earliest step (+0.01), gone by
  the time Canon is present. Not a lever.
- **Does looping help at all** - not just "how many loops," but whether
  looping happens **at all**? T4 (flat, n_loops=1) was compared directly
  against T5 (looped, n_loops=4) and the full L8/L16/L32 sweep. Flat matches
  or beats every looped variant tested, through n_loops=32. Recursion adds
  nothing once Canon is present, at this scale.
- **Do skip connections help, isolated from loop count?** A full factorial
  (block-only, loop-only, both, at n_loops in {4,8}) found no combination
  beats its own no-skip reference.
- **Does RoPE itself matter, independent of Canon?** The full RoPE x Canon x
  N_sup factorial (`T1, T2, R2, R3, C1, RC1, T3, T4`) found Canon-only
  matches or beats RoPE+Canon at both N_sup levels (0.991/0.995 vs
  0.986/0991) - RoPE adds nothing measurable once Canon is present either.
  It stays in the architecture by decision (matching downstream phases),
  not because Phase 1's evidence requires it.

## `1d_recolor_cmp`: investigated, not ignored

Every configuration above left this one task stuck at <=0.2 exact match.
Rather than leave it as an unexplained outlier, the actual rule was decoded
from real data and checked against 300 random instances at 300/300 correct:
**recolor the longest contiguous run(s) of the active colour to a target
colour, ties included.** The task is fully well-posed - not a data problem.

But the target colour is not a fixed constant: checked across 2000
instances, it lands on all 9 possible colours at roughly equal frequency,
with zero signal outside that specific instance's own 3 support examples.
Solving it requires binding a freshly-inferred, unconstrained value (1 of 9
colours) to an abstractly-computed target (the longest run, ties included) -
fresh, per instance. Every other task in the 15-task set only needs a fixed
local transformation or a colour that's already present in the input; none
require this kind of in-context value binding.

Phase 1 trains one *separate* model per task category - no cross-task
sharing - so a single individually-trained model has to encode this
"read context, infer an arbitrary value, apply it based on a global
computation" behaviour as a general algorithm baked into one fixed, tiny
weight set. It failed to across all 14 configurations tried here (scale,
depth, skip connections, ordering, supervision depth).

**Decision**: `1d_recolor_cmp` is excluded from the working set for
individually-trained/direct experiments (**14 tasks**, not 15) going
forward. It stays in the 15-task set from Phase 2 onward - the joint
hypernetwork's separate, larger encoder generating per-instance target
weights is a plausible (untested) mechanism for this specific kind of
in-context binding, unlike a single small fixed-weight model. Whether that
mechanism actually works is an open question for later, not assumed here.

## Where this leaves the architecture question

The minimal *sufficient* recipe (RC1: RoPE+Canon, flat, single pass) is
5,130 params at dim=10. Whether that's the minimal *necessary* recipe -
whether something smaller also reaches ceiling - was not tested in Phase 1
proper; dim=10 was chosen as the closest fit to the historical
`arc1d_capacity_*` family's ~5k scale, not derived from a fresh sweep below
it. That's the subject of the next round of experiments (below).

## Next: three follow-up experiments

Documented in full, with exact configs, once the codebase investigation
below is complete:

1. **Minimal model size** - does something smaller than ~5k params (3k,
   ~1k, fewer blocks) still reach ceiling on the 14-task set?
2. **Multi-task capacity** - can one small model (at whatever size Q1
   lands on) learn all 14 tasks *jointly*, with and without a per-task
   identity signal - or does it need the hypernetwork's cross-task
   weight-generation mechanism to do that?
3. **Minimal data** - how few augmented variants per base task (0, 1, 2, 5,
   10, ... up to the standard ~200) does the minimal model need to still
   reach ceiling, and does that number shrink further once a hypernetwork
   is involved?

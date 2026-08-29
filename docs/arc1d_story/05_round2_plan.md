# 05 - Round 2 plan: from-scratch unified rerun

## Context

The stocktake (`00_overview.md` through `04_checklist.md`) inventoried the
existing ARC-1D hypernetwork research chain and found the foundational
capacity-ablation result predates a batch-count bug fix and was never
re-run, plus several architecture choices drift inconsistently across the
historical chain. This document is the plan for a from-scratch rerun under
one consistent recipe, designed but not yet executed. It went through three
revisions before landing here: the first pass proposed a small
architecture-search grid to resolve `n_loops`/`lora_adapter_rank` before
anything else; this version instead **fixes `n_loops` and `num_layers` by
historical precedent rather than re-searching them**, which removes an
entire phase and substantially reshapes the dependency graph.

Deciding which specific jobs to start first (given current cluster
availability) is a separate follow-up step, not fixed in this document. The
soft priority order at the end of the "Dependency graph" section is a
starting point for that conversation, not a final decision.

## Decisions locked this round

1. **Muon from the start** - every new experiment uses Muon.
2. **`n_loops=4`, no skip - fixed by precedent, not re-searched.** Confirmed
   via `02_architecture.md` (lines 93-94 as of this writing): nearly the
   entire downstream hypernetwork chain (`mechanism_ablation`,
   `capacity_baseline`, `followup`, `grid`, `optimizer`, `zhu_block`,
   `lr_sweep`, `muon`, `muon_sweep`, `moveablation`, `vae_disentanglement`,
   `compositional_generalization`, `generalization`, `lowdata`,
   `lowdata_baseline`) uses `n_loops=4` with no skip. Only 2 configs ever
   used `n_loops=8` (`mix11_td`, `lora_adapter/base.yaml`), and both sourced
   that choice from a narrower, different sweep (the recolor capacity
   study), not from the ancestor capacity-ablation's own L1-L6
   recommendation (which actually favoured n_loops=8, but was never adopted
   downstream). This is genuinely the lowest `n_loops` value used
   successfully anywhere in the hypernetwork chain - the only lower option,
   `n_loops=1` (flat, non-looped), was tested solely in the ancestor
   direct-training study (91.5% test at dim=16, noticeably below the looped
   variants) and never adopted by any hypernetwork experiment. Fixing at 4
   is both the best-precedented and the cheapest choice.
3. **`num_layers=8` (hypernetwork encoder) - fixed by precedent.** Confirmed
   via `02_architecture.md` (lines 184-188): traces to `capacity_baseline`'s
   own finding (mean/std improvement over `num_layers=4`) and is what
   everything from `capacity_baseline` onward actually uses.
4. **Attention pooling only** - already the universal downstream choice, no
   ablation needed (`02_architecture.md` confirmed no inconsistency here
   either).
5. **Target-model architecture family (transformer vs. RNN vs. MLP) is not
   ablated this round** - stays fixed as `RoPECanonLoopedTransformer`. That
   question, if it's ever asked, is a separate future round, not this one.
6. **Rank is a cut-down ablation** - establish the story at full weight
   generation (no LoRA restriction) first, then a dedicated phase sweeps
   rank down from full through {8, 4, 2, 1}.
7. **Data-augmentation count is a cut-down ablation**, mirroring rank.
8. **A dedicated `frozen_td`-vs-`td`-vs-`notd` experiment**, more thorough
   than the historical 2-seed, sibling-only-categories study.
9. **Flat 3 seeds everywhere**, no exceptions, plus a documented contingency
   for topping up any specific cell that reproduces the historical
   `td`-bimodality pattern.

What removing the architecture-search phase changes: an earlier revision's
"Phase 2" (a small `n_loops` x `num_layers` grid to pick winners) no longer
exists - those values are fixed by precedent instead. This collapses most
of the plan's dependency chain, addressed directly below.

## On the dependency graph

An earlier revision's graph gated almost everything behind a "lock the
architecture" phase. With `n_loops` and `num_layers` now fixed by precedent
rather than searched, that gate mostly disappears. The only remaining true
prerequisite is Phase 0.5 (confirm Muon actually trains correctly in both
the direct and hypernetwork contexts, since neither combination this round
needs - direct+Muon, hypernetwork+Muon+`batch_size=2048` together - has
been independently validated before). Once that's cleared:

- **Phase 1** (backbone progression demo) doesn't gate anything downstream
  any more - it no longer searches for `n_loops`, it just demonstrates the
  flat-vs-looped, Canon, RoPE, skip progression at the already-fixed
  `n_loops=4`. It can run in parallel with everything else.
- **Phase 2 (headline), Phase 3 (descriptor study), Phase 4 (compositional),
  Phase 5 (low-data cuts), Phase 6 (rank cuts)** all depend only on the
  fixed architecture and Phase 0.5's validated Muon settings, not on each
  other. There is no computational reason left to sequence Phase 5/6 after
  Phase 2, since Phase 2's number isn't an input to launching Phase 5/6's
  training jobs, only to the write-up that compares them.

So: low-data and low-rank should not wait. They can be built and launched
as soon as Phase 0.5 clears, in parallel with the headline and descriptor
work, not after it. The only remaining "do this last" logic is soft, not a
dependency: if GPU capacity is constrained, prioritise the smaller, more
central phases (headline: 3 jobs, descriptor: 90 jobs, compositional: 6
jobs) before committing the two large ablation sweeps (low-data: 153 jobs,
rank: 12 jobs) - a scheduling preference, not a blocking requirement.

## Consistency check against the stocktake

Every open item from `02_architecture.md`'s "Open questions" and
`04_checklist.md`'s table, dispositioned:

| Source | Open item | Disposition |
|---|---|---|
| `02_architecture.md` | `n_loops` (4 vs 8) | Fixed at 4 by precedent (see "Decisions locked" above), not re-searched. Phase 1 still empirically shows flat (n_loops=1, T4) vs looped (n_loops=4, T5) as part of the progression narrative. |
| `02_architecture.md` | `lora_adapter_rank` (4 vs 8) | Full-generation first (Phases 1-5), dedicated cut-down ablation in Phase 6. |
| `02_architecture.md` | Pooling (attention vs hierarchical) | No inconsistency found in the stocktake - attention pooling confirmed and kept, no ablation needed. |
| `02_architecture.md` | Optimizer choice | Resolved: Muon everywhere, per Fabio's decision. |
| `02_architecture.md` | `hyper_model.num_layers` (4 vs 8) | Fixed at 8 by precedent (see "Decisions locked" above), not re-searched. |
| `04_checklist.md` #1 | Re-run `arc1d_uniform_ablation` | Phase 1 (now narrative-only: T1-T7 progression at fixed n_loops=4, dim=16, 15 tasks, 3 seeds, no L1-L6 sweep, since n_loops isn't being searched this round). |
| `04_checklist.md` #2 | Re-run/replace `arc1d_hypermodel_looped`, `_recolor`, `_mix11` | Not separately rerun as their own phases - superseded by Phase 2's 15-task joint headline. `_recolor`'s hard-task question isn't pre-built as its own sweep; if Phase 2's per-task breakdown shows `1d_flip`/`1d_recolor_cmp` still weak, that's a trigger for a small targeted follow-up, not a committed job count. |
| `04_checklist.md` #3 | Run `arc1d_lowdata_baseline` (found already done, but on the old architecture) | Phase 5 reruns it anyway on the Round 2 fixed architecture. |
| `04_checklist.md` #4 | Verify wandb / `muon_sweep`/`muon_moveablation` high failure rates | Wandb check closed in the stocktake. The failure-rate risk is what Phase 0.5 exists to de-risk before scaling. |
| `04_checklist.md` #5 | Resolve `n_loops` | Fixed by precedent, not re-searched (see above). |
| `04_checklist.md` #6 | Resolve `lora_adapter_rank` | Full-first, cut-down ablation, Phase 6. |
| `04_checklist.md` #7 | Design one unified rerun recipe | This document. |
| `04_checklist.md` #8 | `arc1d_hypermodel_disentanglement` data-recipe mismatch | Closed already, documentation-only, no Round 2 action. |
| `04_checklist.md` #9 | Beta-VAE disentanglement sweep | Closed already, excluded by design. Phase 3a's cluster-map + linear-probe diagnostic is this round's disentanglement evidence. |
| `04_checklist.md` #10 | Stray `outputs_diag_seed1_*.jsonl` files | Unrelated housekeeping, not part of Round 2. |
| `04_checklist.md` #11 | `arc1d_uniform_ablation` L5/L6 missing local run directories | Moot, and doubly so now, since Round 2's Phase 1 doesn't run an L-series sweep at all. Still: every Round 2 run script must retain local `results.txt` files, learning from this gap. |
| `04_checklist.md` #12 | `rank_sweep`/`lr_sweep` partial local coverage | `rank_sweep` (RAdam-era) superseded by Phase 6's Muon-era rank ablation. `lr_sweep` (AdamW-era) superseded by the switch to Muon; Phase 0.5's `muon_lr` check is this round's equivalent. |
| `03_experiments.md` (`arc1d_lowdata_baseline`) | 3-vs-4-supervised-pairs asymmetry (baseline supervises 3 support pairs, hypernetwork side supervises 4 incl. query) | Carried forward as-is, matching historical convention and the stocktake's own assessment that it doesn't explain a large gap. Revisit only if Phase 5's Round 2 comparison hinges on a close call. |

## Locked reference values

**Target model** (`rope_canon_looped_transformer`): `hidden_dim=16,
num_heads=2, inner_dim=16, inner_num_heads=2, dropout=0.1, n_loops=4,
canon_set=ABCD, canon_kernel=5, canon_activation=true, canon_residual=true,
canon_causal=false, use_block_skip=false, use_loop_skip=false`. Fully
fixed, no open axis.

**Hypernetwork encoder** (`rope_canon_zhu_transformer`): `hidden_dim=64,
num_heads=4, num_layers=8, output_dim=64`, Canon ABCD, RMSNorm + QK-norm +
SwiGLU, attention pooling. Fully fixed, no open axis.

**Hyper head, Phases 1-5**: `pooling: attention, bottleneck_dim: 128`, full
weight generation, no LoRA adapter. `lora_adapter_rank` doesn't apply until
Phase 6.

**Task descriptor**: `frozen_td` = `num_tasks=18, freeze_task_indicator=true`
(default for Phases 2, 4, 5, 6). `td`/`notd` only appear in Phase 3.

**Optimizer (Muon everywhere)**: `optimizer=Muon, muon_momentum=0.95`,
AdamW aux group `learning_rate=0.001, weight_decay=0.01,
gradient_clip_val=10.0`. `muon_exclude_lora_heads` only matters once Phase
6 introduces LoRA heads. `muon_lr`/`batch_size` are not yet independently
validated in several of this round's exact contexts - Phase 0.5 checks
both first.

**`N_supervision`/`max_steps`**: `N_supervision=2` throughout, one
total-update budget per phase - 8000 total updates (`max_steps=4000`) as
the hypernetwork-phase default (already demonstrated sufficient
historically). Do not silently double it for `notd`/`td` in Phase 3 to
"even the odds."

## Phase 0 - Data setup (no jobs)

Verify `data/arc_1d_looped_augmented` on disk matches `01_data.md` section
1's spec (721,000 train rows, 40,000 per category except
`1d_scale_dp`=41,000; dev/test 100/category). No rebuild needed. Document
as "frozen for Round 2."

## Phase 0.5 - Infra pilot (the one true prerequisite)

**Directory**: `configs/experiments/arc1d_v2_pilot/`. Direct configs from
`configs/experiments/arc1d_recursion_ablation/base_ablation.yaml`;
hypernetwork configs from
`configs/experiments/arc1d_hypermodel_looped_rope_canon_muon/base.yaml`
(`lora_adapter: false`). 5 hand-written single-seed cells, run sequentially
with `time` around each `train.py` call, at the now-fixed
`n_loops=4`/`num_layers=8`:

| Config | What it checks |
|---|---|
| `direct_muon_lr005_nloops4` | Direct+Muon at the only LR ever used for this combo (copied, not validated, into `arc1d_lowdata_baseline`) |
| `direct_muon_lr02_nloops4` | Does the much smaller direct model tolerate Muon's un-lowered default LR that diverged the 1.58M-param hypernetwork? |
| `hyper_muon_lr005_bsz2048_frozentd_full` | Full-generation mode at this (lr, batch_size) pairing, never run before |
| `hyper_muon_lr005_bsz2048_notd_full` | Same, on the historically harder-to-stabilise `notd` arm |
| `hyper_muon_lr02_bsz2048_frozentd_full` | Single-arm retest of `muon_moveablation`'s risky LR (17/42, 40% finish rate historically) at n=1, before ruling 0.02 in or out |

**Jobs**: 5.

**Output feeding forward**: confirmed `muon_lr` for both paths, real
wall-clock, a go/no-go check before anything scales up. This is the only
phase every other phase actually depends on.

**Dependency**: needs Phase 0's data verification.

## Phase 1 - Backbone progression demo (direct training, Muon, n_loops fixed)

**Directory**: `configs/experiments/arc1d_v2_backbone_capacity/`. Base:
`configs/experiments/arc1d_recursion_ablation/base_ablation.yaml`, with
`optimizer: Muon` plus Phase 0.5's validated `muon_lr`/`muon_momentum`
replacing RAdam. Gen/run pattern: copy `scripts/gen_uniform_ablation_configs.py`
and `scripts/run_ablation_arc1d_uniform_ablation.sh`, trimmed to T1-T7 only
(no L-series - nothing left to search for). Local `results.txt` retention
is mandatory (learning from the historical L5/L6 gap).

**Scope**: dim=16 only, 15-task working set, 3 seeds, T1-T7 only (T5-T7
fixed at `n_loops=4`, matching the round-wide decision; T4 still shows the
flat/n_loops=1 comparison point as part of the progression).

**Jobs**: 15 tasks x 3 seeds x 7 cells = 315.

**Dependency**: needs Phase 0.5's validated `muon_lr`. Does not gate any
other phase - purely a narrative/confirmatory result now.

## Update: Phase 1 actual execution, and the `1d_recolor_cmp` scope decision

Phase 1 as actually run diverged from this original design in ways worth
recording here rather than editing the plan above out of place:

- T6/T7 (skip variants) were dropped entirely, not just deferred - n_loops=4
  no-skip is fixed by precedent for the whole round, so testing skip
  connections against a question already settled elsewhere added nothing.
- dim=16's T1-T5 turned out uninformative (14/15 tasks already ~1.0 from T1
  onward under Muon) - dim=10 (~5,130 params, num_heads=1, the closest
  even-head_dim fit to the older `arc1d_capacity_*` family's ~5k scale) was
  added and is what actually shows a progression.
- The adopted step order changed from T1-T2-T3(Canon)-T4(RoPE)-T5(looped) to
  T1-T2-R3(RoPE, before Canon)-T4(Canon)-T5(looped) - an ordering ablation
  (testing "RoPE before N_sup", "RoPE before Canon", and "RoPE, looped, N_sup,
  Canon last") found reordering never beats the original, and the "Canon
  last" variant is the worst option tested, staying stuck until Canon
  finally lands. RoPE-before-Canon was kept as the carried-forward order
  since it's the more natural narrative (vanilla, more supervision, better
  positional encoding, then the two "real" architecture additions), not
  because it scored better - it doesn't, the two orderings are
  statistically identical wherever they overlap.
- A loop-count/skip diagnostic was added on top of the adopted order's T5
  (L8/L16/L32 - more loop iterations, no skip; B4/B8/P4/P8/S4/S8 - full
  skip factorial at n_loops in {4,8}). Finding: **looping does not help at
  this scale.** T4 (flat, n_loops=1) already matches or beats every looped
  variant tested, all the way through n_loops=32. No skip combination beats
  its own no-skip reference either. Canon is the entire story; nothing else
  in Phase 1's toolkit (reordering, more supervision, more loops, skip
  connections) moves performance further once Canon is present.

**`1d_recolor_cmp` scope decision.** Every one of the ~14 configurations
above (T1-T5, R2/R3/L3/L4, T2n4-T5n4, L8/L16/L32, B4/B8/P4/P8/S4/S8) left
`1d_recolor_cmp` stuck at <=0.2 exact match. Investigated directly rather
than left as an open mystery: the task's actual rule was decoded and
verified against 300 random real instances at 300/300 correct - **recolor
the longest contiguous run(s) of the active colour to a target colour,
ties included, everything else unchanged.** The task is fully well-posed,
not a data problem. But the target colour is not a fixed constant - checked
across 2000 instances, it lands on all 9 possible colours at roughly equal
frequency, with zero signal outside that specific instance's own 3 support
examples. Solving it therefore requires two things bound together, fresh
per instance: identifying the longest run (a global comparison across
however many runs are present) and inferring which of 9 colours to apply
(pure in-context induction, nothing to memorise).

Phase 1 trains one *separate* model per task category (`task_categories:
[task]` in every generated config) - there is no cross-task sharing, so a
single individually-trained model must encode this "read context, extract
an arbitrary value, apply it based on a global computation" behaviour as a
*general, fixed-weight algorithm* discovered by SGD over ~40,000 augmented
instances, all sharing the same tiny (5k-12k parameter) recursive backbone.
That's a categorically different demand from the other 14 tasks, none of
which need to bind a freshly-inferred, unconstrained value to an
abstractly-computed target. It has now failed to learn across every lever
Phase 1 has: scale (5,130-42,464 params), depth (n_loops 1-32), skip
connections, training order, and supervision depth (N_sup 2 and 4) - 14
independent attempts, all stuck.

**Decision**: exclude `1d_recolor_cmp` from Phase 1's working set going
forward (**14 tasks**, not 15, for any further individually-trained/direct
experiments) - re-running the other 14 tasks' already-collected results is
not needed, since each task trains as a fully independent model and none of
their results depend on whether `1d_recolor_cmp` is in the set. **Re-include
it starting at Phase 2** (the joint multi-task hypernetwork headline claim)
and every phase after - the hypernetwork setting is architecturally
different in exactly the way this task needs: a separate, larger encoder
(the Zhu-block hypernetwork, ~1.5M+ params) reads the support examples and
*generates* the target model's weights per-instance, rather than requiring
one fixed tiny weight set to implement a fully general in-context-binding
circuit. Whether that mechanism actually succeeds on this task is an open,
genuinely interesting question for Phase 2 to answer - not assumed here.

Recomputed Phase 1 means with `1d_recolor_cmp` excluded (n=42, vs n=45
including it) confirm the story is even cleaner than it looked with the
task included - `1d_recolor_cmp` alone was responsible for nearly all of
the earlier "not quite at ceiling" signal, all the way back to T1:

| Step | 14-task mean | 15-task mean (incl. `recolor_cmp`) |
|---|---:|---:|
| T1 | 0.886 | 0.827 |
| T2 | 0.905 | 0.844 |
| R3 | 0.905 | 0.849 |
| T4 | 0.991 | 0.933 |
| T5 | 0.986 | 0.929 |
| L8 / L16 / L32 | 0.995 / 0.991 / 1.000 | 0.929 / 0.933 / 0.938 |
| B4 / B8 / P4 / P8 / S4 / S8 | 0.991 / 1.000 / 0.995 / 0.995 / 0.991 / 0.995 | 0.924 / 0.933 / 0.933 / 0.929 / 0.924 / 0.933 |

From T4 onward, every 14-task configuration sits at 0.99-1.00 - the
remaining spread is 1-2 example flips out of 42, not a real effect.

## Phase 2 - Headline capability claim

**Directory**: `configs/experiments/arc1d_v2_headline/`. Fixed architecture
throughout (`n_loops=4, num_layers=8`, full generation), `frozen_td`, Muon,
15-task set. One config, seed-swept only.

**Jobs**: 1 config x 3 seeds = 3.

**Dependency**: only Phase 0.5.

## Phase 3 - Task descriptor mechanism study (frozen_td vs td vs notd)

**Directory**: `configs/experiments/arc1d_v2_descriptor_mechanism/`, two
subgrids.

### 3a - in-distribution parity and disentanglement diagnostics

Base: Phase 2's `base.yaml`, varying only `hyper_head.num_tasks`/
`freeze_task_indicator` across 3 arms (`notd`, `td`, `frozen_td`).
`log_embedding_clusters: true` throughout.

**Jobs**: 3 arms x 3 seeds = 9.

### 3b - held-out-category generalisation robustness

Base: same, existing leave-one-out mechanism
(`Arc1dMetaMulticlassDataModule` already supports differing train/eval
category lists). Gen/run: copy
`configs/experiments/arc1d_hypermodel_looped_rope_canon_generalization/`'s
layout and run script.

Held-out set, expanded from the historical 5 to 9, deliberately mixing
sibling and no-sibling cases:

| Category | Sibling remaining in training? |
|---|---|
| `1d_move_2p` | yes (`move_1p/3p/dp/2p_dp`) - historical |
| `1d_denoising_mc` | yes (`denoising_1c`) - historical, the one success case |
| `1d_flip` | weak (`mirror`) - historical |
| `1d_pcopy_mc` | yes (`pcopy_1c`) - historical |
| `1d_hollow` | weak (`fill`) - historical |
| `1d_scale_dp` | no sibling - new |
| `1d_recolor_cmp` | no sibling - new |
| `1d_move_dp` | weak/dynamic variant of the move family - new |
| `1d_mirror` | weak (`flip`) - new, symmetric complement to holding out `flip` |

**Jobs**: 9 categories x 3 variants x 3 seeds = 81.

**Honest limitation, flagged not hidden**: 3 seeds won't cleanly resolve a
genuinely bimodal outcome. If any cell reproduces `denoising_mc`/`td`'s
historical 0-then-1 split at n=3, treat that as a trigger to top up that
cell to 5 seeds, not as a result to over-interpret from 3 noisy points.

**Dependency**: only Phase 0.5.

## Phase 4 - Compositional generalisation

**Directory**: `configs/experiments/arc1d_v2_compositional/`. Base: Phase
2's `base.yaml`, `notd`/`frozen_td` only. Data reused as-is
(`data/arc_1d_compositional_holdout`). 2 hand-written leaf configs, each
followed by the unmodified `scripts/eval_compositional_holdout.py`.

**Jobs**: 2 arms x 3 seeds = 6.

**Dependency**: only Phase 0.5.

## Phase 5 - Low-data augmentation cuts

**Directories**: `configs/experiments/arc1d_v2_lowdata/` (hypernetwork)
plus `configs/experiments/arc1d_v2_lowdata_baseline/` (no-hypernetwork
control). Both at full generation, fixed architecture. No longer gated
behind an architecture-lock phase, see "On the dependency graph" above.

**Hypernetwork side**: base `configs/experiments/arc1d_lowdata/base.yaml`,
architecture params updated to the round's fixed `n_loops=4`/`num_layers=8`,
`lora_adapter: false`. Existing `variants_per_base_task`/`data_seed`
mechanism reused as-is. Levels `{1,2,3,4,5,20}`. **Jobs**: 6 x 3 = 18.

**No-hypernetwork baseline**: base
`configs/experiments/arc1d_lowdata_baseline/base.yaml`, `n_loops=4`
(already what it used historically, no change needed there). Reuse
`scripts/gen_lowdata_baseline_configs.py` plus
`scripts/run_lowdata_baseline.sh` directly. Levels `{1,2,3}`. **Jobs**: 15
x 3 x 3 = 135.

**Phase 5 total**: 153.

**Dependency**: only Phase 0.5.

## Phase 6 - LoRA-adapter rank ablation

**Directory**: `configs/experiments/arc1d_v2_rank_ablation/`. Base: Phase
2's `base.yaml` (full generation, `frozen_td`, fixed architecture), with
`lora_adapter: true` and `lora_adapter_rank` swept,
`muon_exclude_lora_heads=true` (now applicable). Standard full data recipe
throughout, isolating rank only.

| Level | Meaning |
|---|---|
| `full` | Reference point - Phase 2's actual result, not rerun here, just cited |
| `r8` | Historical rank-sweep's RAdam-era winner, re-tested under Muon |
| `r4` | The rank actually used (inconsistently) by `vae_disentanglement`/`compositional_generalization` historically |
| `r2` | |
| `r1` | Lower bound |

**Jobs**: 4 new rank levels x 3 seeds = 12.

**Dependency**: only Phase 0.5 (cites Phase 2's number for the write-up
comparison, but doesn't block training on it).

## Dependency graph

```
Phase 0 (data verify) ---> Phase 0.5 (infra pilot: muon_lr, wall-clock -- the one real gate)
                                |
        +---------------+------+------+---------------+---------------+
        v               v             v               v               v
  Phase 1 (progression) Phase 2 (headline) Phase 3 (descriptor) Phase 4 (compositional) Phase 5 (lowdata) Phase 6 (rank)
```

All six phases after Phase 0.5 are mutually independent and can be built
and launched in parallel. Phase 6's write-up cites Phase 2's number as its
"full" reference point but doesn't computationally depend on it. If GPU
capacity is constrained, a soft priority order (smaller/more central phases
first) is reasonable: Phase 2 (3 jobs), Phase 3 (90), Phase 4 (6), Phase 1
(315), Phase 6 (12), Phase 5 (153) - but this is scheduling convenience,
not a dependency.

## Total Round-2 job-count estimate

| Phase | Jobs |
|---|---:|
| 0.5 Infra pilot | 5 |
| 1 Backbone progression (T1-T7 only, 15 tasks, dim=16, 3 seeds) | 315 |
| 2 Headline | 3 |
| 3 Descriptor mechanism (3a + 3b) | 90 |
| 4 Compositional | 6 |
| 5 Low-data (hyper + baseline) | 153 |
| 6 Rank ablation | 12 |
| **Total** | **584** |

Well under a third of `arc1d_uniform_ablation` alone (2,210 jobs), mainly
because fixing `n_loops` and `num_layers` by precedent removed both the
L1-L6 sweep from Phase 1 and the architecture-search grid entirely.

## Verification

- Before building any phase, re-confirm its base-derivation file still
  exists and matches what's stated here.
- Phase 0.5 must show loss decreasing and no NaN in all 5 pilot cells
  before any other phase commits to its full job count.
- Every Round 2 run script must retain local `results.txt` files (the
  L5/L6 lesson).
- No training is launched by writing this document. Confirm with Fabio
  before Phase 1 (315 jobs) and Phase 5 (153 jobs), the two largest, are
  actually submitted.

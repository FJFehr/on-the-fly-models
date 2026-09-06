# ARC-1D hypernetwork story: stocktake overview

This is a stocktake, not a paper draft. It exists to answer one question
honestly for each stage of the story: **is this claim actually backed by a
finished, correctly-run experiment, and if not, what exactly is missing?**
The detail behind every line here lives in the three companion documents:

- [`01_data.md`](01_data.md) - the augmentation recipe, task list, and splits
- [`02_architecture.md`](02_architecture.md) - the converged model
  architecture(s), including two unresolved inconsistencies
- [`03_experiments.md`](03_experiments.md) - the full experiment catalog,
  one entry per `legacy/configs/experiments/` directory, with real hyperparameters
  and numbers pulled from source files and wandb

This round produced no new training runs. Every number below is either from
a local `results.txt`, a README's own stated findings, or a fresh wandb
query run during this stocktake (flagged explicitly where used, since it's
often more complete than what made it into the READMEs). A follow-up round,
separate from this one, will use [`04_checklist.md`](04_checklist.md) to plan
a from-scratch rerun under one consistent recipe.

Status key: **Confirmed** = finished, correctly run, numbers solid.
**Provisional** = finished, but with a caveat that weakens how much weight
the number can bear. **Not run** = doesn't exist yet. **Mixed** = finished,
and the honest picture is genuinely partial success, reported as such.

---

## 1. Smallest model that solves individual tasks

**Status: Provisional.** `legacy/configs/experiments/arc1d_uniform_ablation` is the
ancestor of every target-model architecture in this story. It ran a
7-step ablation (T1-T7) plus a 6-cell loop-count diagnostic (L1-L6), at two
widths, 17 tasks x 5 seeds, and the story is a genuine progression, not a
single decision:

- T1→T2: looped (`N_supervision=2`) training alone doesn't help yet at this
  stage (sinusoidal PE, no Canon, no RoPE).
- T3: adding Canon convolutions (ABCD kernel) is the first real architecture
  change.
- T4: adding RoPE, still flat (`n_loops=1`).
- T5: looping the middle (`n_loops=4`) - the first version of the actual
  `RoPECanonLoopedTransformer`.
- T6/T7: block-skip and per-loop-h0 (loop-skip) connections, which only pay
  off at dim=32/n_loops=4 (T7: 95.2% test) and actively hurt at dim=16.
- L1-L6 then swept `n_loops` directly with skips off: the reliable lever
  turns out to be **more loop iterations, not skip connections** - the
  no-skip sweep peaks at **n_loops=8 for dim=16 (93.3% test)** and
  **n_loops=16 for dim=32 (95.2% test, tying T7's skip-based result)**, with
  both widths degrading somewhat by n_loops=32.

The headline conclusion the story converged on: at dim=16, no skip,
n_loops=8, 11,760 backbone params, ~93% val/test - the smallest clean
recipe that's best-or-tied-best at both widths tested. Full tables in
[`03_experiments.md`](03_experiments.md#arc1d_uniform_ablation).

**The caveat that matters most in this whole write-up:** this project ran
entirely before the 2026-07-13 `max_steps`/`N_supervision` fix (commit
`4f8f4f1`). Every `N_supervision=2` cell here - i.e. everything except T1 -
trained on half its intended batches, and its LR scheduler never completed
warmup+decay. **It has not been re-run since the fix.** Every later
experiment in this story inherits its target-model choice from this run.
This is the single highest-priority item in [`04_checklist.md`](04_checklist.md).

Separately, note the L5/L6 cells (n_loops=16/32) have no local run
directories at all despite being reported in the README's table - those
numbers are reproduced here from the README, not re-derivable locally.

## 2. Can a hypernetwork learn to predict all of these?

**Status: Confirmed, for 15 of 18 categories, with frozen task descriptors.**

The individual-task version of this question was answered first
(`arc1d_hypermodel_looped`, per-task hypernetworks, one per category): fresh
wandb numbers (this project's local `results.txt` files don't exist, only
wandb has them) show **13 of 17 tasks reach 100% test exact match**,
`1d_move_dp` reaches 92%, `1d_flip` reaches 50%, `1d_recolor_cmp` reaches
33%, and `1d_recolor_cnt`/`1d_recolor_oe` reach 0% - already visible here,
before any joint training, that the recolor family (except `cmp`) and flip
are the hard cases. Note: 200/200 real recolor instances were separately
verified to be fully determined by their support pairs
([`01_data.md`](01_data.md)) - the recolor-family difficulty is a genuine
model limitation, not underdetermined data.

The joint-multitask version is the real claim, and it was answered across
several experiments as the architecture matured. Using **frozen task
descriptors (`frozen_td`)** - a one-hot task-identity vector projected in
and then frozen at random initialisation, never trained further - the most
recent post-fix runs reach:

- `arc1d_hypermodel_looped_rope_canon_capacity_baseline` (15 tasks,
  `1d_recolor_cnt`/`_oe` dropped for being structurally unsolved at 0% in
  every configuration tried): high-90s exact match.
- `arc1d_hypermodel_compositional_generalization/base.yaml` (same 15 tasks,
  most recent, cleanest recipe): **`frozen_td` reaches 98.75% val / 97.5%
  test exact match.**

`td` (learned, not frozen) performs near-identically in distribution
(98.75% val / 100% test in the same run) - the two are not distinguishable
on in-distribution accuracy alone. `frozen_td` is the one this story carries
forward, for a reason that only becomes visible in section 6 below: a
*learned* task descriptor actively destabilises generalisation to a category
it never saw, while a frozen one doesn't.

`notd` (no descriptor at all) reaches only **48.75% val / 46.25% test** in
that same run - a large, real gap, which is exactly the subject of the next
section.

The earlier 11-task joint run (`arc1d_hypermodel_looped_mix11`, older
architecture, predates several later improvements) reached 76.8-83.9% test
exact match depending on config - consistent with "the architecture and
recipe matured a lot between this run and the final one," not a
contradiction.

## 3. Removing the task-descriptor embedding

**Status: Mixed - "still learns most tasks" is accurate, "still learns all
tasks" is not.**

Without any task-identity signal (`notd`), the model still learns the
majority of tasks but meaningfully worse than with a descriptor, and the
gap is not small or consistent: **46.25-48.75%** (compositional_generalization,
most recent recipe) up to **82.5%** (the earlier `muon_diag` reference run,
different settings) - there is no single controlled number, only a
consistent direction. Three independent measurements agree on the
qualitative finding and roughly its size:

| Source | notd | td/frozen_td | gap |
|---|---:|---:|---:|
| `mechanism_ablation` (`arm_no_task_indicator`) | 63.6% | 74.2% | ~11pt |
| `grid` (pooled means, n=27 each) | 58.3% | 81.9% | ~24pt |
| `muon_diag` reference (quoted in `vae_disentanglement`) | 82.5% | 96.2-97.5% | ~14-15pt |

The origin story for *why* a descriptor was added at all
(`arc1d_hypermodel_disentanglement`, older data recipe, historical) is
instructive: mixing structurally similar tasks (`1d_move_1p` +
`1d_move_3p`) without a descriptor **fails completely** - the model
conflates them - while mixing structurally dissimilar tasks
(`1d_move_3p` + `1d_denoising_1c`) **succeeds fine without any descriptor**.
Adding a one-hot task descriptor resolves the similar-task conflation. This
is the mechanism, not just a correlation: the descriptor's job is
specifically to disambiguate tasks the model would otherwise merge.

## 4. Disentanglement is tricky without a descriptor, but real

**Status: Confirmed**, via cluster maps and linear-probe accuracy - not via
the beta-VAE bottleneck work, which was tried but is excluded from this
write-up by design (see [`03_experiments.md`](03_experiments.md) for the one
paragraph it gets).

The reference measurement (`muon_diag`, quoted inside the
`vae_disentanglement` README): a stratified-k-fold logistic-regression
probe, predicting task category from the hypernetwork's pooled task
representation, scores **73.3% for `notd` vs. 100% for both `td` and
`frozen_td`.** The representation still separates most tasks reasonably
well from the demonstrations alone, with no explicit identity signal - it's
degraded, not destroyed. Cluster maps at the same setting visibly show
`notd`'s move-family tasks (1p/2p/3p/dynamic) partially overlapping,
consistent with the lower probe score and with section 3's finding that
move-family conflation is exactly what a descriptor fixes.

The beta-VAE attempt to close this gap without an explicit descriptor
(11-arm full-training KL-anneal sweep, best beta=1e-3) reached 77.3%
probe / 72.0% exact-match - a mild improvement over plain `notd`'s
probe score, but still well short of `td`/`frozen_td`, and it isn't part of
the paper-track story per direction.

## 5. Compositional generalisation

**Status: Mixed - lead with what worked.** Trained on the 15 in-distribution
categories (`frozen_td`/`notd`, `arc1d_hypermodel_compositional_generalization`),
then evaluated zero-shot on 10 held-out categories built by chaining two
known single-step rules (e.g. "denoise, then shift") -
[`01_data.md`](01_data.md) has the construction details.

The standout: **`1d_comp_denoisemc_denoise1c`** (denoise multi-colour, then
denoise single-colour) reaches **10.0% exact match under `notd`** (0.0% for
every other category under either arm) with **92.1% token-level sequence
accuracy** - by far the highest of any composite category, under either
`notd` or `frozen_td`. The intuition: this composition chains two *variants
of the same underlying skill* rather than two structurally distinct
mechanisms - denoising multi-colour and denoising single-colour are
near-siblings (the same relationship that makes `1d_denoising_mc`
generalisable as a held-out *category* in section 6). Compositions appear
to transfer best when the two stages don't require genuinely combining
distinct mechanisms.

Everywhere else, the pattern is the same "high partial credit, near-zero
exact match" shape: token-level sequence accuracy sits at 0.63-0.92 across
every category, but exact match is 0.000 for 8 of 10 categories under both
arms (two categories reach 2.5% under one arm each). Overall exact match:
1.3% (`notd`), 0.3% (`frozen_td`). The model partially transfers the
constituent skills - it's clearly doing *something* structured, not
guessing - but essentially never reliably chains them into the exact right
sequence. That's the honest aggregate picture, reported as context rather
than the headline.

## 6. Generalisation to unseen task categories

**Status: Mixed - lead with what worked.** Leave-one-category-out: each of
5 categories held out in turn from an otherwise-14-category training set,
evaluated zero-shot. Fresh wandb data (2 seeds per cell for most cells, not
the 3 originally planned - see caveat below) supersedes the README's
seed-1-only table and is more informative, since it exposes seed variance
the single-seed table couldn't show:

| Held-out category | `frozen_td` | `notd` | `td` |
|---|---:|---:|---:|
| `1d_denoising_mc` | **1.0, 1.0** | 0.80, 1.0 | 0.0, 1.0 |
| `1d_flip`, `1d_hollow`, `1d_move_2p`, `1d_pcopy_mc` | 0.0 (every seed) | 0.0 (every seed) | 0.0 (every seed) |

The one success: `1d_denoising_mc`, held out from training that still
included its near-sibling `1d_denoising_1c` (same underlying transform,
differing only in single- vs. multi-colour noise). **`frozen_td` is the
only variant that succeeds consistently there** (both seeds = 1.0). `td` is
strikingly bimodal on the exact same category (0.0 then 1.0) - a learned
descriptor is unstable in a way a frozen one isn't. This is the concrete
evidence behind carrying `frozen_td`, not `td`, forward as this story's
standard choice (section 2).

The other 4 held-out categories fail completely - 0.0 exact match, every
variant, every seed. The model does not generalise to a genuinely unseen
category in the general case; success is specifically tied to having an
unusually close sibling still in the training set, not a general
capability.

**Caveat.** Local results for this project were lost in a disk-wipe
incident (`rm -r outputs/*` on `torrnode12`, intended to fix an NFS-quota
issue, deleted checkpoints and `results.txt` before syncing). Only 2 of the
planned 3 seeds actually landed on wandb (a third was never relaunched), and
2 of those 30 runs (`pcopy_mc/frozen_td`, `pcopy_mc/td`) have only 1 seed
each due to a failed/crashed run. See [`04_checklist.md`](04_checklist.md).

## 7. Low data augmentation

**Status: Mixed, and genuinely surprising - closes the story by testing the
Phase 2 value proposition directly.** The full recipe is up to ~200 unique
colour-permutation variants per base task; this axis subsamples down to as
few as 1-2 variants/base task (`variants_per_base_task`, nested/stratified,
[`01_data.md`](01_data.md)).

**Hypernetwork side** (`arc1d_lowdata`, joint 15-task): fresh wandb numbers
show test exact match already at **89.7% at level 1** (~1 variant/base
task), climbing to 95.4% (level 2), and plateauing at 96.7-98.8% from level
3 onward. The "near-full performance by level 2" framing in the README
roughly holds, though level 1 alone is already fairly strong.

**No-hypernetwork baseline** (`arc1d_lowdata_baseline`, one model per
category, no cross-task sharing at all) - **this is the load-bearing
comparison and it exists now**: all 135 planned jobs are finished (verified
via wandb; the local filesystem alone made this look like pure unrun
scaffolding, which is not correct). At level 1 (~40 examples/category), 12
of 15 categories *already* reach ~100% test exact match with a plain
single-task model - no hypernetwork, no cross-task transfer of any kind.

**This is the headline finding of this section, and it's more nuanced than
the original hypothesis.** The hypernetwork's low-data advantage over a
plain per-task baseline is real but narrow, not broad: at level 1, the two
approaches are statistically indistinguishable on 13 of 15 categories.
The advantage concentrates specifically on the hardest tasks:

- `1d_recolor_cmp`: hypernetwork 20% vs. baseline 0% - the one category the
  baseline cannot solve at any tested data level; a genuine, real
  cross-task-transfer win.
- `1d_scale_dp`: hypernetwork 100% vs. baseline 80% - a smaller real win.
- Everywhere else, both approaches are already near-ceiling by level 1, or
  (for `1d_move_dp`) both are still weak and indistinguishable.

In other words: most of these ARC-1D tasks are already easy enough for this
backbone that a single task-specific model barely needs any augmented data
to solve them. The hypernetwork's value-add from cross-task transfer is
real, but it's concentrated in the small number of genuinely hard tasks,
not a broad efficiency gain across the board. This nuance matters for how
the paper frames the low-data result - "the hypernetwork needs less data"
is true in aggregate but misleading if read as "cross-task transfer is
doing most of the work" for most tasks.

`arc1d_lowdata_lowrank` (rank x data cross) confirms the pattern holds down
to LoRA rank 1-2, not just the reference rank=8.

---

## What this means for the next round

Two structural things need resolving before a from-scratch rerun can be
"the" unified recipe rather than another entry in this same inconsistent
chain, both documented in full in
[`02_architecture.md`](02_architecture.md#open-questions-for-the-next-round-to-settle):

1. **`n_loops`**: `arc1d_uniform_ablation`'s own diagnostics recommend
   n_loops=8 (dim=16), but nearly every downstream hypernetwork experiment
   actually configures n_loops=4 - inherited from a different sweep
   entirely, not from the capacity story that section 1 is built on.
2. **`lora_adapter_rank`**: the rank sweep found rank=8 wins, and most
   experiments use it, but two experiments used for headline results in
   sections 4 and 5 (`vae_disentanglement`, `compositional_generalization`)
   both explicitly use rank=4 instead.

Everything else the next round needs is in [`04_checklist.md`](04_checklist.md).

**Update**: both of these are resolved in [`05_round2_plan.md`](05_round2_plan.md)
by fixing them to historical precedent (`n_loops=4`, `num_layers=8`) rather
than re-searching them, and treating `lora_adapter_rank` as a cut-down
ablation (full generation first, then a dedicated rank-ablation phase) -
see that document for the full Round 2 plan and its consistency check
against every open item in this stocktake.

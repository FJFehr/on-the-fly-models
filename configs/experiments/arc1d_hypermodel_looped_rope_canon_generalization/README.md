# arc1d_hypermodel_looped_rope_canon_generalization

## Goal

Every experiment so far has trained and evaluated on the same 15 task categories. This
experiment asks a different question: can the hypernetwork generalize to a task category it
has **never seen during training**, inferring the task purely from the 3-shot support
examples at inference time -- and does the learned one-hot task-identity embedding
(`hyper_head.num_tasks`) help or actively hurt that generalization?

The intuition for why it might hurt: a held-out category's one-hot embedding column never
receives a gradient during training, so at inference it's whatever it was at random init,
while every other column moved. The rest of the network never learned to handle an
untrained-random vector at that position, so `td` is a plausible way to make the model
*worse* at generalizing to something new, not just neutral.

Fabio's idea for a middle ground: freeze the one-hot projection at random init for **every**
category, not just the held-out one (`hyper_head.freeze_task_indicator`, new flag, see
`models/hypermodel.py`/`models/hypermodel_lightning.py`). If the projection never trains for
any category, the downstream network has to learn to work with fixed random directions
uniformly -- so the held-out category's random column isn't out-of-distribution relative to
what the network learned, unlike plain `td`. This is functionally a random-features /
hashing-trick task tag rather than a learned embedding table.

## Backbone (fixed across every arm)

Zhu block (RMSNorm + QK-norm + SwiGLU, `rope_canon_zhu_transformer`, all three flags on) +
AdamW/lr=0.001 -- the current leader in `arc1d_hypermodel_looped_rope_canon_lr_sweep`'s
zhu_all arm (mean exact-match 0.8625 across 5 seeds, vs. 0.81 at lr=1e-4 and ~0.79 at
5e-4/5e-5; lr_sweep wasn't fully finished when this was chosen, revisit if 1e-3 doesn't hold
up once it is). Not re-swept here -- this experiment's only axis is task-identity
conditioning. No task-descriptor ceiling arm (all 15 tasks in training, `td` on) is built this
round -- not needed to answer the core question, since the `notd`-in-distribution ceiling is
already free (see Reusing existing data below).

## Held-out tasks

5 task categories, each held out one at a time, chosen because each has a structurally
similar sibling (or siblings) left in the 14-task training set, so a generalization failure
can't be blamed on "no related signal existed anywhere in training":

| Held out | Short name | Sibling(s) remaining in training |
|---|---|---|
| `1d_move_2p` | `move2p` | `1d_move_1p`, `1d_move_3p`, `1d_move_dp`, `1d_move_2p_dp` (same move family) |
| `1d_denoising_mc` | `denoisingmc` | `1d_denoising_1c` |
| `1d_flip` | `flip` | `1d_mirror` (different mechanism -- full reversal vs. pivot-colour reflection -- a weaker/harder pairing, closer to true zero-shot) |
| `1d_pcopy_mc` | `pcopymc` | `1d_pcopy_1c` |
| `1d_hollow` | `hollow` | `1d_fill` |

`1d_recolor_cnt`/`1d_recolor_oe` were not considered as held-out targets -- they're
structurally dead (0% exact-match even fully in-distribution, per
`arc1d_hypermodel_looped_rope_canon_capacity_baseline/README.md`), so a held-out failure there
would be uninterpretable.

## Grid

5 held-out tasks x 3 task-identity variants, 3 seeds each = 45 jobs:

| Variant | `hyper_head.num_tasks` | `hyper_head.freeze_task_indicator` | Meaning |
|---|---:|---:|---|
| `notd` | `null` | `false` | No task-identity signal; task inferred purely from 3-shot examples |
| `td` | `18` | `false` | Learned one-hot embedding (held-out category's column never trains) |
| `frozentd` | `18` | `true` | One-hot embedding frozen at random init for every category (Fabio's idea) |

Config naming: `arm_<short_name>_<variant>.yaml`, e.g. `arm_move2p_frozentd.yaml`. Every leaf
sets `task_categories` to the 14-task list (full 15 minus the held-out one) and
`val_task_categories` to the full 15-task list, so `results.txt`'s automatic per-category
exact-match breakdown reports both in-distribution scores and the held-out category's
zero-shot score from the same run.

## Reusing existing data

`arc1d_hypermodel_looped_rope_canon_lr_sweep`'s `arm_zhuall_lr1e-3` (5 seeds,
`looped_hyper_rope_canon_lr_sweep_zhuall_lr1e-3_seed{1..5}`) is an exact match on backbone,
optimizer, and LR, trained on all 15 tasks with `notd` (`num_tasks: null`, same as this
branch's default throughout). Its per-category `val_query_exact_match_by_task_<category>`
breakdown is a ready-made **in-distribution notd ceiling** for each of the 5 held-out
categories above -- no rerun needed. There is no equivalent in-distribution ceiling for `td`
or `frozentd` (not built this round, see Goal).

## Running

```bash
# All 15 leaf configs, 3 seeds each. 45 jobs.
bash scripts/run_hypermodel_looped_rope_canon_generalization.sh
```

Split across nodes via `SEEDS_OVERRIDE` and/or `CELL_GLOB` (see script header for examples).

## Reading results

For each held-out category, pull `val_query_exact_match_by_task_<category>` out of
`results.txt` (mean +/- std across 3 seeds) for `notd`/`td`/`frozentd`, and compare against
the in-distribution ceiling from the reused `lr_sweep` zhu_all data. As of the full rerun below,
`results.txt` also carries `val_query_accuracy_by_task_<category>` (per-category token accuracy,
mirroring `val_query_exact_match_by_task_<category>` -- see
`models/hypermodel_lightning.py`'s `_accumulate_query_accuracy_by_task_category`), added because
the seed-1 findings below flagged one anecdotal case of high partial-credit accuracy despite 0
exact match; this metric lets that be checked systematically across every held-out category
rather than from one logged hard example. Key questions:
- Does `td` score meaningfully lower than `notd` on the held-out category specifically (while
  scoring similarly or better on the 14 in-distribution categories)? That's the direct
  evidence that the task descriptor holds back generalization.
- Does `frozentd` recover some or all of the gap between `td` and `notd`? That would support
  the frozen-random-embedding idea as a practical fix that keeps whatever benefit task
  conditioning gives in-distribution while not actively hurting zero-shot transfer.
- Does the answer differ between `1d_move_2p` (strong same-family siblings) and `1d_flip`
  (weaker sibling pairing)? A consistent pattern across both would be stronger evidence than
  either alone.

## Findings (full rerun, 3 seeds, all 45 jobs)

**Supersedes the earlier seed-1-only table below.** The full 3-seed grid changes the read on
`1d_denoising_mc` and `1d_move_2p` substantially -- what looked like a clean `frozen_td`-specific
win in seed 1 turns out to be seed noise once seeds 2 and 3 are in.

Held-out-category exact match (`val_query_exact_match_by_task_<category>`, one value per seed,
mean across the 3):

| Held out | `td` (seeds) | `td` mean | `notd` (seeds) | `notd` mean | `frozentd` (seeds) | `frozentd` mean |
|---|---|---:|---|---:|---|---:|
| `1d_denoising_mc` | 0.4, 1.0, 0.0 | 0.467 | 0.0, 1.0, 0.8 | 0.600 | 0.0, 1.0, 0.8 | 0.600 |
| `1d_flip` | 0.0, 0.0, 0.0 | 0.000 | 0.0, 0.0, 0.0 | 0.000 | 0.0, 0.0, 0.0 | 0.000 |
| `1d_hollow` | 0.0, 0.0, 0.0 | 0.000 | 0.0, 0.0, 0.0 | 0.000 | 0.0, 0.0, 0.0 | 0.000 |
| `1d_move_2p` | 0.8, 0.0, 0.0 | 0.267 | 0.0, 0.0, 0.4 | 0.133 | 0.6, 0.0, 0.0 | 0.200 |
| `1d_pcopy_mc` | 0.0, 0.0, 0.0 | 0.000 | 0.0, 0.0, 0.0 | 0.000 | 0.0, 0.0, 0.0 | 0.000 |

**Headline, revised: 3 of 5 held-out categories (`1d_flip`, `1d_hollow`, `1d_pcopy_mc`) are a
robust, seed-independent null result** -- exact match is 0 for every seed and every
task-identity variant, 9/9 cells each. The model genuinely does not generalize to these unseen
categories, and that conclusion is now solid, not a seed-1 artifact.

**The other 2 categories (`1d_denoising_mc`, `1d_move_2p`) do show real, non-trivial partial
exact-match success -- but it's noisy and not cleanly attributable to any one task-identity
condition.** Seed 1's original finding (`frozen_td` hits a clean 1.0 on `denoising_mc`, `td`
never does) does not hold up: `td` scores 1.0 on *seed 2*, and all three variants land somewhere
between 0.0 and 1.0 across the three seeds, with no consistent ranking. Same story for
`move_2p`: exact match is nonzero for at least one seed under every variant (`td`: seed 1;
`notd`: seed 3; `frozentd`: seed 1), never more than one seed per variant, and never the same
seed across variants. Read this as: these two categories (each with the closest sibling pairing
in the grid) are *sometimes* within reach of the model, regardless of task-identity condition,
but whether a given run actually lands it looks like it depends on something seed-level
(initialization, training dynamics) rather than a property of `td`/`notd`/`frozen_td` as such.
The original "`frozen_td` uniquely cracks `denoising_mc`" story was real for seed 1, but isn't
the general pattern.

**Token accuracy tells a calmer, more consistent story than exact match, and (at this
architecture scale) doesn't show a strong `notd` advantage the way the smaller v2-scale
experiments do.** Mean held-out-category accuracy, averaged over categories (note: several
`results.txt` files predate the accuracy-by-category metric and were correctly left alone by the
launcher's skip-logic rather than re-run, so accuracy coverage is 2/3 or 1/3 seeds for a few
cells -- see each category's own seed list above for exactly which):

| Variant | Mean held-out accuracy (avg of per-category means) |
|---|---:|
| `td` | 0.864 |
| `notd` | 0.850 |
| `frozentd` | 0.855 |

All three are within 1.4 points of each other -- essentially tied, unlike `arc1d_v2_generalization`
(same question, ~150x smaller matched-scale architecture) where `notd` clearly wins (0.841 vs.
0.740/0.736). Whatever is driving `notd`'s accuracy advantage at the smaller scale isn't showing
up here at this architecture's much larger capacity.

**Caveats**: accuracy figures above mix seed counts per cell (see the per-category tables in the
raw `results.txt` files for exact coverage) since re-running already-complete jobs purely to
backfill a metric wasn't done this pass -- worth a full clean rerun if the accuracy comparison at
this scale needs to be load-bearing for a paper claim, rather than a directional read.

## Findings (seed 1 only, historical -- superseded above)

Kept for provenance; the 3-seed table above is the current read.

Held-out-category exact match (`val_query_exact_match_by_task_<held_out_category>`, pulled
from wandb after the disk-space incident wiped local `results.txt`/checkpoints for these runs,
see below):

| Held out | `td` | `notd` | `frozentd` |
|---|---:|---:|---:|
| `1d_denoising_mc` | 0.0 | 0.80 | **1.0** |
| `1d_flip` | 0.0 | 0.0 | 0.0 |
| `1d_hollow` | 0.0 | 0.0 | 0.0 |
| `1d_move_2p` | 0.0 | 0.0 | 0.0 |
| `1d_pcopy_mc` | 0.0 | 0.0 | 0.0 |

`1d_move_2p`'s `frozentd` run showed high per-example token accuracy on some held-out examples
despite 0 exact match (`val_hard_final_task | 1d_move_2p:47 | query_acc=0.90`) -- the anecdote
that originally motivated adding the systematic `val_query_accuracy_by_task_<category>` metric.

**Caveat on this data**: after this experiment's first pass finished (44 of 45 jobs done, one
NCCL-timeout failure), a manual `rm -r outputs/*` on torrnode12 -- an attempt to fix what
looked like a disk-space failure but was really the shared `/homes/55` NFS quota being full,
mostly from an un-cleaned local `wandb/` run cache (49G, freed via `wandb sync --clean
--clean-force`) -- deleted every local `results.txt`/checkpoint for this run before it was
rsynced. The seed-1 numbers above were reconstructed from wandb (metrics/config/logged images
survive independently of local files; `log_model=False` means no checkpoints were ever
uploaded there, so those are gone for good). Seeds 2-3 were relaunched from scratch after
freeing the quota.

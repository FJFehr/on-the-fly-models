# arc1d_hypermodel_compositional_generalization

## Goal

`arc1d_hypermodel_looped_rope_canon_muon_diag`'s notd/td/frozentd comparison and the
follow-on `vae_disentanglement` work trained a hypernetwork on 15 in-distribution ARC-1D
task categories and found `frozentd` (a frozen/untrained one-hot task-identity projection)
matches `td`'s task-solving accuracy while being simpler.

This experiment asks a different question: **does the model generalise compositionally?**
If it has learned "denoise" and "shift" as separate rules, can it solve a never-seen task
that requires denoising a block *then* shifting it — a combination it was never trained on,
built from two skills it was?

## Why this needed new data, not just a new config

ARC-1D tasks are not procedurally generated anywhere in this repo. `scripts/build_arc_1d.py`
only ingests static, pre-made JSON from the external `khalil-research/1D-ARC` benchmark
(confirmed by cloning it directly: it ships only `dataset/*.json` task files and matplotlib
visualisations, no generator source). That benchmark has no notion of chaining two rules
together, so there was no way to build "denoise then shift" out of existing data — it had to
be generated from scratch. That generator is `data_modules/arc1d_compositional.py`, tested in
`tests/test_arc1d_compositional.py`.

## Base rule semantics

Reverse-engineered by inspecting real 1D-ARC JSON examples directly (not assumed from category
names):

| Rule | Semantics |
|---|---|
| `denoise_1c` | Solid block of colour C + scattered isolated noise pixels of colour C outside the block → noise removed (set to background 0). |
| `denoise_mc` | Solid block of colour C with a few pixels *within* the block corrupted to other colours → corrupted pixels fixed back to C. |
| `shift(n)` | Solid block translated by a fixed `n` cells in a fixed direction; vacated cells become background. |
| `move_dynamic` | Block translated so it ends adjacent to a pointer marker; direction/distance implied by the pointer's position, pointer stays in place. |
| `mirror` | Block removed from its original position and reflected to the far side of a pivot marker; pivot stays in place. |
| `fill` | Two markers of the same colour C with a gap between them → gap filled with C (endpoints inclusive). |
| `hollow` | Solid block of colour C → interior cleared to 0, only the first/last cell of the block survive (inverse of `fill`). |
| `copy` (pcopy) | A template shape in colour C1 + a lone marker pixel of colour C2 elsewhere → marker location gets the template's shape stamped in colour C2; template stays in place. |

## Conventions verified against the real benchmark and baked into the generator

Several things in this generator aren't arbitrary choices — they were checked directly against
the raw 1D-ARC JSON and the actual training dataset (`data/arc_1d_looped_augmented`) rather than
assumed, because getting them wrong would confound the compositional-generalisation question
with an unrelated distribution-shift the model was never exposed to at all:

- **Marker/pivot directionality.** `move_dp`/`move_2p_dp`/`mirror`/`pcopy_mc` all place their
  marker strictly to the *right* of the primary object in 100% of sampled real examples — both
  the raw benchmark (200/200 each) and the actual augmented training set (2000-3000/2000-3000
  sampled rows each) — never to the left. This holds despite `scripts/augment_arc_1d.py`'s
  `--mirror` (sequence-reversal) flag defaulting on; whoever built `arc_1d_looped_augmented`
  must have run with it off, since there's no trace of a leftward example anywhere in it. The
  generator places every marker/pivot to the right for exactly this reason: a leftward example
  would be a geometry the model never saw even in single-task training, not a genuine test of
  composing two known skills.
- **`copy`'s template shape.** `1d_pcopy_1c`/`1d_pcopy_mc` always use a template of exactly
  length 3 in the real benchmark (200/200 examples each), with the marker always exactly
  centred (offset 1, checked on 289 real `pcopy_mc` marker→stamp pairs — never varies). This
  generator deliberately deviates from that constant-3 convention per user request: template
  length is 3 *or* 5, chosen once per task instance (so all of that task's own support/query
  pairs stay consistent) via `COPY_TEMPLATE_LENGTHS`. Both are odd, so the centre-anchored
  marker convention (`center_offset = shape_len // 2`) still holds for either.
- **Mirror's pivot colour.** `1d_mirror`'s pivot is colour 9 in literally every real example —
  200/200 raw benchmark, 3000/3000 sampled training rows — a hard global constant the
  augmentation script explicitly protects (`FIXED_RULE_COLOURS` in `scripts/augment_arc_1d.py`
  never remaps it). The generator uses a fixed `MIRROR_PIVOT_COLOUR = 9` for exactly this
  reason, not an arbitrary "some other colour" pick. `move_dp`/`move_2p_dp`'s pointer colour,
  by contrast, only needs to be consistent *within* one task (real data varies it across
  different task instances) — kept as a simple global constant here too (`MOVE_POINTER_COLOUR
  = 8`), deliberately distinct from 9 so a combo using both markers in one grid can't collide.
- **Block/template length for `denoise_1c`/`denoise_mc`/`shift`/`hollow`/`move_dynamic` is
  *not* calibrated to the real distribution.** Real `1d_denoising_1c` blocks are length 10-15
  and real `1d_denoising_mc` blocks are length 20-26 (within a 32-33-length sequence — the
  block occupies nearly the entire sequence), both considerably larger than this generator's
  ranges. Recalibrating to match would make composing a second stage geometrically
  tight-to-impossible for `denoise_mc` in particular (little to no free space left for a second
  marker). Decided to keep the current smaller-block simplification rather than chase exact
  size-fidelity here.

## The generator's architecture

Each composite category chains two base rules sequentially: a **stage 1** function constructs
a fresh random object from scratch (plus whatever corruption/markers its own single-task rule
needs) and reports what that rule's own output would be — this is the object stage 2 receives.
A **stage 2** function then further transforms (`shift`, `mirror`, `move_dynamic`), corrupts
(`denoise_1c`), hollows (`hollow`), or duplicates (`copy`) that object to produce the final
input/output pair. `fill` is stage-1-only (it *is* the initial marker representation of the
not-yet-solidified object) and `copy` is stage-2-only (duplication only makes sense once
there's already a settled object to copy). Every task record uses the same schema
`scripts/build_arc_1d.py` already produces: `task_category`, `task_id`, `sequence_length`,
`support_inputs`/`support_outputs` (3), `query_input`/`query_output` (1).

## Combo resolution — final 10

Full pairwise space across {`denoise_1c`, `denoise_mc`, `shift`, `move_dynamic`, `mirror`,
`fill`, `hollow`, `copy`} was worked through systematically; here's what survived and why.

### Included

| Combo | Resolved order | Pairing type | Notes |
|---|---|---|---|
| `denoise_1c` + `shift 3` | `shift3(denoise_1c(x))` | content + position | The worked example this whole idea started from. |
| `mirror` + `fill` | `mirror(fill(x))` | content + position | Reordered from "mirror and fill" — fill first gives `mirror` an actual block to reflect; `mirror(x)` then `fill` has no gap left to fill. |
| `fill` + `shift` | `shift(fill(x))` | content + position | |
| `fill` + `move_dynamic` | `move_dynamic(fill(x))` | content + position | Replaces the removed `move_dynamic + mirror` combo (see Excluded below); restores `move_dynamic` to 2 appearances. |
| `hollow` + `shift` | `shift(hollow(x))` | content + position | No degeneracy since shift doesn't undo hollow. |
| `denoise_mc` + `copy` | `copy(denoise_mc(x))` | content + duplication | Originally `denoise_1c + copy`; swapped to `denoise_mc` — `denoise_1c`'s own noise pixels are isolated same-colour dots outside the block, which visually collide with `copy`'s isolated different-colour marker dot (two "lone dot" elements with different meanings). `denoise_mc`'s corruption sits *inside* the block, so the only lone dot in the grid is the copy marker. |
| `denoise_mc` + `denoise_1c` | `denoise_1c(denoise_mc(x))` | content + content | Fixes internal multicolour corruption first, then removes external same-colour noise — combines both corruption "flavours" in one instance. |
| `move_dynamic` + `hollow` | `hollow(move_dynamic(x))` | position + content | |
| `shift` + `copy` | `copy(shift(x))` | position + duplication | |
| `denoise_mc` + `mirror` | `mirror(denoise_mc(x))` | content + position | |

Rule coverage across the 10: `denoise_1c`×2, `denoise_mc`×3, `shift`×3, `mirror`×2, `fill`×3,
`hollow`×2, `move_dynamic`×2, `copy`×2 — every rule appears at least twice, no lonely one-offs.

### Excluded, with reasons

- **`hollow + fill` (either order)** — degenerates to the identity function: fill-then-hollow
  strips a block back to its original two endpoints; hollow-then-fill re-solidifies it back to
  the original block. Net input==output, nothing to learn.
- **`denoise_1c + fill` (either order)** — `denoise_1c` expects a single corrupted block; `fill`
  expects two disjoint gap-markers. The conventions don't hand off cleanly in either order.
- **`shift + move_dynamic` (either order)** — both are pure positioning rules. Composing them
  collapses into "move to a different final spot," not a combination of two distinct skills.
- **`shift + mirror`** — reflecting position `p` across pivot `m` gives `2m − p`; shifting `p`
  by a fixed `n` first gives `2m − p − n = 2(m − n/2) − p`, i.e. exactly "mirror around a
  different fixed pivot." Since the shift is constant every time, a model could ace this purely
  by generalising plain `mirror` with an offset pivot, never needing to invoke `shift` as a
  separately-chained skill. (`move_dynamic + mirror` escapes this same collapse — its
  translation amount is a *data-dependent* function of the pointer's position, not a fixed
  constant — but see below, it was dropped anyway.)
- **`fill + copy`** — fill can produce a degenerate single-cell block (zero-gap markers), an
  ambiguous/confusing template to copy.
- **`move_dynamic + copy`, `hollow + copy`, `denoise_mc + hollow`, `denoise_mc + fill`,
  `denoise_mc + move_dynamic`** — dropped per user call during the diversity-selection pass.
- **`move_dynamic + mirror`** — originally included (and the only position+position pairing
  that isn't a linear collapse, since `move_dynamic`'s translation is pointer-position-dependent
  not constant). Removed after visual review — didn't work out in practice. Replaced by
  `fill + move_dynamic` to keep `move_dynamic` from being a one-off.
- **`1d_recolor_*` categories** — excluded from consideration entirely, per the original request.

## Files

- `data_modules/arc1d_compositional.py` — the generator (stage 1/2 functions, `ObjectState`,
  `COMPOSITE_TASK_SPECS` registry, `generate_pair`/`generate_task`/`generate_dataset`).
- `tests/test_arc1d_compositional.py` — golden mathematical checks for every stage-2 transform
  against hand-built `ObjectState`s, structural checks for every stage-1 constructor across many
  seeds, and full-pipeline schema/consistency validation for all 10 categories.
- `data/task_visualisations/compositional_prototype/` — rendered example grids (via the
  existing `visualisation/plot_tasks.py` CLI — no new plotting code needed, since task records
  already match the schema `visualisation/arc.py:render_task_figure` consumes).
- `scripts/build_arc1d_compositional.py` — generates the 400-instance (10 categories × 40)
  held-out `DatasetDict` at `data/arc_1d_compositional_holdout`, single `holdout_test` split.
- `base.yaml`/`notd.yaml`/`td.yaml`/`frozen_td.yaml` — the training configs (see below).
- `scripts/eval_compositional_holdout.py` — loads a trained `notd` or `frozen_td` checkpoint
  and runs it zero-shot on the held-out set (`td` is not supported — see its docstring).
- `scripts/run_hypermodel_compositional_generalization.sh` — trains all 3 arms then runs the
  held-out eval for `notd`/`frozen_td`, mirroring
  `scripts/run_hypermodel_vae_disentanglement.sh`'s conventions (skip-if-already-done, one GPU
  node per run, `FREE_GPUS_FLAG`).

## Training configs and the `num_tasks` / `TASK_CATEGORY_INDEX` mechanics

`base.yaml` reuses the `arc1d_hypermodel_looped_rope_canon_muon`/`vae_disentanglement` recipe
verbatim (Zhu backbone, Muon `muon_lr=0.005`/`muon_momentum=0.95`, `lora_adapter_rank=4`,
`max_steps=2000`, `N_supervision=2`, the same 15 `task_categories`/`val_task_categories`).
`notd.yaml`/`td.yaml`/`frozen_td.yaml` only override `hyper_head.num_tasks`/
`freeze_task_indicator`.

The task-category → one-hot-index mapping (`TASK_CATEGORY_INDEX` in
`models/hypermodel_lightning.py`) is a **fixed, hardcoded registry covering all 18 known
ARC-1D categories** — not derived from a config's `task_categories` list. This is why every
existing `td`/`frozen_td` config in this codebase sets `hyper_head.num_tasks: 18`, even ones
that only train on a 15-category subset: the registry (and therefore the one-hot width) always
covers the full 18, regardless of which subset is actually sampled during training.

`frozen_td.yaml` sets `num_tasks: 28` (18 + the 10 composite categories) for exactly this
reason: the 10 composite categories need reserved indices (18-27) in the frozen projection
matrix, and that matrix's width is fixed at model-init time — it can't be widened after
training. Since indices 18-27 are never sampled during training (only the 15 categories in
`task_categories` are), this is safe: those rows are just additional random-but-fixed
directions the frozen (never-trained) projection never had a reason to rely on, exactly like
the 15 in-distribution rows it also never learns. `scripts/eval_compositional_holdout.py`
registers the composite category names into `TASK_CATEGORY_INDEX` at indices 18-27 for the
duration of the eval run only (a pure-addition, backward-compatible mutation — training itself
never needs the composite names in that dict, since it never looks them up).

`td.yaml` keeps `num_tasks: 18` (no reserved composite indices) — trained for parity with the
original notd/td/frozentd comparison, but deliberately **not evaluated** on the held-out set:
a learned one-hot embedding is undefined for an index it never saw during training, unlike
`frozen_td`'s untrained projection. `scripts/eval_compositional_holdout.py` refuses to run
against a config whose `num_tasks` is too small to hold the composite indices, with a message
explaining why, rather than crashing inside `F.one_hot`.

## Held-out evaluation and the embedding-cluster overlay diagnostic

`scripts/eval_compositional_holdout.py` loads a trained checkpoint, runs it unmodified (no
fine-tuning) on the held-out set via the model's own `predict_batch`/`build_task_records`
(the same methods the standard validation loop uses), and reports per-category exact-match and
sequence accuracy to `results.txt`, plus a few rendered qualitative predictions per category.

It also extends the existing end-of-run embedding-cluster diagnostic
(`training/logging.py:log_embedding_cluster_plots`,
`visualisation/embedding_clusters.py:render_embedding_cluster_figure`) with an optional
`holdout_dataloader` — when passed, the pooled task-latent projection (PCA/t-SNE/UMAP) is fit
on the combined reference-validation + held-out embeddings, then rendered with reference points
as translucent circles and held-out points as full-colour X's drawn on top, so it's visually
clear whether the model places never-seen compositional examples near their constituent base
rules' clusters. This is purely additive — omitting `holdout_dataloader` (every other
experiment's call site) reproduces the original single-group behaviour exactly.

Two things worth knowing if you touch either of these files:

- `LightningModule.trainer` raises `RuntimeError` (not `AttributeError`) when the model isn't
  attached to a `Trainer`, so the old `getattr(model, "trainer", None)` guard in
  `log_embedding_cluster_plots` didn't actually protect a standalone caller like this eval
  script (which never calls `Trainer.fit()`). Fixed with a `try`/`except RuntimeError` — a
  pre-existing latent bug, only ever exposed once something called this function outside the
  normal `train.py` → `Trainer.fit()` → `run_post_training_artifacts` lifecycle.
- t-SNE/UMAP don't cleanly support projecting new points into a separately-fitted embedding,
  which is why the reference and held-out vectors are concatenated and projected together,
  then split back out by group for styling — not fit once on reference and re-applied to
  held-out points.

## Status

Phase 1 (generator + tests + prototype visuals) and Phase 2 (held-out dataset generation,
training configs, held-out eval script + run script, embedding-cluster overlay) are both done.
Training and held-out evaluation have both run (`notd`/`td`/`frozentd` trained, `notd`/`frozen_td`
evaluated zero-shot), single seed 42 — see Findings below.

## Findings (seed 42)

In-distribution (val/test, all 15 training categories): `notd` reaches 48.8% val exact match
(93.6% token accuracy), while both task-identity variants are near-ceiling — `td` 98.75% exact
match, `frozen_td` 98.75% exact match (test: `td` 100%, `frozen_td` 97.5%). Expected: this is the
same notd/td/frozentd pattern found throughout the rest of this story, just at this experiment's
own (larger) recipe.

Zero-shot on the 10 held-out composite categories (`scripts/eval_compositional_holdout.py`,
`n=40` per category):

| category | `notd` exact_match | `notd` seq_accuracy | `frozen_td` exact_match | `frozen_td` seq_accuracy |
|---|---:|---:|---:|---:|
| `denoise1c_shift3` | 0.000 | 0.810 | 0.000 | 0.716 |
| `denoisemc_copy` | 0.000 | 0.634 | 0.025 | 0.742 |
| `denoisemc_denoise1c` | 0.100 | 0.921 | 0.000 | 0.898 |
| `denoisemc_mirror` | 0.025 | 0.860 | 0.000 | 0.466 |
| `fill_mirror` | 0.000 | 0.792 | 0.000 | 0.698 |
| `fill_movedynamic` | 0.000 | 0.678 | 0.000 | 0.672 |
| `fill_shift3` | 0.000 | 0.756 | 0.000 | 0.721 |
| `hollow_shift3` | 0.000 | 0.854 | 0.000 | 0.743 |
| `movedynamic_hollow` | 0.000 | 0.807 | 0.000 | 0.780 |
| `shift3_copy` | 0.000 | 0.713 | 0.000 | 0.629 |
| **overall** | **0.013** | – | **0.003** | – |

**Headline: the model does not compose two known skills into a never-seen combination, but it
is doing something far from random on the way there.** Exact match is essentially 0 for both
`notd` and `frozen_td` on every composite category (one 10% and one 2.5% partial exception),
while token-level accuracy sits at 46-92% per category — the model reliably gets most of the
sequence right without ever landing the fully-composed answer. Unlike the in-distribution and
leave-one-out-category results, `frozen_td` does not clearly beat `notd` here; if anything `notd`
looks marginally better on several categories (e.g. `denoisemc_mirror`: 86.0% vs 46.6%) — `notd`
wins the token-accuracy comparison on 9/10 categories, mean 0.782 vs. 0.707. This isn't just this
run's noise: the same direction (`notd` > `frozen_td`, 6/10 categories, mean 0.714 vs. 0.650)
shows up again in `arc1d_v2_compositional_generalization`'s independent rerun at a completely
different (much smaller) architecture — see that README's Findings section for the combined
comparison, outlier caveat, and a candidate mechanism (a task-identity anchor, even a frozen
one, may help when the true answer is a known category but work against blending two rules when
it isn't — the opposite regime from leave-one-out generalization, where freezing that anchor
helped).

**Caveats**: single seed (42) only — no variance estimate. See
`arc1d_v2_compositional_generalization/README.md` for a rerun of this same question at the
minimal "matched-scale" architecture (10,156 params, ~150x smaller) the rest of the v2 story now
uses — same near-zero-exact-match/50-90%-token-accuracy pattern holds there too, so this isn't an
artifact of this experiment's particular (much bigger) recipe.

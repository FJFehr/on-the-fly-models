# Experiment 4: first joint (all-task) hypernetwork, minimal recipe

## Question

Can a hypernetwork small enough to match the target model's own new scale
(dim=6/dim=4, Experiment 1) recover what Experiment 2 showed a single
shared weight set cannot do — solve all 14 ARC-1D tasks *jointly*, not one
model per task? And how small can the hypernetwork mechanism itself be,
now that what it needs to generate is tiny? Full background:
`docs/arc1d_story/06_phase1_findings.md`,
[`arc1d_v2_minimal_size`](../arc1d_v2_minimal_size/README.md) (Experiment 1),
[`arc1d_v2_multitask`](../arc1d_v2_multitask/README.md) (Experiment 2, the
capacity-cliff motivation for this experiment).

## The finding

**With a task-identity signal, a hypernetwork closes most of the capacity
cliff Experiment 2 exposed — even shrunk all the way down to match the
size of the tiny target model it generates (~11.4K params at dim=4, ~20x
smaller than the fixed-width ~200K recipe this grid used to run at).
Without one, the hypernetwork still helps enormously over direct joint
training, but a well-understood, specific failure mode (move-family
conflation) accounts for most of the remaining gap — and at the smallest,
matched scale that failure mode starts to bite `frozen_td` too, not just
`notd`.**

dim=6 still runs the original fixed-width hypernetwork encoder (`hidden_dim=32,
num_heads=4, num_layers=2, output_dim=64`, `bottleneck_dim=128` — see
"Method" below); dim=4 has been resized to the proportionally-sized
"matched-scale" recipe (every pipeline dimension set to match the dim=4
target itself) that used to be a separate exploratory aside — that is now
the standard sizing for dim=4, not an alternative:

| Config | test exact match | mean | linear probe accuracy |
|---|---|---:|---:|
| **dim6, `frozen_td`** (3 seeds, ~338K params) | 0.986 / 0.986 / 0.986 | **98.6%** | 100% |
| dim6, `notd` (3 seeds, ~337K params) | 0.729 / 0.686 / 0.614 | 67.6% | 81-87% |
| **dim4, `frozen_td`** (5 seeds, ~11.4K params, matched-scale) | 0.936/0.922/0.949/0.955/0.889 | **93.0%** | 92.9-100% |
| dim4, `notd` (5 seeds, ~11.4K params, matched-scale) | 0.541/0.531/0.551/0.671/0.575 | 57.4% | 54.5-68.1% |

Reading this against the two experiments it sits between:

| | Individual (Exp. 1) | Joint direct (Exp. 2) | Joint hypernetwork (this experiment) |
|---|---:|---:|---:|
| dim=6, with task ID | 98.9% | 74.0% | **98.6%** |
| dim=6, no task ID | - | 40.9% | 67.6% |
| dim=4, with task ID | 94.6% | 19.7% | **93.0%** |
| dim=4, no task ID | - | 12.0% | 57.4% |

At dim=6's larger, fixed-width hypernetwork, `frozen_td` doesn't just
improve on the joint-direct capacity cliff — it erases it, landing within
noise of individual training, remarkably stable across seeds (the *exact
same* 0.9857 all three runs). At dim=4's matched scale, `frozen_td` still
closes most of the gap on average (93.0% vs. individual training's 94.6%)
but no longer erases it cleanly — the per-task breakdown shows real
softening on specific tasks, not just noise:

```
dim4 (matched-scale) per-task exact match, mean across 5 seeds:
  frozen_td                            notd
  1d_move_dp        55.2               1d_move_1p         6.2
  1d_mirror          73.6              1d_move_3p         8.4
  1d_flip            80.0              1d_flip           13.2
  1d_move_2p         94.4              1d_move_dp        21.2
  everything else   99.8-100.0         1d_move_2p        23.8
                                        1d_move_2p_dp     40.6
                                        1d_mirror         41.6
                                        1d_scale_dp       76.2
                                        everything else   89.2-100.0
```

Without a task-identity signal, the whole move family (`1d_move_1p/2p/3p`)
collapses together — exactly the conflation mechanism `00_overview.md`
section 3 already documented for direct training
(`arc1d_hypermodel_disentanglement`'s original motivation for adding a
descriptor at all). `frozen_td` fixes most of this at dim=6's larger
encoder, matching the project's standing explanation. At dim=4's genuinely
matched scale, the same failure mode now shows through even *with*
`frozen_td` on the hardest members of that family (`move_dp`, `mirror`,
`flip`) — a real, new finding from shrinking the hypernetwork all the way
down: a task-identity signal still helps enormously, but it's no longer
enough on its own to fully paper over a hypernetwork with this little
spare capacity.

**Disentanglement mostly matches the exact-match story, with one new
wrinkle.** At dim=6, linear-probe accuracy (predicting task category from
the hypernetwork's pooled task representation) is a clean 100% for every
`frozen_td` run and 81-87% for every `notd` run. At dim=4's matched scale,
`frozen_td` is no longer *always* perfect (92.9-100% across the 5 seeds,
mean 97.9%) and `notd` drops much further, to 54.5-68.1% (mean 60.9%) —
consistent with the exact-match story: a hypernetwork this small has
measurably less spare capacity to keep task representations cleanly
separated once the identity signal is gone. See
`outputs/figures/02_hypernetwork_multitask/seed1/cluster_dim4_paired_seed1_tsne.png`
(regenerate with `plot_embedding_clusters.py --dim 4 --all-seeds`).

## Method

- **Target model**: `rope_canon_looped_transformer`, `hidden_dim=6`
  (primary, 2,444 params trained directly in Experiment 1) and `hidden_dim=4`
  (secondary, 1,398 params) — flat (`n_loops=1`), single pass, identical to
  Experiment 1's recipe, now hypernetwork-generated instead of directly
  trained.
- **Hypernetwork encoder, dim=6**: `rope_canon_transformer` (plain, non-Zhu
  — no RMSNorm/QK-norm/SwiGLU, architecturally identical to
  `rope_canon_zhu_transformer` with those three flags off), `hidden_dim=32,
  num_heads=4, num_layers=2, output_dim=64`, `hyper_head.bottleneck_dim=128`.
  Selected over the historically-safer `hidden_dim=64` after a single-seed
  smoke test showed it tracking at least as well (lower loss, higher train
  exact match at matched step counts) — see `smoke_dim6_frozentd_enc32.yaml`
  / `smoke_dim6_frozentd_enc64.yaml`.
- **Hypernetwork encoder, dim=4 ("matched-scale")**: every pipeline
  dimension shrunk to match the dim=4 target it's generating, rather than
  the fixed dim=6-sized width above — `task_encoding.embedding_dim=4`,
  encoder `hidden_dim=4, num_heads=1, num_layers=1, output_dim=4`,
  `hyper_head.bottleneck_dim=8`. This is the standard sizing for dim=4 now
  (was originally a single-seed, `frozen_td`-only exploratory aside — see
  "How small can the hypernetwork go?" below for how this was found).
- **Descriptor arms**: `notd` (no task-identity signal) and `frozen_td`
  (one-hot task embedding, projected in at random init, never trained
  further) — `td` (learned) excluded per project precedent (historically
  bimodal/unstable on its one generalisation win).
- **Hyper head**: `pooling: attention`, full weight generation
  (`lora_adapter: false`, no LoRA); `bottleneck_dim` is 128 at dim=6, 8 at
  dim=4 (see above).
- **Optimizer**: Muon, `muon_lr=0.005, muon_momentum=0.95`, AdamW aux group
  `learning_rate=0.001, weight_decay=0.01, gradient_clip_val=10.0`,
  `batch_size=512`.
- **Budget**: `N_supervision=1` (single pass, not the historical default of
  2), `max_steps=8000, warmup_steps=800` — same 8000-total-optimizer-update
  budget as the `N_supervision=2` convention.
- **Data**: full recipe, no `variants_per_base_task` subsampling.
- **Task set**: 14 categories, matching Experiments 1-3 exactly
  (`1d_recolor_cmp` excluded, deferred to a follow-up).
- **Seeds**: dim=6, 3 per config (original grid, unchanged); dim=4, 5 per
  config (resized + rerun, see "Dim=4 rerun" below). `save_checkpoints:
  true` for dim=4 (a checkpoint exists for every seed); dim=6 still has
  none (`save_checkpoints: false` at the time it was run, not yet rerun).

**Hypernetwork size** (at dim=6, dominated by the `hyper_projection` MLP
`hyper_output_dim(64) -> bottleneck_dim(128) -> total_target_params`, *not*
the encoder — the encoder itself is only ~30K params, ~9-15% of the total;
at dim=4's matched scale every part of the pipeline is small, not just the
projection):

| Target | Total hypernetwork params | vs. every prior hypernetwork experiment (~1.58M) |
|---|---:|---:|
| dim=6 | 336,726 (`notd`) / 337,878 (`frozen_td`) | ~4.7x smaller |
| dim=4 (matched-scale) | 11,360 (`notd`) / 11,432 (`frozen_td`) | ~139x smaller |

## How small can the hypernetwork go? (sizing sweep)

**Question**: given the main grid's 337,878-param encoder+projection setup
already matches individual-training's ceiling, how much of that is actually
necessary? Swept `bottleneck_dim` (the `hyper_projection` MLP's
intermediate width, `hyper_output_dim -> bottleneck_dim ->
total_target_params`), encoder `num_layers`/`hidden_dim`/`output_dim`, and
finally `task_encoding.embedding_dim` itself — each single-seed,
`frozen_td`, dim=6 target unless noted.

| Config | Params | test EM |
|---|---:|---:|
| Original (`bd128/nl2`, dim=6, 3 seeds) | 337,878 | 98.6% |
| `bd64/nl1` | 172,150 | 97.1% |
| `bd64/nl2` | 185,942 | 98.6% |
| `bd32/nl1` | 96,182 | 98.6% |
| `bd32/nl2` | 109,974 | 98.6% |
| `bd16/nl1` | 58,198 | 98.6% |
| `bd16/nl2` | 71,990 | 98.6% |
| `bd8/nl1` | 39,206 | 98.6% |
| `bd8/nl2` | 52,998 | 100% |
| `hd8_od32` (encoder shrunk too, on `bd16/nl1`) | 39,320 | 98.6% |
| `bd16/nl1`, **dim=4 target** | 37,808 | 97.1% |
| **`matched-scale`, dim=4 target** (`embedding_dim=4`, encoder `hidden_dim=4/output_dim=4/num_layers=1`, `bd8` — original exploratory pass, 3 seeds) | **10,156 trainable** | **95.7% mean (95.7/92.9/98.6)** |

This exploratory 3-seed, `frozen_td`-only, no-checkpoints pass is superseded
by the formal one: `matched-scale` is now dim=4's standard sizing (both
arms, 5 seeds, checkpoints) — see "The finding" above and "Dim=4 rerun"
below for the current numbers (93.0% mean `frozen_td`, 57.4% mean `notd`).
The trainable-param count matches exactly (10,156); "Total hypernetwork
params" above and in "The finding" (11,360/11,432) additionally counts
`frozen_td`'s frozen, non-trainable task-indicator projection.

**Finding: essentially flat across a ~33x parameter range.** Every cut —
`bottleneck_dim` 128->8, `num_layers` 2->1, encoder `hidden_dim`/`output_dim`
down to 8/32, even `embedding_dim` 16->4 (which also shrinks the target's own
`input_projection`) — cost nothing. The `hyper_projection` MLP dominates the
parameter count at every size tested (e.g. 69% of the total even at
`bd16/nl1`), not the encoder, so `bottleneck_dim` was the lever worth
pushing hardest.

**A real crash, and what it turned out to mean.** The first `matched-scale`
3-seed attempt (on `torrnode14`) showed 1 CUDA "illegal memory access"
crash and 2 runs that looked like they were ignoring `max_steps` (later
found to be a misread of wandb's own `_step` counter, not
`trainer.global_step` — real batch progress was on track). Re-run cleanly
from scratch on `torrnode7` (fully idle node, no contention): **all 3 seeds
finished with no errors**, including the exact seed that crashed before.
Conclusion: the original crash was node-specific hardware/driver flakiness
on `torrnode14`, not an architecture instability — `matched-scale` is
genuinely stable. (Standing direction since: only launch jobs on
`torrnode8`, `torrnode9`, `torrnode11`-`torrnode15`, not `torrnode10`, not
`torrnode1`-`torrnode7`.)

## Dense generation vs. LoRA adapter (rank sweep)

**Question**: below the `matched-scale` dense floor (10,156 params), does
switching the generation head from full/dense to a LoRA adapter (per-tensor
low-rank factors instead of one shared MLP) buy a further, genuine size
reduction without losing accuracy? 5 seeds each (presentation-grade, not
exploratory), `matched-scale` base (dim=4 target, `embedding_dim=4`, encoder
`hidden_dim=4/output_dim=4/num_layers=1`), `hyper_head.lora_adapter: true`.

| Rank | Params | test EM (5 seeds) | mean ± sd |
|---:|---:|---|---:|
| dense (no LoRA, reference -- the original 3-seed exploratory number; see "Dim=4 rerun" below for the current formal 5-seed one, 93.0%) | 10,156 | 0.957 / 0.929 / 0.986 (3 seeds) | 95.7% |
| 8 | 18,796 | *not run to completion — larger than dense, no point* | — |
| 4 | 11,948 | *not run to completion — larger than dense, no point* | — |
| 2 | 8,524 | 0.900 / 0.929 / 0.929 / 0.900 / 0.871 | 90.6% ± 2.1% |
| 1 | 6,812 | 0.857 / 0.886 / 0.829 / 0.800 / 0.943 | 86.3% ± 4.9% |

Note the target's own weight matrices are mostly 4x4 (one 4x10 output
head), so `min(d_out, d_in) = 4` is *full rank* for this target — rank=8 is
already over-parameterized relative to full rank (and, at 18,796 params,
larger than the dense baseline it was meant to shrink), and rank=4 is the
true full-rank point, also larger than dense. Both were killed early once
their sizes were confirmed larger than the thing they were meant to be an
efficient alternative to; only rank=2 and rank=1 (genuinely smaller than
dense) were run to completion.

**Finding: unlike the encoder/bottleneck sizing sweep's flat plateau, LoRA
rank shows a real, monotonic accuracy/size tradeoff below the dense floor** —
dense (95.7%) > rank=2 (90.6%) > rank=1 (86.3%), with rank=1 also showing
markedly higher seed variance (4.9% sd vs 2.1%). **Decision: keep dense
generation.** At this scale the dense `hyper_projection` MLP is already
smaller than any LoRA configuration worth using (rank>=4 costs *more* than
dense; rank<=2 costs less but genuinely degrades), so there's no regime
where LoRA is the better choice here — LoRA's usual efficiency argument
(parameter savings on large per-tensor matrices) doesn't apply when the
target's tensors are this small to begin with.

## Configs in this folder

All training configs live under `configs/`; everything else here (this
README, the plot scripts) is code, not config. `results*.csv` and the
`embeddings_*.npz` dumps aren't in this folder at all -- they live under
`outputs/results/02_hypernetwork_multitask/`, alongside every run's raw
output, and are regenerated from there by `plot_all.py`/the individual
plot scripts (see "Reproducing" below), not hand-copied in.

`configs/base.yaml` (shared architecture/data/optimizer/logging, dim=6
default) + `configs/dim6_notd.yaml` / `dim6_frozentd.yaml` / `dim4_notd.yaml`
/ `dim4_frozentd.yaml` (main grid). Run via the shared launcher:

```bash
CFG_DIR=experiments/02_hypernetwork_multitask CELL_GLOB="dim*.yaml" SEEDS_OVERRIDE="1 2 3" \
    bash scripts/run_config.sh
```

## Dim=4 rerun: resized to matched-scale, 5 seeds, both arms, checkpoints

The dim=4 pair (`dim4_notd.yaml`/`dim4_frozentd.yaml`) has been resized in
place from the original fixed-width recipe (~200K params, the same encoder
dim=6 still uses) to the matched-scale recipe (~11.4K params, every
pipeline dimension sized to match the dim=4 target itself — see "Method"
above) and rerun clean-slate: 5 seeds, both arms, `save_checkpoints: true`
so a checkpoint exists for every seed (dim=6 still has none; that pair is
unresized and unrerun -- see "Open follow-ups"). This is not a second
variant living alongside the old ~200K dim=4 data -- dim=4 *is* this recipe
now; the old data is archived, not deleted (`*_archive_200k` suffix,
locally and on the cluster), for anyone who wants to compare against it.

Previously, this same rerun (before the resize) was the size experiment 3's
preliminary pass used, and separately the matched-scale point had only ever
been run exploratorily (`frozen_td`-only, 3 seeds, no checkpoints -- see
"How small can the hypernetwork go?" above). This rerun replaces both: a
real, 5-seed, both-arms, checkpointed result at the size that mattered.

**Results.** `frozen_td`: 93.0% mean test exact match (93.6 / 92.2 / 94.9 /
95.5 / 88.9), close to the original 3-seed exploratory number (95.7%).
`notd`: 57.4% mean (54.1 / 53.1 / 55.1 / 67.1 / 57.5) -- this is the first
time `notd` has ever been run at this scale (the exploratory sizing/LoRA
sweeps only tested `frozen_td`). Per-task breakdown (see "The finding"
above) shows `frozen_td` no longer uniformly near-100% at this size --
`move_dp` (55.2%), `mirror` (73.6%), and `flip` (80.0%) show real
degradation even with a task-identity signal, unlike the larger dim=6
encoder's near-total closure of the capacity cliff.

Linear-probe accuracy across all 5 seeds (mean +- 1 s.d.): `frozen_td`
**97.91% +- 2.83pp** (100.0 / 92.9 / 100.0 / 96.7 / 100.0) -- no longer a
clean 100% every seed, unlike dim=6's encoder; `notd` **60.91% +- 4.89pp**
(59.3 / 57.9 / 54.5 / 64.7 / 68.1) -- substantially lower than dim=6's
81-87% range, consistent with a much smaller hypernetwork having
measurably less spare capacity to keep task representations separated
without an identity signal to lean on.

**Reproducing.** One script to run, one to plot everything -- same pattern
as experiment 1:

```bash
bash experiments/02_hypernetwork_multitask/run.sh                        # sequential, 1 GPU
GPUS="0,1,2,3,4,5,6,7" bash experiments/02_hypernetwork_multitask/run.sh  # 8-way parallel

uv run python experiments/02_hypernetwork_multitask/plot_all.py
```

`run.sh` wraps `scripts/run_config.sh` with this experiment's `CFG_DIR` and
a `dim4*.yaml` `CELL_GLOB` baked in (dim6's configs live in the same folder
but are the original, un-rerun 3-seed data, deliberately not part of this).
Skips any `(config, seed)` pair that already has a `results.txt`, so it's
always safe to rerun.

`plot_all.py` rescans `outputs/` and, in one pass, refreshes
`results_per_task_dim4_combined.csv` and `results.csv`, renders both
per-task figures, prints the linear-probe/exact-match summary above, and
renders every seed's cluster maps (PCA/t-SNE/UMAP, `--originals-only`'s
5/category look, one `seed<N>/` folder each). `--from-csv` replots the
per-task figures from the CSV instead of rescanning (the linear-probe
report and cluster maps always rescan -- there's no CSV to fall back to
for those); `--skip-clusters` skips the slow t-SNE/UMAP pass. Nothing here
reads from wandb or the cluster -- it's all local `outputs/`, so anyone who
reruns `run.sh` (here or on their own machine) can plot immediately after,
with no hand-copied file in between. `results.csv`/the per-condition
`embeddings_*.npz` files under `outputs/results/` are themselves just
`outputs/`, gitignored, and regenerated by this same pass -- not
hand-maintained.

Each script is also runnable standalone with the same flags (see each
file's own docstring), e.g. to just refresh the linear-probe report:

```bash
uv run python experiments/02_hypernetwork_multitask/report_linear_probe.py
uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py --dim 4 --seed 3
```

**Cluster plots, any seed.** Every seed's own `embeddings.npz` (per-run,
dumped automatically by `log_embedding_clusters: true`) is reachable
directly, one seed at a time or all five at once (each into its own
`seed<N>/` subfolder, all three projections -- PCA, t-SNE, UMAP):

```bash
uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py --dim 4 --seed 3
uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py --dim 4 --all-seeds
```

The live dump covers the whole validation split (100 points/category for
this experiment's data: 5 non-augmented base tasks x 20 colour variants
each, per `scripts/augment_arc_1d.py`'s `--dev-test-n-permutations`), which
reads a lot busier than the paper figures' original 5/category.
`--originals-only` keeps just the real, non-augmented example per base task
(`task_id % 10000 == 0` -- `augment_task()` always places the unmodified
task first) rather than an arbitrary N of the 100 -- `plot_all.py` above
uses this by default; `--max-per-category N` is the more general cut if a
different, arbitrary count is wanted instead. Both subsample *before*
fitting the projection, not just the display after (t-SNE/UMAP fit a
different-looking embedding on 1,400 points than on 70, so filtering
post-hoc wouldn't reproduce the sparser look):

```bash
uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py \
    --dim 4 --all-seeds --originals-only
uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py \
    --dim 4 --all-seeds --max-per-category 5
```

The encoder-size pre-flight, sizing, and LoRA-rank sweeps described above
("How small can the hypernetwork go?", "Dense generation vs. LoRA adapter")
were exploratory, one-off runs — their configs (`smoke_*.yaml`,
`shrink_*.yaml`, `lora_matched_dim4_r*.yaml`) aren't kept here; the findings
above are the record of what they showed. Recover them from git history
(`git log --all --diff-filter=D -- 'experiments/02_hypernetwork_multitask/configs/shrink_*'`,
similarly for `smoke_*`/`lora_matched_*`) if you need to rerun one.

## Open follow-ups (not part of this experiment)

- **dim=6 still runs the older, larger, fixed-width hypernetwork
  (~337K params) and is still 3 seeds with no checkpoints.** dim=4 is now
  the standard matched-scale (~11.4K param) recipe; dim=6 hasn't been
  resized or rerun to match -- a natural next step, not done here.
- No true floor found below 10,156 trainable params (dense) — haven't
  tried pushing `bottleneck_dim` below 8, or a smaller `num_tasks`/pooling
  change.
- The per-task softening `frozen_td` now shows at dim=4's matched scale
  (`move_dp`/`mirror`/`flip`, see "The finding") hasn't been diagnosed
  further -- is it purely a hypernetwork-capacity effect, or does it
  interact with the target model's own tiny size (1,398 params)?
- `1d_recolor_cmp` re-inclusion (deferred from this 14-task run).
- A low-data cut for the joint hypernetwork case (this run used full data).

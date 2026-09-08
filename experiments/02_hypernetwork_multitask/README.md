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

**With a task-identity signal, a ~200-338K-param hypernetwork matches
individual training's per-task ceiling almost exactly — closing the
capacity cliff Experiment 2 exposed. Without one, the hypernetwork still
helps enormously over direct joint training, but a well-understood, specific
failure mode (move-family conflation) accounts for most of the remaining
gap.**

| Config | test exact match (3 seeds) | mean | linear probe accuracy |
|---|---|---:|---:|
| **dim6, `frozen_td`** | 0.986 / 0.986 / 0.986 | **98.6%** | 100% |
| dim6, `notd` | 0.729 / 0.686 / 0.614 | 67.6% | 81-87% |
| **dim4, `frozen_td`** | 0.943 / 0.957 / 0.971 | **95.7%** | 100% |
| dim4, `notd` | 0.714 / 0.743 / 0.671 | 71.0% | 79-81% |

Reading this against the two experiments it sits between:

| | Individual (Exp. 1) | Joint direct (Exp. 2) | Joint hypernetwork (this experiment) |
|---|---:|---:|---:|
| dim=6, with task ID | 98.9% | 74.0% | **98.6%** |
| dim=6, no task ID | - | 40.9% | 67.6% |
| dim=4, with task ID | 94.6% | 19.7% | **95.7%** |
| dim=4, no task ID | - | 12.0% | 71.0% |

`frozen_td` doesn't just improve on the joint-direct capacity cliff — it
erases it, landing within noise of individual training at both sizes,
remarkably stable across seeds (dim=6 landed on the *exact same* 0.9857
all three runs). `notd` closes most of the direct-training gap (e.g. dim=4:
12.0% -> 71.0%) but not all of it, and per-task breakdown shows exactly why:

```
dim6_notd per-task exact match (one seed):
  1d_move_1p        0.0
  1d_move_2p        0.0
  1d_move_3p        0.0
  1d_move_dp        0.8   (soft everywhere, not a notd-specific issue)
  everything else   1.0
```

Without a task-identity signal, the whole move family (`1d_move_1p/2p/3p`)
collapses together — exactly the conflation mechanism `00_overview.md`
section 3 already documented for direct training
(`arc1d_hypermodel_disentanglement`'s original motivation for adding a
descriptor at all). `frozen_td` fixes precisely this, matching the
project's standing explanation rather than a new, unexplained failure mode.
At dim=4 the same pattern holds but is more diffuse (`mirror`/`scale_dp`
also soften alongside the move family), consistent with a smaller model
having less slack to partially compensate.

**Disentanglement matches the exact-match story.** Linear-probe accuracy
(predicting task category from the hypernetwork's pooled task
representation) is a clean 100% for every `frozen_td` run and 79-87% for
every `notd` run — the representation is genuinely worse at separating
tasks without an identity signal, not just harder to decode from. See `outputs/figures/02_hypernetwork_multitask/cluster_dim6_paired_tsne.png`
(regenerate with `plot_embedding_clusters.py --projections t-SNE`).

## Method

- **Target model**: `rope_canon_looped_transformer`, `hidden_dim=6`
  (primary, 2,444 params trained directly in Experiment 1) and `hidden_dim=4`
  (secondary, 1,398 params) — flat (`n_loops=1`), single pass, identical to
  Experiment 1's recipe, now hypernetwork-generated instead of directly
  trained.
- **Hypernetwork encoder**: `rope_canon_transformer` (plain, non-Zhu — no
  RMSNorm/QK-norm/SwiGLU, architecturally identical to
  `rope_canon_zhu_transformer` with those three flags off), `hidden_dim=32,
  num_heads=4, num_layers=2, output_dim=64`. Selected over the
  historically-safer `hidden_dim=64` after a single-seed smoke test showed
  it tracking at least as well (lower loss, higher train exact match at
  matched step counts) — see `smoke_dim6_frozentd_enc32.yaml` /
  `smoke_dim6_frozentd_enc64.yaml`.
- **Descriptor arms**: `notd` (no task-identity signal) and `frozen_td`
  (one-hot task embedding, projected in at random init, never trained
  further) — `td` (learned) excluded per project precedent (historically
  bimodal/unstable on its one generalisation win).
- **Hyper head**: `pooling: attention`, `bottleneck_dim: 128`, full weight
  generation (`lora_adapter: false`, no LoRA).
- **Optimizer**: Muon, `muon_lr=0.005, muon_momentum=0.95`, AdamW aux group
  `learning_rate=0.001, weight_decay=0.01, gradient_clip_val=10.0`,
  `batch_size=512`.
- **Budget**: `N_supervision=1` (single pass, not the historical default of
  2), `max_steps=8000, warmup_steps=800` — same 8000-total-optimizer-update
  budget as the `N_supervision=2` convention.
- **Data**: full recipe, no `variants_per_base_task` subsampling.
- **Task set**: 14 categories, matching Experiments 1-3 exactly
  (`1d_recolor_cmp` excluded, deferred to a follow-up).
- **Seeds**: 3 per config (main grid), 1 for the two encoder-size smoke
  tests. `save_checkpoints: false` throughout (no checkpoint saved for any
  run — retrieving one requires a rerun with the flag on).
- 4 main-grid jobs x 3 seeds = 12 total, all finished cleanly, 0 failures.

**Hypernetwork size** (dominated by the `hyper_projection` MLP
`hyper_output_dim(64) -> bottleneck_dim(128) -> total_target_params`, *not*
the encoder — the encoder itself is only ~30K params, ~9-15% of the total):

| Target | Total hypernetwork params | vs. every prior hypernetwork experiment (~1.58M) |
|---|---:|---:|
| dim=6 | 336,726 (`notd`) / 337,878 (`frozen_td`) | ~4.7x smaller |
| dim=4 | 200,244 (`notd`) / 201,396 (`frozen_td`) | ~7.9x smaller |

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
| **`matched-scale`, dim=4 target** (`embedding_dim=4`, encoder `hidden_dim=4/output_dim=4/num_layers=1`, `bd8` — 3 seeds) | **10,156** | **95.7% mean (95.7/92.9/98.6)** |

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
| dense (no LoRA, reference) | 10,156 | 0.957 / 0.929 / 0.986 (3 seeds) | 95.7% |
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
README, the plot scripts, `results*.csv`, the `embeddings_*.npz` dumps) is
code/output, not config.

`configs/base.yaml` (shared architecture/data/optimizer/logging, dim=6
default) + `configs/dim6_notd.yaml` / `dim6_frozentd.yaml` / `dim4_notd.yaml`
/ `dim4_frozentd.yaml` (main grid, `results.csv`). Run via the shared
launcher:

```bash
CFG_DIR=experiments/02_hypernetwork_multitask CELL_GLOB="dim*.yaml" SEEDS_OVERRIDE="1 2 3" \
    bash scripts/run_config.sh
```

All results pulled from wandb (`fjfehr/arc1d_v2_hypernetwork_multitask`).

## Dim=4 rerun, 5 seeds (checkpoints + per-seed clusters)

The dim=4 pair (`dim4_notd.yaml`/`dim4_frozentd.yaml`) has since been rerun
clean-slate at 5 seeds, with `save_checkpoints: true` (was `false`) so a
checkpoint now exists for every seed -- unblocking experiment 3, which was
blocked on exactly this. `frozen_td`: 94.8% mean test exact match (95.5 /
94.9 / 92.3 / 96.1 / 95.2), close to the original 3-seed 95.7%. `notd`:
65.3% mean (51.8 / 68.6 / 67.7 / 65.1 / 73.5), a bit below the original
71.0% with wider seed variance, consistent with the same failure mode
already documented above (move-family conflation), not a new problem.

Reproduce it with two commands, run then plot:

```bash
bash experiments/02_hypernetwork_multitask/run.sh                        # sequential, 1 GPU
GPUS="0,1,2,3,4,5,6,7" bash experiments/02_hypernetwork_multitask/run.sh  # 8-way parallel

uv run python experiments/02_hypernetwork_multitask/plot_per_task_combined.py --outputs-dir outputs
```

`run.sh` wraps `scripts/run_config.sh` with this experiment's `CFG_DIR` and
a `dim4*.yaml` `CELL_GLOB` baked in (dim6's configs live in the same folder
but are the original, un-rerun 3-seed data, deliberately not part of this).
Skips any `(config, seed)` pair that already has a `results.txt`, so it's
always safe to rerun.

Every seed's own `embeddings.npz` (per-run, dumped automatically by
`log_embedding_clusters: true`) is reachable directly, one seed at a time
or all five at once (each into its own `seed<N>/` subfolder, all three
projections -- PCA, t-SNE, UMAP):

```bash
uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py --dim 4 --seed 3
uv run python experiments/02_hypernetwork_multitask/plot_embedding_clusters.py --dim 4 --all-seeds
```

`plot_per_task.py`/`plot_per_task_combined.py` can now rescan live
`outputs/` too (previously they only read the committed, 3-seed CSV):

```bash
uv run python experiments/02_hypernetwork_multitask/plot_per_task_combined.py --outputs-dir outputs
```

The encoder-size pre-flight, sizing, and LoRA-rank sweeps described above
("How small can the hypernetwork go?", "Dense generation vs. LoRA adapter")
were exploratory, one-off runs — their configs (`smoke_*.yaml`,
`shrink_*.yaml`, `lora_matched_dim4_r*.yaml`) aren't kept here; the findings
above are the record of what they showed. Recover them from git history
(`git log --all --diff-filter=D -- 'experiments/02_hypernetwork_multitask/configs/shrink_*'`,
similarly for `smoke_*`/`lora_matched_*`) if you need to rerun one.

## Open follow-ups (not part of this experiment)

- No true floor found below 10,156 params (dense) — haven't tried pushing
  `bottleneck_dim` below 8, or a smaller `num_tasks`/pooling change.
- `notd` hasn't been retested at any of the shrunk sizes — the sizing and
  LoRA sweeps only used `frozen_td`.
- `1d_recolor_cmp` re-inclusion (deferred from this 14-task run).
- A low-data cut for the joint hypernetwork case (this run used full data).
- Retrieving a checkpoint requires a rerun with `save_checkpoints: true`
  for at least one cell — none exists from this run.

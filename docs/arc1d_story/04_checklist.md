# Checklist: gaps to resolve before/during the next round

This is the handoff artefact for the follow-up planning round (a
from-scratch rerun for consistency). Nothing in this table has been acted
on beyond verification during this stocktake - no training was launched.
See [`00_overview.md`](00_overview.md) for the narrative these items support.

**Update**: every row in this table now has an explicit disposition in
[`05_round2_plan.md`](05_round2_plan.md)'s "Consistency check against the
stocktake" section, which is the actual from-scratch rerun plan this
checklist was written to feed.

| # | Item | Current status | Confidence | Action needed | Priority |
|---|---|---|---|---|---|
| 1 | Re-run `arc1d_uniform_ablation` (T1-T7, L1-L6) | Ran entirely before the 2026-07-13 `max_steps`/`N_supervision` fix; every `N_supervision=2` cell (everything but T1) trained on half its intended batches. Never re-run since. | Low - this is the foundational backbone choice every later experiment inherits | Full rerun post-fix, 2 widths x 17 tasks x 5 seeds x 13 cells (T1-T7, L1-L6) | **Highest** |
| 2 | Re-run/replace bug-affected hypernetwork experiments: `arc1d_hypermodel_looped`, `arc1d_hypermodel_looped_recolor` (N_sup=4 cells especially), `arc1d_hypermodel_looped_mix11` | Also predate the fix | Low for N_sup=4 cells specifically; the N_sup=2 cells lose less (half of an already-short schedule, but still not the intended budget) | Rerun once the unified post-fix recipe (item 7) is designed, rather than patching these individually | High |
| 3 | Run `arc1d_lowdata_baseline` (the no-hypernetwork per-task control) | **Correction from this stocktake: this is NOT unrun.** All 135 planned jobs are finished on wandb; only the local filesystem looked empty. Real numbers now in [`03_experiments.md`](03_experiments.md#arc1d_lowdata_baseline) and [`00_overview.md`](00_overview.md#7-low-data-augmentation). | High - verified directly via wandb API this session | None needed to have the result; consider syncing the wandb data to local `results.txt` files so it isn't invisible to local tooling again | Closed (was flagged as a gap, turned out already done) |
| 4 | Verify wandb for experiments with no local results | Done this session for: `arc1d_hypermodel_looped_rope_canon_generalization` (30 runs, 28 finished - 2 seeds/cell not 3, 2 cells with only 1 seed), `arc1d_lowdata` (30 runs, 25 finished), `arc1d_lowdata_lowrank` (17 runs, 14 finished), `arc1d_hypermodel_looped_mix11` (7 runs, 3 finished + 3 crashed + 1 killed), `arc1d_hypermodel_looped` (113 runs, 75 finished), `arc1d_hypermodel_looped_rope_canon_muon_sweep` (230 runs, 123 finished, 99 failed, 8 crashed - more incomplete than the README's "cancelled after 99/216" framing suggests), `arc1d_hypermodel_looped_rope_canon_muon_moveablation` (42 runs, only 17 finished, 17 failed, 8 crashed), `arc1d_lowdata_baseline` (135/135 finished, see item 3) | High for all of these - wandb API confirmed reachable (`wandb` CLI + `~/.netrc` present, entity `fjfehr`) and queried directly | Numbers now folded into `03_experiments.md`; `muon_sweep`/`muon_moveablation`'s high failure rates are worth a root-cause look before relying on their partial results in a paper | Done this session |
| 5 | Resolve `n_loops` (4 vs. 8) inconsistency | `arc1d_uniform_ablation` recommends n_loops=8 (dim=16)/16 (dim=32), no skip; almost every downstream hypernetwork experiment actually uses n_loops=4, inherited from a different sweep (the "flip+recolor" one), not from the capacity story. Only `mix11_td` and `lora_adapter/base.yaml` use n_loops=8. Full detail: [`02_architecture.md`](02_architecture.md#2-inconsistency-n_loops). | High - confirmed by grepping every downstream config | Decide the single canonical value for the next round's unified recipe (item 7) | High |
| 6 | Resolve `lora_adapter_rank` (4 vs. 8) inconsistency | rank=8 is the "committed default" and the rank-sweep winner; `vae_disentanglement` and `compositional_generalization` both explicitly use rank=4, with their own header comments acknowledging this deviates from the default. Full detail: [`02_architecture.md`](02_architecture.md#4-inconsistency-lora_adapter_rank). | High - confirmed by grepping every downstream config | Decide the single canonical value; note this affects two experiments whose numbers feed sections 4 and 5 of the overview | High |
| 7 | Design one unified rerun recipe | Not designed yet - explicitly deferred to the next round | - | Fixed target backbone (pending items 1 and 5), fixed data recipe (`arc_1d_looped_augmented`), fixed 15-task working set, fixed optimizer (Muon, LoRA-heads excluded from Muon, batch_size=2048 - the `muon_sweep`/`muon_moveablation` winners), fixed LoRA rank (item 6), fixed seed count (recommend 3, given how much seed 6/6 above and 30 seed-variance showed up in generalisation, item 6 of the overview) | Next round |
| 8 | `arc1d_hypermodel_disentanglement` data-recipe mismatch | Uses `data/arc_1d_all_tasks_augmented`, an older/different recipe than the canonical `arc_1d_looped_augmented` used everywhere else | High | **No rerun** - per explicit decision, document only (done, see [`01_data.md`](01_data.md)). The finding it supports (descriptor fixes similar-task conflation) is qualitative, not a precise number being relied on. | Closed |
| 9 | Beta-VAE disentanglement sweep | 22 arms, finished, single seed each | High | **Excluded from the write-up by design** (per explicit decision) - noted as tried, only its plain notd/td/frozen_td baseline reference numbers kept (see overview section 4) | Closed |
| 10 | Stray `outputs_diag_seed1_*.jsonl` files in repo root | 4 files, ~4.5-4.7MB each, dated 2026-07-14, unrelated gradient-clipping/loss-spike diagnostics from a separate investigation - not connected to this story | High | Flag as cleanup candidates; **don't delete without asking** | Low, no rush |
| 11 | `arc1d_uniform_ablation` L5/L6 raw run directories missing locally | README reports full L5/L6 numbers (n_loops=16/32) but no local `results.txt` exists for those 340 jobs - presumably fetched/summarised then not retained | Medium - numbers are in the README, just not independently re-derivable from local files | Will be moot once item 1's full rerun happens; until then, treat L5/L6 numbers as README-sourced only | Low (folds into item 1) |
| 12 | `arc1d_hypermodel_looped_rope_canon_rank_sweep` and `_lr_sweep` partial local coverage | `rank_sweep`: 15/25 local results. `lr_sweep`: 33/~35-37 local results. | Medium | Check wandb for the remainder if these numbers matter for the paper beyond their already-stated headline findings (rank=8 wins; the lr_sweep leader used downstream) | Low, only if precision on these specific sweeps is needed |

## Notes for whoever picks up the next round

- Every number in [`00_overview.md`](00_overview.md) and
  [`03_experiments.md`](03_experiments.md) traces to either a local
  `results.txt`, a README's own stated text, or a wandb query run during
  this session (all flagged explicitly where used) - nothing here is
  extrapolated or estimated.
- The two "surprising" findings worth carrying into the paper framing
  deliberately, not just filing away: (a) `td`'s bimodal seed variance on
  the one category where generalisation succeeds at all (section 6), and
  (b) the low-data baseline already solving 12/15 tasks at ~100% with no
  hypernetwork at all (section 7) - both complicate a simple "the
  hypernetwork/frozen descriptor is strictly better" narrative in ways that
  make the honest story more interesting, not less.

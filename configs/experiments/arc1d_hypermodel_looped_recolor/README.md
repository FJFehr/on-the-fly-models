# arc1d_hypermodel_looped_recolor

## Goal

`1d_flip` and the three recolor tasks (`1d_recolor_cmp/cnt/oe`) are the only tasks in
`arc1d_hypermodel_looped`'s phase-1 sweep that don't hit ~100% val exact match — every
other task category already works. This experiment targets exactly these four, testing
whether more hypernetwork capacity (deeper internal loops, more external supervision, skip
connections) closes the gap on the real task, rather than changing the task itself.

## Why the data isn't the problem

Rigorously verified (not assumed) that the recolor tasks have **no support/query data
bug**: for each recolor category, derived the true rule from the 3 support pairs only
(`recolor_oe`: run-length parity; `recolor_cnt`: exact run length; `recolor_cmp`: every run
tied for the *maximum* length gets one colour, all others get another), then checked that
rule against the query's actual output across 200 real task instances per category (600
total) on `data/arc_1d_looped_augmented`. **200/200 OK for all three — the support pairs
always fully and correctly determine the query.**

Output colours do vary across different task instances (`1d_recolor_oe` showed 17 distinct
colour pairs in a small sample) — but that's the *correct*, intended shape of the task:
colour identity is part of what must be inferred from support each time, same as every
other task category in this repo relies on colour-permutation augmentation to force
rule-based generalization instead of colour memorization (see the main `README.md`'s Data
Augmentation section). A canonical-colour variant (fixing colours to be the same across all
instances) was built and tested as a diagnostic, then deliberately dropped — the model
should be able to handle the colour variance as-is, since that variance is the real task.
This experiment stays on `data/arc_1d_looped_augmented` throughout and only varies model
capacity.

`1d_flip` is a separate, likely-unrelated issue: it already fails when trained directly
(0.0% at n_loops=4, 11.6% at n_loops=8, dim=16, `arc1d_uniform_ablation`) — a
backbone/architecture limitation (full-sequence reversal needs each position to know its
distance from the *far end*, which RoPE's relative-only position encoding doesn't give for
free), not specific to the hypernetwork. Included here anyway since "more capacity might
help" applies to it too, and we already know the others (all non-flip, non-recolor tasks)
are capable at the phase-1 settings.

## Sweep

`n_loops ∈ {4, 8, 16}` × `N_supervision ∈ {2, 4}` × `skip ∈ {none, block+loop}` (the
capacity story's best-performing skip combination, matching T7/L4) = 12 configs/task × 4
tasks (`1d_flip`, `1d_recolor_cmp`, `1d_recolor_cnt`, `1d_recolor_oe`) = 48 total, all on
`data/arc_1d_looped_augmented`. The `n_loops=4, N_supervision=2, no-skip` cell in each task
dir is exactly the existing phase-1 config, included in the naming scheme for completeness,
not a new result.

```bash
python scripts/gen_hypermodel_looped_recolor_configs.py
bash scripts/run_hypermodel_looped_recolor.sh        # 8 GPUs, single seed, skips completed
bash scripts/run_hypermodel_looped_recolor.sh 4
```

## Reading results

Compare `val_query_exact_match` across the 3×2×2 grid per task against the phase-1 baseline
(`n_loops=4, N_supervision=2, no-skip`). A jump from deeper loops alone isolates internal
recursion depth as the lever; a jump from more `N_supervision` alone isolates external
supervision; a jump from skip connections alone isolates that mechanism; combinations
tell us whether they compound. If nothing in the grid closes the gap, that's a real finding
about a capacity ceiling at this model size, not evidence to change the task's data.

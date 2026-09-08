# Experiment 1: multi-task capacity

**Question**: can one small model learn all 14 ARC-1D tasks *jointly* (a
single shared weight set, no per-task model), with and without a per-task
identity signal? Full background: `docs/arc1d_story/06_phase1_findings.md`.

## The plot

![Capacity cliff](../../../outputs/figures/01_multitask_capacity/capacity_cliff.png)

*(Not committed, regenerate with `uv run python plot_all.py` from this
folder, or see "Figures" in `experiments/README.md`.)*

X-axis is total parameter count (the same RoPE+Canon architecture at four
widths, `hidden_dim` 4/6/10/14, `embedding_dim` fixed at 10 throughout for
comparability), log scale. Y-axis is test-split exact match accuracy. Solid
lines are joint training (one shared model across all 14 categories); the
dotted line is individual training (one model per task, no sharing). Colour
separates the two joint arms: light purple has no task signal, dark purple
has a per-task identity embedding. Shaded bands are &plusmn;1 s.d. across
seeds.

All raw per-seed results are aggregated into
`outputs/results/01_multitask_capacity/results.csv`, not committed (see
"Figures and results" in `experiments/README.md`), so it must exist locally
(either from your own training runs, or a snapshot force-added at a paper
milestone) before the plot can be rebuilt. `plot_all.py` (see "Running"
below) is the easiest way to do that; the underlying script still works
standalone if you only want this one figure:

```bash
# rescan a live outputs/<project>/*/results.txt tree, write the CSV, then plot
uv run python experiments/01_multitask_capacity/plot_capacity_cliff.py \
    --outputs-dir outputs

# once the CSV exists, replot from it directly (no outputs/ tree needed)
uv run python experiments/01_multitask_capacity/plot_capacity_cliff.py
```

## The finding

Individual training saturates almost immediately: 92.6% at 1,398 params,
98.3% at 2,444, and 98.6-98.7% from 5,400 params up. Joint training does
not, even with task identity, though the gap closes steadily with scale:

| params | Individual | Joint, no task ID | Joint + task-ID embedding |
|---:|---:|---:|---:|
| 1,398 (dim=4) | 92.6% (n=5) | 13.4% (n=5) | 21.2% (n=5, unstable: 5.5-44.6%) |
| 2,444 (dim=6) | 98.3% (n=5) | 40.7% (n=5) | 73.2% (n=5) |
| 5,400 (dim=10) | 98.7% (n=5) | 57.5% (n=5) | 93.2% (n=5) |
| 9,508 (dim=14) | 98.6% (n=5) | 65.6% (n=5) | 97.5% (n=5) |

The sharpest single-size demonstration is still at **2,444 params**:
individual training is already close to saturated there (98.3%), but joint
training with task identity is still 25 points behind (73.2%). Task
identity alone cannot close that gap at this size: there is a real
model-capacity regime that solves every task *alone* but cannot hold them
*together*, even when told which task it's looking at. By 9,508 params
though, the task-ID arm has nearly caught up (97.5% vs. 98.6%), while the
no-task-ID arm is still far short (65.6%): a task-identity signal genuinely
buys headroom that raw capacity alone reaches only slowly.

This motivates the hypernetwork: a mechanism that generates per-task
weights, rather than sharing one fixed set across tasks.

## Which tasks actually fail

![Per-task breakdown at dim=6](../../../outputs/figures/01_multitask_capacity/per_task_dim6.png)

*(Not committed, regenerate with `uv run python plot_per_task.py --dim 6`
from this folder, or `plot_all.py` for every size at once.)*

The aggregate 73.2% (joint + task ID, dim=6) hides a sharp split: most
tasks are fully recovered, but a few collapse completely. `Flip` is the
starkest case: 100% individually, **4%** in both joint arms, task ID or
not. `Mirror` and `Move Dynamic` show task ID recovering *some* signal but
nowhere near the individual ceiling. Most other tasks (`Denoise`, `Pattern
Copy`, `Move 1 Pixel`, ...) are at or near 100% in every condition: the
joint-training penalty is concentrated in a handful of tasks, not spread
evenly across all 14. The same pattern holds at every other size (per-task
plots for dims 4/10/14 are available via the commands below); the sizes
mainly shift how far the recoverable tasks climb, not which tasks stay
stuck.

Only the validation split has a per-task breakdown (`on_validation_epoch_end`
accumulates `val_query_exact_match_by_task_<category>`; there's no test-time
equivalent), so this chart uses validation exact match throughout, for all
three conditions: internally consistent, but not directly comparable
split-wise to the capacity-cliff plot above (which uses test).

```bash
uv run python experiments/01_multitask_capacity/plot_per_task.py --dim 6
uv run python experiments/01_multitask_capacity/plot_per_task.py \
    --outputs-dir outputs --dim 6   # refresh results_per_task.csv first
```

## Canon ablation

**Question**: does the RoPE+Canon backbone's Canon layer (short causal
convolutions applied at various points in the block) meaningfully boost
accuracy, and is that boost largest at the smallest sizes? `configs/nocanon/`
mirrors every Joint and Individual config in this experiment with
`canon_set: ''` instead of `'ABCD'` (pure RoPE, no Canon), same architecture
width, optimizer, and step budget otherwise, so the two are directly
comparable size for size.

![Canon ablation capacity cliff](../../../outputs/figures/01_multitask_capacity/capacity_cliff_canon_ablation.png)

*(Not committed, regenerate with `uv run python plot_canon_ablation.py
--outputs-dir outputs` from this folder.)* Same colour-by-condition scheme
as the main capacity-cliff plot; line style now carries the ablation
instead: solid is canon, dashed is no canon.

| condition | 1.4K (dim=4) | 2.4K (dim=6) | 5.4K (dim=10) | 9.5K (dim=14) |
|---|---:|---:|---:|---:|
| Individual, canon | 92.6% | 98.3% | 98.7% | 98.6% |
| Individual, no-canon | 37.5% | 78.9% | 87.9% | 95.8% |
| → Canon boost | **+55.1pp** | +19.4pp | +10.8pp | +2.8pp |
| Joint notd, canon | 13.4% | 40.7% | 57.5% | 65.6% |
| Joint notd, no-canon | 0.9% | 5.7% | 32.9% | 41.2% |
| → Canon boost | +12.5pp | +35.0pp | +24.6pp | +24.4pp |
| Joint td, canon | 21.2% | 73.2% | 93.2% | 97.5% |
| Joint td, no-canon | 1.2% (0.5-2.9) | 23.7% | 76.4% | 88.5% |
| → Canon boost | +20.0pp | +49.5pp | +16.8pp | +9.0pp |

Canon helps everywhere, substantially, in every condition and size tested.
For the Individual arm the hypothesis holds cleanly: the boost shrinks
monotonically as size grows (55.1 to 2.8pp), exactly "biggest where
capacity is scarcest." For the two Joint arms it's more nuanced: the boost
*peaks at dim=6*, not dim=4. A plausible read is a floor effect rather than
Canon mattering less at the smallest size: without Canon, joint training at
dim=4 is already so capacity-starved (0.9%/1.2%, barely above the
near-random baseline) that there's little room left for Canon to lift it
further.

Run the same way as the main sweep, with a narrower `CFG_DIR`:

```bash
CFG_DIR=experiments/01_multitask_capacity/configs/nocanon PROJECT=01_multitask_capacity \
    SEEDS_OVERRIDE="1 2 3 4 5" GPUS="0,1,2,3,4,5,6,7" bash scripts/run_config.sh
```

`PROJECT` must be set explicitly here: `CFG_DIR`'s own basename is
`nocanon`, which would otherwise scatter results under
`outputs/nocanon/` instead of alongside everything else in
`outputs/01_multitask_capacity/`. Then regenerate the comparison above:

```bash
uv run python experiments/01_multitask_capacity/plot_canon_ablation.py --outputs-dir outputs
```

## Optimizer ablation

**Question**: how much of this experiment's story depends on Muon
specifically? `configs/adamw/` mirrors every Joint and Individual config
with `optimizer: AdamW` instead of Muon, Canon left enabled, so this
isolates the optimizer choice alone.

![Optimizer ablation capacity cliff](../../../outputs/figures/01_multitask_capacity/capacity_cliff_optimizer_ablation.png)

*(Not committed, regenerate with `uv run python plot_optimizer_ablation.py
--outputs-dir outputs` from this folder.)* Same colour-by-condition scheme
as the other capacity-cliff plots; solid is Muon, dashed is AdamW. Learning
rates and scheduler are below, not on the figure itself.

| condition | 1.4K (dim=4) | 2.4K (dim=6) | 5.4K (dim=10) | 9.5K (dim=14) |
|---|---:|---:|---:|---:|
| Individual, Muon | 92.6% | 98.3% | 98.7% | 98.6% |
| Individual, AdamW | 83.9% | 96.6% | 98.2% | 99.2% |
| Joint notd, Muon | 13.4% | 40.7% | 57.5% | 65.6% |
| Joint notd, AdamW | 4.4% | 23.5% | 43.1% | 50.1% |
| Joint td, Muon | 21.2% | 73.2% | 93.2% | 97.5% |
| Joint td, AdamW | 18.1% | 65.2% | 87.0% | 94.4% |

Muon beats AdamW in every cell except one (Individual at dim=14, where
AdamW edges ahead slightly, 99.2% vs. 98.6% -- likely noise given the
seed-to-seed variance elsewhere). The gap is largest at the smallest sizes
and for joint training, narrowing as capacity grows. Worth reading with a
caveat though: the AdamW learning rate (0.001) is base.yaml's existing
RAdam-tuned default, not a value tuned for AdamW on this task, so part of
this gap may reflect that rather than a fundamental Muon-vs-AdamW
difference -- see `gen_adamw_configs.py`'s docstring.

Muon and AdamW's parameter groups aren't directly comparable one-to-one:
Muon configs actually run two optimizers together (a Muon group for hidden
`nn.Linear` weights at `muon_lr=0.005, muon_momentum=0.95`, plus an AdamW
*aux* group for everything else -- embeddings, norms, Canon conv weights,
biases, the final output layer -- at `learning_rate=0.0005`), while the
AdamW-ablation configs use a single AdamW optimizer for every parameter at
`learning_rate=0.001`. Both share `weight_decay=0.01` and the same
schedule: 200-step linear warmup, then cosine decay to 0 over the
remaining `max_steps=8000` steps.

Run the same way as the canon ablation, with a narrower `CFG_DIR`:

```bash
CFG_DIR=experiments/01_multitask_capacity/configs/adamw PROJECT=01_multitask_capacity \
    SEEDS_OVERRIDE="1 2 3 4 5" GPUS="0,1,2,3,4,5,6,7" bash scripts/run_config.sh
```

```bash
uv run python experiments/01_multitask_capacity/plot_optimizer_ablation.py --outputs-dir outputs
```

## Method

- **Individual**: one model per task category, no cross-task sharing,
  generated by `gen_individual_configs.py` into `configs/individual/`
  (14 tasks x 4 dims = 56 configs, one directory per task). Previously
  sourced from two now-gone locations outside this experiment (a deleted
  `arc1d_v2_minimal_size` sweep for dim=4/6, `legacy/configs/experiments/
  arc1d_v2_backbone_capacity/` for dim=10 at 3 seeds); this experiment is
  now self-contained, all four sizes, 5 seeds throughout.
- **Joint, no task ID** (`notd*.yaml` here): one model trained across all 14
  categories at once, no task signal.
- **Joint + task ID** (`td*.yaml` here): same joint setup, plus a per-task
  `nn.Embedding(18, 10)` looked up via `TASK_CATEGORY_INDEX` and added into
  the token embeddings before the backbone
  (`models/direct_supervised_lightning.py`,
  `task_encoding.use_task_embedding: true`), mirroring the hypernetwork's
  `task_indicator_proj` (`models/hypermodel.py`).
- All test-split exact match (`train_split=train`, `val_split=dev` for
  monitoring only, no checkpoint selection since `save_checkpoints: false`
  everywhere in this project, `test_split=test` reported once at the end).
- RoPE + Canon backbone, Muon optimizer (`muon_lr=0.005`,
  `muon_momentum=0.95`), flat (`n_loops=1`), `N_sup=1`, `max_steps=8000`.
- 5 seeds per condition/size, throughout, including Individual at every
  size.

## Configs in this folder

All training configs live under `configs/`; everything else here (this
README, the plot scripts, `results*.csv`) is code/output, not config.

| File | Condition | Size | Status |
|---|---|---|---|
| `configs/notd.yaml` / `configs/td.yaml` | Joint, no ID / with ID | dim=10 (5,400 params) | done, 5 seeds |
| `configs/notd_dim6.yaml` / `configs/td_dim6.yaml` | Joint, no ID / with ID | dim=6 (2,444 params) | done, 5 seeds |
| `configs/notd_dim4.yaml` / `configs/td_dim4.yaml` | Joint, no ID / with ID | dim=4 (1,398 params) | done, 5 seeds |
| `configs/notd_dim14.yaml` / `configs/td_dim14.yaml` | Joint, no ID / with ID | dim=14 (9,508 / 9,688 params) | done, 5 seeds |
| `configs/individual/<task>/dim{4,6,10,14}.yaml` | Individual (56 configs, one per task x dim) | all 4 sizes | done, 5 seeds |
| `configs/nocanon/joint/{notd,td}_dim{4,6,10,14}.yaml` | Joint, canon ablation | all 4 sizes | done, 5 seeds |
| `configs/nocanon/individual/<task>/dim{4,6,10,14}.yaml` | Individual, canon ablation (56 configs) | all 4 sizes | done, 5 seeds |
| `configs/overfit/td_overfit.yaml` | sanity check for the task-embedding code path | dim=10, 3 categories | n/a |

Run names dropped the vestigial `v2_`/`multitask` prefix this experiment
originally carried from before the paper-repro reorg: `joint_{notd,td}_dim
{4,6,10,14}_seed{N}`, `individual_dim{4,6,10,14}_<task>_seed{N}`, with a
`nocanon` tag inserted for the ablation arms (`joint_nocanon_...`,
`individual_nocanon_...`).

## Running

Two commands reproduce this experiment end to end, run and plot:

```bash
bash experiments/01_multitask_capacity/run.sh                        # sequential, 1 GPU
GPUS="0,1,2,3,4,5,6,7" bash experiments/01_multitask_capacity/run.sh  # 8-way parallel

uv run python experiments/01_multitask_capacity/plot_all.py --outputs-dir outputs
```

`run.sh` wraps `scripts/run_config.sh` with this experiment's `CFG_DIR`
baked in; `CFG_DIR` recurses, so one call sweeps every config under this
experiment's whole tree: the 45 Joint/overfit jobs, the 280 Individual jobs,
and both ablations (320 Canon, 320 optimizer) together, 965 jobs total at 5
seeds. Skips any `(config, seed)` pair that already has a `results.txt`, so
it's always safe to rerun: backfill missing seeds, resume after an
interruption, or add an ablation you haven't run yet (whatever's already
done is skipped automatically). To run just one piece instead, narrow
`CFG_DIR` to its subfolder (`configs/individual`, `configs/nocanon`,
`configs/adamw`) and call `scripts/run_config.sh` directly with
`PROJECT=01_multitask_capacity` set explicitly (see "Canon ablation" above
for a worked example) -- the subfolder's own basename would otherwise become
the project name.

`plot_all.py` rescans `outputs/`, refreshes all four CSVs, and renders all
eight figures (capacity-cliff, a per-task breakdown per size, and both
ablations) in one pass -- the single entry point for "just finished
training, rebuild everything." `plot_capacity_cliff.py`/`plot_per_task.py`/
`plot_canon_ablation.py`/`plot_optimizer_ablation.py` still work standalone
(see their own sections above) if you only want one figure refreshed.

# arc1d_hypermodel_looped_rope_canon_capacity_baseline

## Goal

`arc1d_hypermodel_looped_rope_canon_grid` (layers x clip x task-descriptor, 54 jobs, all
complete) found that at `gradient_clip_val=10` with the task descriptor on, `num_layers=8`
gives a real stability improvement over `num_layers=4` (mean exact-match 0.864 vs. 0.856, but
std 0.009 vs. 0.027, roughly 3x tighter). Pooled across every `clip=10`/descriptor-on run,
essentially every task is solved (mean >= 0.93) except:

- `1d_recolor_cnt` and `1d_recolor_oe`: stuck at a literal `0.000` in every single run
  regardless of layers or clip. No signal at all, a qualitatively different (likely
  structural) failure, not a capacity question.
- `1d_move_dp`: 0.711 mean (0.6-0.8 range), never zero. Partial but not catastrophic.

So this experiment fixes `gradient_clip_val=10` (no longer swept) and drops
`1d_recolor_cnt`/`1d_recolor_oe` from the task list (15 tasks remain, `1d_move_dp` stays in
as the task most likely to actually show a layers=4-vs-8 capacity difference).

Separately, `arc1d_hypermodel_looped_rope_canon_mechanism_ablation` found removing the task
descriptor (`hyper_head.num_tasks: null`) cost a real chunk of accuracy (0.629 vs. 0.746 mean
exact-match), but that was measured at the old settings (`clip=5`, `layers=4`, the harder
17-task list including the now-excluded recolor tasks). This experiment re-measures that gap
at the improved settings, crossed with `layers` in `{4, 8}`, and, for the no-descriptor arms
only, doubled training length to check whether the shortfall is partly just undertraining.

## Arms

| Arm | `num_layers` | `num_tasks` | `max_steps` | `warmup_steps` |
|---|:---:|:---:|:---:|:---:|
| `arm_layers4_td` | 4 | 18 | 4000 | 400 |
| `arm_layers8_td` | 8 | 18 | 4000 | 400 |
| `arm_layers4_notd` | 4 | `null` | 4000 | 400 |
| `arm_layers8_notd` | 8 | `null` | 4000 | 400 |
| `arm_layers4_notd_long` | 4 | `null` | 8000 | 800 |
| `arm_layers8_notd_long` | 8 | `null` | 8000 | 800 |

3 seeds each, 18 jobs total. Fixed across every arm: RoPE+Canon encoder, attention pooling,
`lora_adapter_rank=8`, `gradient_clip_val=10.0`, `N_supervision=2`, `batch_size=512`, target
model, and the 15-task list (identical `task_categories`/`val_task_categories`, no holdout
this round, every task is seen in training).

## Not built this round: unseen-task generalisation

Fabio's underlying hypothesis is that hypernetworks shouldn't need task descriptors at all,
and should be able to generalise to unseen task categories by borrowing structure from seen
ones (a Chollet-style ARC generalisation argument). That's a genuinely different experiment
(training on a subset of task categories, evaluating zero-shot on held-out ones never seen in
training), confirmed architecturally feasible: `task_categories`/`val_task_categories` can
already be disjoint with zero code changes (`data_modules/arc1d_meta_multiclass.py`), and the
3-shot support-example pathway is fully task-category-agnostic
(`models/hypermodel_lightning.py`'s `prepare_inputs`). Important caveat found while checking
this: it only works cleanly with the task descriptor **off**, `task_indicator_proj`'s
one-hot embedding table (`models/hypermodel.py`) would give a held-out category an untrained,
random embedding row (never having received a gradient update), so descriptor-on generalises
to unseen tasks with meaningless conditioning. This experiment stays with `task_categories ==
val_task_categories` throughout (no holdout) and is a stepping stone: establishing how well
descriptor-free training works on tasks it *has* seen, before spending compute on a genuine
held-out-category test.

## Running

```bash
# Full grid (6 configs, 3 seeds each, 18 jobs total).
bash scripts/run_hypermodel_looped_rope_canon_capacity_baseline.sh
```

No overfit smoke configs this round, matching every experiment since the followup; running
directly on the cluster.

## Reading results

Compare `val_query_exact_match` (mean ± std across 3 seeds) across all 6 arms, per task and
averaged:

- Does the `_td` vs `_notd` gap (0.746 vs. 0.629 at the old settings) narrow at `clip=10`/15
  tasks/either layer count?
- Do the `_long` arms close more of that remaining gap than the standard-length `_notd` arms
  alone, i.e. is the no-descriptor shortfall partly just an undertraining problem?
- Does `layers=8`'s stability advantage (seen with the descriptor on, in the grid experiment)
  also show up without the descriptor, where the model has strictly less information and
  might benefit from capacity differently?
- Particular attention to `1d_move_dp` throughout, as the task most likely to reveal a genuine
  capacity effect.

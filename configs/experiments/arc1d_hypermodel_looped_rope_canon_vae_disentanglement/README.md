# arc1d_hypermodel_looped_rope_canon_vae_disentanglement

## Goal

`arc1d_hypermodel_looped_rope_canon_muon_diag`'s notd/td/frozentd comparison (seed=1, Muon,
`lora_adapter_rank=4`, `max_steps=2000`, 15 tasks) found:

| Arm | Linear-probe accuracy | `val_query_exact_match` |
|---|---:|---:|
| `notd` (no task descriptor) | 73.3% | 82.5% |
| `td` (learned task descriptor) | 100% | 97.5% |
| `frozentd` (frozen task descriptor) | 100% | 96.2% |

`td` and `frozentd` scored identically -- there's no evidence freezing the one-hot embedding
buys anything over learning it in-distribution, so **frozen stays the default going forward,
no further td-vs-frozentd work is planned**. Cluster maps also showed `notd`'s move-family
tasks (Move 1p/2p/3p/Dynamic) partially overlapping, consistent with its lower probe score.

This experiment asks: can a beta-VAE bottleneck on `notd`'s pooled task representation --
regularizing it toward disentangled structure via KL divergence, with no explicit
task-identity signal at all -- close some of that gap? Or does KL regularization alone fail
to substitute for an explicit descriptor here?

## Architecture (fixed across every arm, carried over from `arc1d_hypermodel_looped_rope_canon_muon_diag`)

- Zhu block backbone (`rope_canon_zhu_transformer`, RMSNorm + QK-norm + SwiGLU).
- `optimizer: Muon, muon_lr: 0.005, muon_momentum: 0.95`.
- `hyper_head.lora_adapter: true, lora_adapter_rank: 4` (NOT the committed
  `arc1d_hypermodel_looped_rope_canon_muon` default of 8 -- matches the rank used in the
  notd/td/frozentd comparison this follows on from).
- `hyper_head.num_tasks: null` (notd only -- no task descriptor at all, in every arm).
- `max_steps: 2000, warmup_steps: 200`, `learning_rate: 0.001` (AdamW-aux group).
- 15 in-distribution task categories, `task_categories == val_task_categories`.
- Single seed (`seed=1`) per arm -- not a multi-seed sweep.
- `project_name: arc1d_hypermodel_looped_rope_canon_muon_diag` (reused, not a new project),
  so these runs sit in the same W&B dashboard as the notd/td/frozentd comparison above.

## The variational bottleneck

`hyper_head.variational: true` (models/hypermodel.py) adds two `Linear(hyper_output_dim ->
hyper_output_dim)` heads (`vae_mu_head`, `vae_logvar_head`) on the pooled task representation,
before the (here, always-inactive) task-descriptor add-in. Samples via the reparameterization
trick during training; uses the posterior mean deterministically at eval time -- including for
the embedding-cluster/linear-probe diagnostics, which run in eval mode. `kl_beta` is the
target multiplier on the KL term added to the training loss (`recon_loss + kl_beta_effective *
kl_loss`). `val_loss`/`train_loss` stay reconstruction-only throughout (so
`primary_metric`-based checkpoint selection is unaffected by beta and comparable across every
arm); `train_kl_loss`, `train_elbo_loss`, `train_kl_beta` (the effective, possibly-annealed
weight actually used that step), and `{split}_kl_loss` are logged separately for visibility.

**KL annealing** (`HyperModelLightning`'s `kl_beta_anneal`/`kl_beta_warmup_steps`, top-level
flat config keys, not under `hyper_head`): `"constant"` (default) applies `kl_beta` unchanged
from step 0. `"cosine"` eases the weight in from 0 up to `kl_beta` over `kl_beta_warmup_steps`,
keyed to `self.global_step` -- the count of completed manual `opt.step()` calls, which (like
this repo's own `max_steps`) advances `N_supervision` times per batch, not once per batch. Added
after `beta0_01` (constant, no annealing) diverged -- see Grid below.

**Muon exclusion note**: `vae_mu_head`/`vae_logvar_head` are added to
`HyperModelLightning._MUON_EXCLUDED_MODULE_NAMES` (kept on AdamW, not Muon), on the same
rationale as the existing `task_indicator_proj` exclusion -- they produce/consume the pooled
representation itself, not an internal hidden-to-hidden interaction matrix. This is a design
call, not forced by Keller Jordan's own Muon guidance.

## Grid

6 constant-`kl_beta` values, 1 seed each:

| Config | `kl_beta` | Result |
|---|---:|---|
| `beta0_01.yaml` | 0.01 | **Diverged** -- non-finite gradient at `global_step=2772`. At this low a weight, the KL term barely constrains `logvar`, so the model can drift toward large posterior variance over training (unpenalized), eventually producing a sampled `z` large enough to blow up the gradient. Left as-is (not rerun) -- see the annealed re-run below instead. |
| `beta0_1.yaml` | 0.1 | Probe 38.7%, `val_query_exact_match` **0%**. |
| `beta0_5.yaml` | 0.5 | Probe 17.3%, `val_query_exact_match` **0%**. |
| `beta1.yaml` | 1 | Probe 6.7% (chance), `val_query_exact_match` **0%**. |
| `beta2.yaml` | 2 | Probe 10.7%, `val_query_exact_match` 5%. |
| `beta10.yaml` | 10 | Probe 6.7% (chance -- full posterior collapse), `val_query_exact_match` **0%**. |

Every constant-beta arm from 0.1 up collapsed task-solving to ~0% `val_query_exact_match` --
not a graceful disentanglement/accuracy trade-off, outright failure to learn the task at all,
worse the higher beta goes. This is why every beta got an annealed re-run, not just the
diverged 0.01 arm.

**Annealed re-run of the full grid**: does easing the KL weight in via `kl_beta_anneal: cosine`
(`kl_beta_warmup_steps: 400`, 10% of the 4000 scaled optimizer steps) avoid both the small-beta
divergence and the larger-beta task-accuracy collapse? Note annealing only ramps the KL loss
*weight* -- the reparameterization sampling itself still happens from step 0 regardless of
beta, so this isn't guaranteed to fix a noise-injection problem, only a loss-weighting one.

| Config | `kl_beta` (post-warmup) | Probe acc | `val_query_exact_match` |
|---|---:|---:|---:|
| `beta1e-5_anneal.yaml` | 0.00001 | 66.7% | 65.3% |
| `beta1e-4_anneal.yaml` | 0.0001 | **74.7%** | 49.3% |
| `beta1e-3_anneal.yaml` | 0.001 | 68.0% | 66.7% |
| `beta1e-2_anneal.yaml` | 0.01 | 69.3% | 65.8% |
| `beta0_1_anneal.yaml` | 0.1 | 50.7% | 9.3% |
| `beta0_5_anneal.yaml` | 0.5 | 9.3% | 0.0% |
| `beta1_anneal.yaml` | 1 | 18.7% | 1.3% |
| `beta2_anneal.yaml` | 2 | 6.7% (chance) | 0.0% |
| `beta10_anneal.yaml` | 10 | **Diverged** -- later than the constant arm (`global_step=2973` vs. constant beta=10 which completed without diverging), since the 400-step warmup only delays reaching full strength, it doesn't prevent the instability once beta gets there. |

**Verdict on the 400-step (10%-of-training) anneal**: a narrow stable band exists at
`kl_beta &le; 0.01`, where probe accuracy roughly matches or slightly beats `notd` (73.3%) but
`val_query_exact_match` (49-67%) still falls well short of `notd`'s 82.5%. Every beta at or
above 0.1 still collapses task-solving toward 0%, same as the constant-beta arms. Annealing
over just the first 10% of training doesn't fix the larger-beta problem -- it only delays it.

## Full-training anneal (`kl_beta_warmup_steps: 4000`)

Prompted by the above: the 400-step warmup reaches full target beta by 10% of the way through
training and holds it constant for the remaining 90% -- effectively "constant-beta training,
started late" -- which is exactly why `beta10_anneal` still diverged once warmup completed.
This variant instead ramps across the **entire** scaled training run (`kl_beta_warmup_steps:
4000` = `max_steps(2000) * N_supervision(2)`, the full run length), so beta only reaches its
target value at the very last step, never holding at full strength for an extended period.

`kl_beta` values chosen densely up to 0.1 (both `1x` and `5x` per decade), since every constant
and 400-step-annealed arm at `kl_beta &ge; 0.1` collapsed task-solving -- no need to re-test
0.5/1/2/10 with this scheme:

| Config | `kl_beta` (final, reached only at the last step) | Probe acc | `val_query_exact_match` |
|---|---:|---:|---:|
| `beta1e-7_annealfull.yaml` | 0.0000001 | | |
| `beta1e-6_annealfull.yaml` | 0.000001 | | |
| `beta1e-5_annealfull.yaml` | 0.00001 | | |
| `beta5e-5_annealfull.yaml` | 0.00005 | | |
| `beta1e-4_annealfull.yaml` | 0.0001 | 78.7% | 65.3% |
| `beta5e-4_annealfull.yaml` | 0.0005 | **84.0%** | 68.0% |
| `beta1e-3_annealfull.yaml` | 0.001 | 77.3% | **72.0%** |
| `beta5e-3_annealfull.yaml` | 0.005 | 65.3% | 61.3% |
| `beta1e-2_annealfull.yaml` | 0.01 | 66.7% | 73.3% |
| `beta5e-2_annealfull.yaml` | 0.05 | 66.7% | 68.0% |
| `beta0_1_annealfull.yaml` | 0.1 | 65.3% | 70.7% |

**Result so far (0.0001 to 0.1): no divergence and no collapse anywhere in this range** -- a
sharp contrast with both the constant-beta arms (collapsed to ~0% task accuracy from
`beta=0.1` up) and the 400-step anneal (collapsed above `beta=0.01`, diverged at `beta=10`).
Spreading the ramp across the *entire* run means beta never sits at full strength for long
before training ends, avoiding the extended fight between reconstruction and a fully-engaged
KL penalty that caused every earlier failure mode. Several betas (`5e-4`, `1e-4`, `1e-3`) beat
the `notd` probe-accuracy baseline (73.3%) outright. `beta=1e-3` is the best combined result of
the whole experiment so far -- 77.3% probe accuracy (above `notd`) alongside 72.0% task
accuracy (closest yet to `notd`'s 82.5%, without any of the collapse seen everywhere else in
this beta range).

`1e-7` through `5e-5` extend the grid below `1e-4` to check whether performance keeps
improving toward (or past) the `notd` baseline as beta approaches zero, or whether it
plateaus/degrades -- i.e. whether the injected reparameterization noise itself costs
something regardless of how weakly the KL term penalizes it.

## Running

```bash
# Sequential, whole-node-per-job (original constant-beta grid):
bash scripts/run_hypermodel_vae_disentanglement.sh

# Just the 400-step-warmup annealed re-run, sequential:
CELL_GLOB="beta*_anneal.yaml" bash scripts/run_hypermodel_vae_disentanglement.sh

# Parallel, one job per GPU -- for a node with several genuinely free GPUs:
NUM_GPUS=8 CELL_GLOB="beta*_anneal.yaml" bash scripts/run_hypermodel_vae_disentanglement_parallel.sh

# Full-training anneal grid, parallel (7 jobs, fits on one 8-GPU node):
NUM_GPUS=8 CELL_GLOB="beta*_annealfull.yaml" bash scripts/run_hypermodel_vae_disentanglement_parallel.sh
```

Single seed (`seed=1`), idempotent (skips a config whose `results.txt` already exists) --
note `beta0_01` (no `results.txt`, since it diverged) would be retried, and is expected to
diverge again identically, if the full unfiltered glob is rerun.

## Reading results

For each beta, in `outputs/arc1d_hypermodel_looped_rope_canon_muon_diag/looped_hyper_rope_canon_vae_disentanglement_beta<X>_seed1/`:

1. `embedding_clusters/linear_probe_summary.txt` and `pooled_task_latent.png` -- compare
   `mean_accuracy` against `notd`'s 73.3% baseline and `td`/`frozentd`'s 100% ceiling, and
   check qualitatively whether the move-family cluster overlap resolves as beta increases.
2. `results.txt` -- `val_query_exact_match` against `notd`'s 82.5% baseline: does any beta
   improve disentanglement without a large drop in task-solving accuracy?
3. `train_kl_loss` / `train_elbo_loss` curves (W&B) -- watch for posterior collapse (KL
   dropping to ~0, most likely at low beta, meaning the bottleneck isn't doing anything) at
   one end and reconstruction quality visibly degrading (`val_loss`/`val_query_exact_match`
   cratering) at the other (most likely at `beta=10`). For the annealed arms, `train_kl_beta`
   confirms the ramp shape (0 at step 0, target `kl_beta` by step 400) and whether the run
   stays stable once annealing completes and the KL weight is held at its small target value.

The headline question: is there a beta where linear-probe accuracy rises meaningfully above
73.3% without `val_query_exact_match` dropping much below 82.5% (KL buying disentanglement
without breaking task-solving)? If no such beta exists across this range, that's evidence KL
regularization alone can't substitute for an explicit task descriptor here.

## Overall conclusion (across all three grids, 22 arms total)

Whether the KL bottleneck is usable at all depends entirely on **how** beta is introduced, far
more than on its final value:

- **Constant beta** (no annealing): unstable at the low end (`0.01` diverged) and destructive
  everywhere else (`0.1` and up collapsed task-solving to ~0%).
- **400-step (10%-of-training) anneal**: only postpones the same failure modes -- stable at
  `beta &le; 0.01`, but still collapses at `0.1`+ and still diverges at `10` once the ramp
  completes and beta holds at full strength for the remaining 90% of training.
- **Full-training anneal** (ramp spans the whole run, beta only reaches its target at the
  final step): stable across the **entire** tested range (`1e-4` to `0.1`), no collapse, no
  divergence. Several betas here (`5e-4`, `1e-4`, `1e-3`) beat the `notd` probe-accuracy
  baseline outright, with `1e-3` the best combined result of the whole experiment (77.3% probe
  / 72.0% task accuracy).

Even at its best, though, the bottleneck doesn't fully close the gap to an explicit task
descriptor -- `1e-3`'s 72.0% task accuracy is still meaningfully below `td`/`frozentd`'s
~97%. The practical takeaway: a KL-regularized bottleneck **can** mildly improve
disentanglement over `notd` without wrecking the model, but only when annealed across the
full training run; and even then it's a partial improvement, not a substitute for a real
task-identity signal.

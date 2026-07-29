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
| `beta0_1.yaml` | 0.1 | |
| `beta0_5.yaml` | 0.5 | |
| `beta1.yaml` | 1 | |
| `beta2.yaml` | 2 | |
| `beta10.yaml` | 10 | |

**Annealed re-run of the small-beta end** (4 more jobs), after `beta0_01`'s divergence: does
easing the KL weight in via `kl_beta_anneal: cosine` (`kl_beta_warmup_steps: 400`, 10% of the
4000 scaled optimizer steps) avoid the same instability at very small target beta values?

| Config | `kl_beta` (post-warmup) |
|---|---:|
| `beta1e-2_anneal.yaml` | 0.01 |
| `beta1e-3_anneal.yaml` | 0.001 |
| `beta1e-4_anneal.yaml` | 0.0001 |
| `beta1e-5_anneal.yaml` | 0.00001 |

## Running

```bash
bash scripts/run_hypermodel_vae_disentanglement.sh

# Or just the annealed re-run:
CELL_GLOB="beta*_anneal.yaml" bash scripts/run_hypermodel_vae_disentanglement.sh
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

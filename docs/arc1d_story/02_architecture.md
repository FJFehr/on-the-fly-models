# 02 — Converged Architecture(s)

This note documents the model architecture(s) the ARC-1D hypernetwork story has converged
on, as they actually exist in `models/` and in the committed experiment configs under
`configs/experiments/`. Every number below is read directly from source code or a config
YAML (or computed by instantiating the real classes) — none is recalled from memory or
paraphrased from an earlier write-up. Two open inconsistencies are called out explicitly
rather than resolved; see the closing section.

## 1. Target model — `RoPECanonLoopedTransformer`

Defined in `models/rope_looped_transformer.py`. This is the small "task solver" model whose
weights the hypernetwork generates on the fly, one set per task.

### Architecture

Uniform-width variant (the one actually used everywhere below, `inner_dim == hidden_dim`):

```
input_projection → pre_layer (1×) → middle_layer (n_loops×, weight-shared) → post_layer (1×) → final_norm → output_head
```

- `pre_layer`, `middle_layer`, `post_layer` are all `RoPECanonBlock` instances (pre-norm,
  `LayerNorm` → attention → residual → `LayerNorm` → MLP → residual).
- **RoPE**: applied inside `RoPECanonSelfAttention` to Q and K, per head, after the linear
  projection and reshape, before the dot product (`models/rope_looped_transformer.py:80-81`).
  The input embedder must be configured with `use_sinusoidal_pe: false` — RoPE is the only
  positional signal.
- **Canon convolutions**: depthwise `nn.Conv1d` short convolutions
  (`models/canon_layer.py`), based on "Physics of Language Models: Part 4.1, Architecture
  Design and the Magic of Canon Layers" (Zeyuan Allen-Zhu, NeurIPS 2025). Four possible
  positions per block: A (before attention, on `norm(x)`), B (inside attention, on
  concatenated `[q,k,v]`), C (before the MLP, on `norm(h)`), D (inside the MLP, on the
  GELU/SwiGLU gate tensor).
- **Looping**: the `middle_layer` block is applied `n_loops` times with **shared weights**
  (the same `nn.Module` instance called repeatedly), so parameter count is independent of
  `n_loops` — only compute/activation memory scales with it
  (`models/rope_looped_transformer.py:540`, confirmed by the class docstring).
- `use_block_skip`: adds `x = x + x_input` at the end of `pre_layer`/`middle_layer`/
  `post_layer` (residual around the whole block, on top of the block's own internal
  residuals).
- `use_loop_skip`: re-injects the pre-loop hidden state `h_inner` after every single loop
  iteration (`h = h + h_inner`, `models/rope_looped_transformer.py:647`) — distinct from
  `use_inner_bypass` (a single skip over all `n_loops` iterations combined) and
  `use_outer_bypass` (a single skip over the whole pre+middle+post stack), neither of which
  is used in the winning config below.

### Winning config

Read directly from `configs/experiments/arc1d_hypermodel_looped_rope_canon_capacity_baseline/base.yaml`
and `configs/experiments/arc1d_hypermodel_looped_rope_canon_muon/base.yaml` (`target_model` block,
identical in both, and reused unchanged across essentially every later hypernetwork experiment —
see the n_loops table below for the one axis that does vary):

| Field | Value |
|---|---|
| `hidden_dim` | 16 |
| `num_heads` | 2 |
| `inner_dim` | 16 (equal to `hidden_dim` — uniform width, no wide middle) |
| `inner_num_heads` | 2 |
| `n_loops` | 4 (see inconsistency below) |
| `dropout` | 0.1 |
| `canon_set` | `"ABCD"` |
| `canon_kernel` | 5 |
| `canon_activation` | `true` |
| `canon_residual` | `true` |
| `canon_causal` | `false` |
| `use_block_skip` | `false` |
| `use_loop_skip` | `false` |

This is structurally **T5** from `arc1d_uniform_ablation`'s T1–T7 story (RoPE + Canon ABCD +
looped middle, no skip mechanisms), not T6 or T7 (which add block skip and loop skip
respectively) — see the inconsistency discussion below.

**Total parameter count**: `arc1d_uniform_ablation/README.md` reports **11,760 backbone
params at dim=16** (T3–T7, held exactly flat across n_loops/skip changes since parameter
count doesn't depend on those). Instantiating the actual class with the config above and
`use_output_head=true` (the default) gives **11,920** total target-model parameters — the
extra 160 is the `output_head: Linear(16 → 10, bias=False)` (10 output classes for the
multiclass ARC-1D prediction task), which the backbone-only 11,760 figure excludes. Verified
directly: `HyperModel.__repr__()` on a live-instantiated model from
`configs/experiments/arc1d_hypermodel_looped_rope_canon_muon/arm_frozentd_muon.yaml` prints
`Target params: 11,920`.

## 2. Inconsistency: `n_loops`

`arc1d_uniform_ablation/README.md`'s own stated conclusion (its "Result" paragraph, loop
diagnostics L1–L6) is that **more loop iterations with no skip mechanism at all is the
reliable lever**: the no-skip sweep (n_loops = 1→4→8→16→32) peaks at **n_loops=8** for
dim=16 (93.3% test exact-match) and **n_loops=16** for dim=32 (95.2% test), with degradation
by n_loops=32. Separately, the plotted T1–T7 narrative's own final step, **T7**, uses
`n_loops=4` with **both** block skip and loop skip on. Neither of those is what the actual
downstream hypernetwork experiments configure — they overwhelmingly use `n_loops=4` with
**no** skip at all (structurally T5, not T7, and not the README's own n_loops=8/16
recommendation either).

Grepped directly from `target_model.params.n_loops` across every relevant experiment
directory's YAML files:

| Experiment | `n_loops` | Skip flags |
|---|---:|---|
| `arc1d_hypermodel_looped_hypernet_rope_canon_ablation` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_capacity_baseline` | 4 | none |
| `arc1d_hypermodel_looped_rope_canon_followup` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_generalization` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_grid` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_lr_sweep` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_mechanism_ablation` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_muon` | 4 | none |
| `arc1d_hypermodel_looped_rope_canon_muon_moveablation` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_muon_sweep` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_optimizer` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_rank_sweep` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_vae_disentanglement` | 4 | — |
| `arc1d_hypermodel_looped_rope_canon_zhu_block` | 4 | — |
| `arc1d_hypermodel_compositional_generalization` | 4 | — |
| `arc1d_lowdata` | 4 | — |
| `arc1d_lowdata_baseline` (direct-supervised, not a hypermodel) | 4 | — |
| `arc1d_lowdata_lowrank` | 4 | — |
| `arc1d_hypermodel_looped_lora_adapter/base_n2_loop4_noskip.yaml` | 4 | none |
| `arc1d_hypermodel_looped_mix11/mix11_n2_loop4_noskip.yaml` | 4 | none |
| `arc1d_hypermodel_looped_mix11/mix11_td.yaml` | **8** | block + loop skip |
| `arc1d_hypermodel_looped_lora_adapter/base.yaml` | **8** | block + loop skip |

The two `n_loops=8` configs both trace their justification to a *different* experiment
(`arc1d_hypermodel_looped_recolor`'s "flip+recolor sweep", per their own comments — "the
winning settings from the flip+recolor sweep"), not to `arc1d_uniform_ablation`. So there are
effectively three distinct, uncoordinated "winning n_loops" claims floating around this
codebase: `arc1d_uniform_ablation`'s own loop-diagnostics recommendation (8 or 16, no skip),
its plotted T7 narrative endpoint (4, block+loop skip), and what the large majority of
downstream hypernetwork experiments actually run (4, no skip). Not resolved here — see
Open Questions.

## 3. Hypernetwork encoder — `HyperModel` + `RoPECanonZhuTransformer`

`models/hypermodel.py` (`HyperModel` class) wires an arbitrary hypernetwork encoder to the
target model via `functional_call`/`vmap`: the encoder emits tokenwise features over the
serialized support examples, those are pooled to one task vector, projected to the target's
flat parameter vector, and applied per-example. The target model's own parameters are frozen
(shape templates only, except under `lora_adapter` — see §4).

### Encoder: the "Zhu block"

`models/rope_looped_transformer.py` defines `RoPECanonZhuTransformer` /
`RoPECanonZhuBlock` / `RoPECanonZhuSelfAttention`, registered as `rope_canon_zhu_transformer`
in `HYPERNETWORK_REGISTRY` (`models/hypermodel_lightning.py:80-83`). "Zhu" refers to Zeyuan
Allen-Zhu, author of the Canon Layers paper this repo's Canon mechanism is already based on
(`configs/experiments/arc1d_hypermodel_looped_rope_canon_zhu_block/README.md`). Three
independent boolean flags, all confirmed as real components in code, default `false` (in
which case the class is byte-identical to the plain `RoPECanonTransformer`):

- `use_rmsnorm`: `RMSNorm` (`models/transformer.py:24-33`, Zhang & Sennrich 2019 — no bias, no
  mean-centering) replaces `LayerNorm` for both pre-attention/pre-MLP norms and the final norm.
- `use_qk_norm`: a per-head `RMSNorm(head_dim)` applied separately to query and key
  (`models/rope_looped_transformer.py:220-222,238-240`), right after the per-head reshape and
  *before* RoPE rotation — normalise magnitude, then rotate direction. Matches the
  Gemma2/Qwen2/OLMo2 convention.
- `use_swiglu`: `CanonZhuMLP` (`models/canon_transformer.py:83-121`) replaces `CanonMLP`:
  `gate_up_proj` → Canon-D on the concatenated `[gate, up]` tensor → SwiGLU → `down_proj`,
  `intermediate_dim` sized to `int(2/3 * 4 * hidden_dim)` rounded to a multiple of 8 (LLaMA
  convention) to keep FFN parameter count close to the GELU MLP's.

### Winning encoder config

Read from `configs/experiments/arc1d_hypermodel_looped_rope_canon_muon/base.yaml`'s
`hyper_model` block, which fixes the Zhu backbone with all three flags on across every arm
("the current leader in `arc1d_hypermodel_looped_rope_canon_lr_sweep`'s `zhu_all` arm, mean
exact-match 0.8625 across 5 seeds"):

| Field | Value |
|---|---|
| `name` | `rope_canon_zhu_transformer` |
| `hidden_dim` | 64 |
| `num_heads` | 4 |
| `num_layers` | 8 |
| `output_dim` | 64 |
| `canon_set` | `"ABCD"` |
| `canon_kernel` | 5 |
| `canon_activation` / `canon_residual` / `canon_causal` | `true` / `true` / `false` |
| `use_rmsnorm` | `true` |
| `use_qk_norm` | `true` |
| `use_swiglu` | `true` |

`num_layers=8` traces to `arc1d_hypermodel_looped_rope_canon_capacity_baseline`'s own finding
(stated in that experiment's `base.yaml` header comment): at `gradient_clip_val=10` with the
task descriptor on, `num_layers=8` gave a real stability improvement over `num_layers=4` (mean
exact-match 0.864 vs 0.856, std 0.009 vs 0.027 — roughly 3× tighter). Note this differs from
the `num_layers=2` used earlier in `arc1d_hypermodel_looped_rope_canon_followup`'s
`arm_default`, and from the 2/4/8 grid `arc1d_hypermodel_looped_rope_canon_grid` swept without
its README recording a definitive numeric winner in the committed text.

### Pooling: attention vs hierarchical

`models/hypermodel.py` defines two poolers: `AttentionPooler` (a single learned-query
attention pool, fixed-size regardless of depth, `LearnedQueryAttentionPool`) and
`HierarchicalPooler` (6 support segments → attention-pool → interaction blocks → 3 examples →
attention-pool → interaction blocks → attention-pool → 1 task vector).

`arc1d_hypermodel_looped_rope_canon_followup/README.md` set up the attention-vs-hierarchical
comparison (`arm_default` = hierarchical, `arm_pooling_attention` = attention) but its own
committed text doesn't record numeric results. The verdict is recorded downstream instead, in
`arc1d_hypermodel_looped_rope_canon_grid/README.md`: "attention pooling only (no hierarchical:
confirmed no benefit in the followup experiment and barely-there, higher-variance benefit in
the mechanism_ablation experiment)". **Attention pooling won.**

Checked directly against later configs (`hyper_head.pooling` key) — this is the one place
where "what won" and "what's actually configured downstream" agree:

| Experiment | `hyper_head.pooling` |
|---|---|
| `arc1d_hypermodel_looped_rope_canon_capacity_baseline` | `attention` |
| `arc1d_hypermodel_looped_rope_canon_muon` | `attention` |
| `arc1d_hypermodel_compositional_generalization` | `attention` |
| `arc1d_hypermodel_looped_rope_canon_vae_disentanglement` | `attention` |

### LoRA-adapter weight-generation mechanism

`hyper_head.lora_adapter: true` switches `HyperModel` to a third weight-generation mode
(alongside full-MLP generation and the older `low_rank_output` path), documented in
`configs/experiments/arc1d_hypermodel_looped_lora_adapter/README.md` and implemented in
`models/hypermodel.py:337-384,485-520`:

- The target model (`RoPECanonLoopedTransformer`) keeps its own ordinary random PyTorch init
  as a persistent, **frozen** backbone `W_backbone` (frozen unless
  `lora_adapter_train_backbone: true`, not used in any winning config here).
- For each of the model's 2D weight matrices (14 of them: `input_projection`, and per block ×
  {pre, middle, post}: `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`; plus
  `output_head`), the hypernetwork generates a per-tensor rank-`r` delta `ΔW = B @ A`, sized to
  that tensor's own `(d_out, d_in)` shape. Effective weight: `W = W_backbone + B @ A`.
  `B` is zero-initialised (standard LoRA convention), so `ΔW ≡ 0` at step 0.
- The remaining non-matrix tensors (LayerNorm weight/bias, Canon depthwise conv kernels — 2,288
  of the model's 11,920 params, 19%) have no meaningful low-rank structure and are generated
  fully from scratch via a separate `lora_proj_other` head, exactly as the full-generation
  baseline does for every parameter.

### Rank sweep

`arc1d_hypermodel_looped_rope_canon_rank_sweep/README.md`: sweeps `lora_adapter_rank` ∈
{1, 2, 4, 8}. Result (recorded in its Goal section, restating the prior followup experiment's
finding it acts on): **rank 8 unlocked every task except `1d_recolor_oe` and
`1d_recolor_cnt`**, clearly better than rank 1. **Rank 8 wins**, confirmed as the expected
value. `arc1d_hypermodel_looped_rope_canon_grid/README.md` explicitly states
"`lora_adapter_rank=8` (confirmed the winning rank in the rank_sweep experiment)".

## 4. Inconsistency: `lora_adapter_rank`

Despite rank=8 being the confirmed winner, several later "downstream" experiments configure
rank=4 instead — traced explicitly in their own header comments to copying an earlier,
smaller, single-seed comparison (`arc1d_hypermodel_looped_rope_canon_muon_diag`, referenced in
comments but not present as a directory in the current tree) rather than the rank_sweep
result. `arc1d_hypermodel_compositional_generalization/base.yaml`'s header states this
outright: "lora_adapter_rank=4 here, **NOT** the committed
`arc1d_hypermodel_looped_rope_canon_muon` default of 8".

Grepped directly from `lora_adapter_rank` across every relevant experiment directory:

| Experiment | `lora_adapter_rank` | Note |
|---|---:|---|
| `arc1d_hypermodel_looped_rope_canon_capacity_baseline` | 8 | |
| `arc1d_hypermodel_looped_rope_canon_generalization` | 8 | |
| `arc1d_hypermodel_looped_rope_canon_grid` | 8 (all 18 cells) | "confirmed the winning rank" |
| `arc1d_hypermodel_looped_rope_canon_lr_sweep` | 8 | |
| `arc1d_hypermodel_looped_rope_canon_mechanism_ablation` | 8 | "no longer swept, confirmed the winning rank" |
| `arc1d_hypermodel_looped_rope_canon_muon` (all 4 arms) | 8 | referred to elsewhere as "the committed ... default" |
| `arc1d_hypermodel_looped_rope_canon_muon_moveablation` | 8 | |
| `arc1d_hypermodel_looped_rope_canon_muon_sweep` | 8 | |
| `arc1d_hypermodel_looped_rope_canon_optimizer` | 8 | |
| `arc1d_hypermodel_looped_rope_canon_rank_sweep/rank8.yaml` | 8 | the sweep's own winning arm |
| `arc1d_hypermodel_looped_rope_canon_zhu_block` | 8 | |
| `arc1d_lowdata` | 8 | |
| `arc1d_hypermodel_looped_rope_canon_followup/arm_default,arm_pooling_attention,arm_nsup4` | 1 | pre-rank-sweep baseline |
| `arc1d_hypermodel_looped_rope_canon_rank_sweep/rank1,rank2,rank4` | 1, 2, 4 | sweep arms |
| **`arc1d_hypermodel_looped_rope_canon_vae_disentanglement`** | **4** | explicitly "NOT the committed ... default of 8" |
| **`arc1d_hypermodel_compositional_generalization`** | **4** | explicitly "NOT the committed ... default of 8" |
| `arc1d_lowdata_lowrank` | 1, 2, 4 (swept) / "full" (no adapter) | its own dedicated rank sweep, different question |
| `arc1d_lowdata_baseline` | n/a | direct-supervised model, no `hyper_head` block at all |

So the rank-sweep result (8) and the "committed default" it fed into `arc1d_hypermodel_looped_rope_canon_muon`
onward are internally consistent with each other, but two later experiments
(`vae_disentanglement`, `compositional_generalization`) both deliberately deviate to rank=4,
each citing the other/a pre-sweep comparison rather than the rank_sweep result. Not resolved
here — see Open Questions.

## 5. Task descriptor mechanism

`models/hypermodel.py`'s `HyperModel.__init__` (`task_indicator_proj`, `num_tasks`,
`freeze_task_indicator`) implements three modes:

- **`notd`**: `num_tasks=null` → `task_indicator_proj = None`. No task-identity signal at all;
  the pooled hypernetwork representation alone drives weight generation.
- **`td`**: `num_tasks` set, `freeze_task_indicator` unset/`false` → `task_indicator_proj =
  nn.Linear(num_tasks, hyper_output_dim, bias=False)`, a trainable one-hot task embedding
  added to the pooled representation (`extract_parameter_vectors`, lines 469-471).
- **`frozen_td`**: `num_tasks` set, `freeze_task_indicator: true` → the same projection matrix,
  but `weight.requires_grad_(false)` (line 289): it stays at its random init for every
  category, trained or not. Rationale in code comment (lines 283-286): a held-out category's
  column is then drawn from the same distribution the rest of the network learned to interpret,
  rather than being "the one column that never received a gradient while its 17 siblings
  moved."

**`frozen_td` is the converged choice** for the paper narrative:
`arc1d_hypermodel_looped_rope_canon_generalization` established it as a middle ground between
`td` (learned, actively harmful for the one held-out category it helped least) and `notd` (no
signal at all); `arc1d_hypermodel_looped_rope_canon_muon` and
`arc1d_hypermodel_compositional_generalization` both carry it forward as one of their two
primary arms (alongside `notd`), with plain `td` dropped.

### `num_tasks=18` for the 15-task working set

`TASK_CATEGORY_INDEX` (`models/hypermodel_lightning.py:37-56`) is a fixed 18-entry registry
(indices 0–17) covering every ARC-1D category ever used in this codebase. The standard
"15-task working set" used by `arc1d_hypermodel_looped_rope_canon_muon`,
`arc1d_hypermodel_looped_rope_canon_capacity_baseline`, `arc1d_hypermodel_compositional_generalization`,
etc. omits exactly 3 of those 18: `1d_padded_fill` (excluded from every experiment "by
established convention", per `arc1d_uniform_ablation/README.md`'s Jobs section) and
`1d_recolor_oe`/`1d_recolor_cnt` (excluded per
`arc1d_hypermodel_looped_rope_canon_capacity_baseline/base.yaml`'s header comment: "stuck at
a literal 0.000 in every single run regardless of layers or clip, a qualitatively different,
likely structural failure, not a capacity question"). So `num_tasks=18` is set to the size of
the full category registry, not to 15 — the frozen/learned projection reserves 3 unused rows
(indices for the excluded categories) even though training only ever samples 15 of them.

### `num_tasks=28` for compositional generalization

`configs/experiments/arc1d_hypermodel_compositional_generalization/frozen_td.yaml` sets
`hyper_head.num_tasks: 28` instead of the usual 18 — 10 extra indices (18..27) reserved for
the held-out composite task categories assigned at eval time by
`scripts/eval_compositional_holdout.py`. Per that config's own comment: since only 15
in-distribution categories are ever sampled during training and `freeze_task_indicator: true`
keeps the whole projection at its random init throughout, the 10 extra rows are simply
additional random-but-fixed directions the model never had a reason to rely on — well-defined
for `frozen_td` at eval time, unlike a *learned* (`td`) one-hot embedding would be for an
index it never saw.

## 6. Optimizer / training recipe

### Muon setup

`arc1d_hypermodel_looped_rope_canon_muon/README.md` introduces the Muon optimizer (Keller
Jordan) for this model, split via `models/hypermodel_lightning.py`'s `_build_muon_param_groups`
(lines 1317-1358+) into a Muon-eligible group and an AdamW "aux" group:

- **Muon-eligible**: only `nn.Linear` weights, matched against every dotted path segment
  (fixed in `muon_sweep` after a bug where only the last segment was checked, which silently
  failed to exclude `lora_proj_a.<idx>`/`lora_proj_b.<idx>` `ModuleList` members) — this
  includes the internal attention/MLP Linear weights in every transformer block **and** the
  `lora_proj_a`/`lora_proj_b`/`lora_proj_other` hyper-head projections (the actual generated
  target-model weights), a deliberate call rather than the strictest reading of "final output
  layer".
- **Explicit AdamW exclusions** (`_MUON_EXCLUDED_MODULE_NAMES`, always excluded regardless of
  being `nn.Linear`): `input_projection` (hypernetwork input embedding), `output_head`
  (hypernetwork final output layer), `task_indicator_proj` (structurally `nn.Linear` but
  semantically a one-hot task-identity embedding table).
- Everything non-`nn.Linear` (embeddings, `RMSNorm`/`LayerNorm`, Canon `nn.Conv1d` weights,
  the attention pooler's `pool_query` vector, biases) is AdamW by construction.

**Hyperparameters** (`models/hypermodel_lightning.py:261-265`, defaults `muon_lr=0.02`,
`muon_momentum=0.95`; actual values used in `arc1d_hypermodel_looped_rope_canon_muon`):
`muon_lr=0.005`, `muon_momentum=0.95`. `muon_lr` started at Keller Jordan's published default
(0.02) but the first `arm_frozentd_muon` attempt diverged to NaN loss around global_step
~915/4000 (pre-clip grad norms ~1e18); lowered 4× to 0.005 as the cheapest first fix.
AdamW-aux group used `learning_rate=0.001`, `weight_decay=0.01` (same as the plain-AdamW
arms).

### `arc1d_hypermodel_looped_rope_canon_muon_sweep` findings

A 216-cell grid (`muon_lr` × `learning_rate` × `weight_decay` × `batch_size` ×
`muon_exclude_lora_heads`), cancelled after 99 cells once the direction was clear enough to
act on (per `arc1d_hypermodel_looped_rope_canon_muon_moveablation/README.md`, which restates
the findings and locks them in):

- **Excluding `lora_proj_a`/`lora_proj_b` from Muon (`muon_exclude_lora_heads: true`) beat
  including them**: mean `val_query_exact_match` 0.654 vs 0.618 across the 99 completed cells;
  9 of the top 10 individual cells used exclusion.
- **`batch_size=2048` pulled ahead** of 512/1024 (mean 0.658 vs ~0.623–0.629); 7 of the top 10
  cells used it. `batch_size=4096` was tried in the follow-up and genuinely OOMs (44.2/44.4 GiB
  on a single GPU) — dropped from the grid entirely.
- `muon_lr` itself showed no clear trend across 0.008/0.01/0.016 (all ~0.63–0.64, within
  noise); `muon_lr=0.02` (the value that originally caused the NaN divergence) was never
  reached by the 99-cell partial sweep.

Both winning settings (`muon_exclude_lora_heads: true`, `batch_size=2048`) are locked in as
fixed, non-swept values in `arc1d_hypermodel_looped_rope_canon_muon_moveablation`.

### AdamW / RAdam elsewhere

Not every experiment in this lineage uses Muon. `optimizer: RAdam`, `learning_rate: 0.001` is
the baseline used by e.g. `arc1d_hypermodel_looped_rope_canon_capacity_baseline` and
`arc1d_hypermodel_looped_rope_canon_zhu_block`.
`arc1d_hypermodel_looped_rope_canon_optimizer/README.md` directly compares this RAdam baseline
against an `arm_adamw` arm (`optimizer: AdamW`, `learning_rate: 0.0005`), everything else held
identical to the capacity-baseline's `layers=8, notd` config — a clean single-axis AdamW-vs-
RAdam comparison layered on top of the Muon-vs-AdamW comparisons run separately in the `_muon`
and `_muon_sweep` experiments.

## 7. A concrete instantiation

Instantiating the full stack directly from a real committed config
(`configs/experiments/arc1d_hypermodel_looped_rope_canon_muon/arm_frozentd_muon.yaml`, via
`training/config.load_config` + `models.MODEL_REGISTRY["hyper_model"]`) gives:

```
HyperModelLightning(
  Hypernetwork: RoPECanonZhuTransformer(input=16, hidden=64, layers=8, heads=4, output=64,
                                         canon_set='ABCD', canon_kernel=5,
                                         rmsnorm=True, qk_norm=True, swiglu=True)
  Hyper pooling: attention_pool(query over token sequence, hidden=64)
  Hyper projection: Linear(64 -> 128) + GELU +
                     [14x per-tensor rank-8 B@A adapters on frozen random backbone] +
                     [other: Linear(-> 2288)]
  Target: RoPECanonLoopedTransformer(input=16, hidden=16, hidden=16, heads=2, n_loops=4,
                                      canon_set='ABCD', canon_kernel=5)
  Target params: 11,920
)
```

Total parameters: **1,582,112**. Trainable parameters: **1,569,040** (the ~13k difference is
the frozen LoRA-adapter backbone — the target model's own random init, kept but not trained).

## Open questions for the next round to settle

- **`n_loops`**: `arc1d_uniform_ablation`'s own loop-diagnostics conclusion recommends
  n_loops=8 (dim=16) or n_loops=16 (dim=32), no skip. Its plotted T1–T7 narrative's endpoint
  (T7) uses n_loops=4 with block+loop skip. Almost every downstream hypernetwork experiment
  instead uses n_loops=4 with **no** skip (structurally T5). Two configs
  (`arc1d_hypermodel_looped_mix11/mix11_td.yaml`, `arc1d_hypermodel_looped_lora_adapter/base.yaml`)
  use n_loops=8 with block+loop skip, sourced from a *different* experiment
  (`arc1d_hypermodel_looped_recolor`'s flip+recolor sweep), not from `arc1d_uniform_ablation`.
  Which n_loops/skip combination should the paper's headline architecture actually claim?
- **`lora_adapter_rank`**: the rank_sweep experiment found rank=8 the clear winner, and that
  became the "committed default" for most downstream experiments — but
  `arc1d_hypermodel_looped_rope_canon_vae_disentanglement` and
  `arc1d_hypermodel_compositional_generalization` both explicitly use rank=4 instead, each
  citing a smaller pre-rank-sweep comparison (`arc1d_hypermodel_looped_rope_canon_muon_diag`,
  no longer present as a directory in the tree) rather than the rank_sweep result. Should
  those two experiments be re-run at rank=8 before anything from them goes in the paper?
- **Pooling**: this is the one area where "what won" (attention pooling, per the grid
  README) and "what's actually configured downstream" (attention pooling, confirmed by grep)
  agree — flagged here only for completeness/contrast with the other two axes, not as an open
  problem.
- **Optimizer choice across experiments**: RAdam (capacity_baseline, zhu_block), AdamW
  (arm_adamw in the optimizer experiment, and the AdamW arms of the muon experiments), and
  Muon (muon, muon_sweep, muon_moveablation, and presumably compositional_generalization/
  vae_disentanglement given their "exact same recipe" language) are all live in different
  experiments with no single "this is now the default optimizer" statement anywhere in the
  configs. Which optimizer does the from-scratch rerun standardise on?
- **`hyper_model.num_layers`**: `arc1d_hypermodel_looped_rope_canon_capacity_baseline`
  reports num_layers=8 beating num_layers=4 on stability (std 0.009 vs 0.027) at
  `gradient_clip_val=10`, and this is what `arc1d_hypermodel_looped_rope_canon_muon` onward
  uses — but `arc1d_hypermodel_looped_rope_canon_grid`'s 2×3×3 layers×clip×td grid (whose
  express purpose was to check whether layers and clip interact) does not record a numeric
  verdict in its committed README text. Worth double-checking the grid's actual result data
  before treating num_layers=8 as fully settled.

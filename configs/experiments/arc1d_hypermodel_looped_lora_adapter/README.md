# arc1d_hypermodel_looped_lora_adapter

## Goal

Test a random-backbone + generated LoRA-adapter weight-generation mode against the existing
full-weight-generation baseline, on the same 11-task descriptor mix, at **two** capacity
settings -- mirroring how `arc1d_hypermodel_looped_mix11` itself keeps both `mix11_td.yaml`
and `mix11_n2_loop4_noskip.yaml` as sibling variants rather than picking one:

- **`base.yaml`** -- matches `arc1d_hypermodel_looped_mix11/mix11_td.yaml`'s capacity settings
  (the flip+recolor sweep's winning settings: `n_loops=8`, `N_supervision=4`, block+loop skip
  on).
- **`base_n2_loop4_noskip.yaml`** -- matches
  `arc1d_hypermodel_looped_mix11/mix11_n2_loop4_noskip.yaml`'s capacity settings (the original
  phase-1 default settings: `n_loops=4`, `N_supervision=2`, no block/loop skip).

Both are preserved side by side rather than one replacing the other, so the lora_adapter
mechanism can be checked at both capacity settings independently, same reason the mix11
project keeps both.

### Background: why not the existing `low_rank_output` path

The hypernetwork already has a `hyper_head.low_rank_output` mode (see
`arc1d_hypermodel_augmented_td_low_rank/`), which reportedly performed considerably worse than
full generation when tried against the `rnn` target. Reading the implementation
(`models/hypermodel.py`) surfaced two compounding reasons, not one:

1. **No base to add to.** The rank-`r` outer product *is* the entire returned weight -- unlike
   LoRA, which freezes a competent pretrained `W0` and only trains an additive `ΔW = BA`.
2. **The "low rank" isn't shape-meaningful.** `low_rank_output` flattens *every* target
   parameter (all layers concatenated) into one vector, reshapes it into an arbitrary `m×m`
   square (`m = ceil(sqrt(total_params))`, unrelated to any real weight tensor's shape),
   predicts a rank-`r` outer product of that square, and slices the result back apart into the
   individual weight tensors. Slicing a rank-`r` matrix into a differently-shaped tensor does
   not preserve rank-`r` structure -- so no individual weight matrix was ever actually
   low-rank in a meaningful sense. The constraint only shrank the hyper-head's own output
   layer, not the target model's real expressivity.

## Mechanism

`hyper_head.lora_adapter: true` switches `HyperModel` (`models/hypermodel.py`) to a third
weight-generation mode, alongside the existing full-MLP and `low_rank_output` paths:

- The target model (`RoPECanonLoopedTransformer`) is instantiated with its own ordinary random
  PyTorch init, same as always -- but instead of only using those values as a shape template
  (as the other two modes do), this mode **keeps** them as a persistent backbone `W_backbone`.
- For each of the model's 14 true 2D weight matrices (`input_projection`, and per block x
  {pre, middle, post}: `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`; plus
  `output_head`), the hypernetwork generates a per-tensor rank-`r` delta `ΔW = B @ A`, sized to
  that specific tensor's own `(d_out, d_in)` shape, and the effective weight is
  `W_backbone + ΔW`.
- The remaining 20 non-matrix tensors (LayerNorm `weight`/`bias`, depthwise Canon conv
  kernels -- 2,288 of the model's 11,920 params, 19%) have no meaningful low-rank structure and
  are generated fully from scratch, exactly as the full-generation baseline already does for
  every parameter.
- `B`-heads are zero-initialized, `A`-heads keep their default init (standard LoRA
  convention). Since the backbone is nonzero, `ΔW ≡ 0` at step 0 means the target model starts
  as an ordinary-scale random network, not the degenerate all-zero network the old
  `low_rank_output` path collapsed to before its variance-matching fix
  (`5640fe4`/`e4d5f2d`) -- gradients flow into `B` (and then `A`) from step 0 without needing
  that fix's careful init derivation.
- `hyper_head.lora_adapter_train_backbone` (default `false`) controls whether the backbone
  stays frozen or joins the optimizer as a single shared, jointly-meta-learned copy -- see
  "Not yet built" below.

**Mechanism caveat, worth keeping in mind when reading results:** a *frozen random* backbone
gives the adapter nothing competent to steer, unlike real LoRA's pretrained base. The effect
being tested here is closer to a random-feature-expansion / reservoir-computing argument:
adding a low-rank correction to a fixed *full-rank* matrix (even a random one) breaks the hard
`rank ≤ r` ceiling that a from-scratch low-rank generation imposes, since
`rank(W_random + BA)` generically isn't bounded by `r` the way `rank(BA)` alone is. That's a
different mechanism from LoRA's actual one. A hypernetwork predicting a LoRA-style adapter in
a single forward pass with no gradient descent at inference does have precedent
(HyperDreamBooth, Ruiz et al. 2023, for diffusion-model personalization) -- but its frozen base
is pretrained, not noise, which is the case that would most directly validate "unlock
pretrained models" as a future direction. A positive result here validates the
additive-full-rank-preserving mechanism, not LoRA's actual "reuse pretrained competence"
mechanism.

**Rank-sweep caveat:** most of this target's weight matrices have `min(d_out, d_in) = 16`, so
`r=8` is already half of max possible rank -- that cell sits close to full-rank behavior, not
deeply low-rank.

## Settings

Both `base.yaml` and `base_n2_loop4_noskip.yaml` use the same 11-task subset
(`1d_denoising_1c, 1d_denoising_mc, 1d_fill, 1d_hollow, 1d_mirror, 1d_move_1p,
1d_move_2p_dp, 1d_move_dp, 1d_pcopy_1c, 1d_pcopy_mc, 1d_scale_dp`) and one-hot task descriptor
(`hyper_head.num_tasks: 18`), differing only in `target_model.params`
(`n_loops`/`use_block_skip`/`use_loop_skip`) and the compute-matched `N_supervision`/
`max_steps`/`warmup_steps` triple (`max_steps * N_supervision = 2000` total optimizer
updates in both cases -- 500*4 vs 1000*2). Each is self-contained (no `_base_`, flattening
`base_hypermodel_looped.yaml` + the corresponding `arc1d_hypermodel_looped_mix11` variant's
overrides together) because the config loader only resolves one level of `_base_`
inheritance and both `mix11_td.yaml`/`mix11_n2_loop4_noskip.yaml` already have their own.
The only new axis relative to each baseline is `hyper_head.lora_adapter: true`, so both are
single-variable comparisons against their respective `mix11` sibling.

Rank sweep: `r = 1, 2, 4, 8` for each capacity setting (8 configs total), matching the
historical `low_rank_output` sweep's convention --
`mix11_td_lora_adapter_r{1,2,4,8}.yaml` (`_base_: base.yaml`) and
`mix11_n2_loop4_noskip_lora_adapter_r{1,2,4,8}.yaml`
(`_base_: base_n2_loop4_noskip.yaml`).

## Not yet built

**Trainable/shared backbone variant** (`hyper_head.lora_adapter_train_backbone: true`). The
toggle is implemented (see `models/hypermodel.py`), but no sweep configs exist for it yet.
This variant would make the backbone a single copy, meta-learned jointly across all tasks via
the normal optimizer (still not regenerated per task) -- much closer in spirit to "eventually
plug in a real pretrained model," and the more direct analogue of HyperDreamBooth's setup than
the frozen-random primary experiment above.

## Running

```bash
# Smoke test first (CPU, single task, single batch) -- confirms gradients flow and loss
# drops before committing to the full sweep, same check that caught the historical
# vanishing-gradient bug in the old low_rank_output path.
uv run python train.py --config configs/experiments/arc1d_hypermodel_looped_lora_adapter/overfit/lora_adapter_overfit.yaml
uv run python train.py --config configs/experiments/arc1d_hypermodel_looped_lora_adapter/overfit/standard_overfit.yaml

# Full rank sweep, both capacity settings (8 configs), 3 seeds each -- 24 jobs total.
bash scripts/run_hypermodel_looped_lora_adapter.sh

# Or just the n2_loop4_noskip variant (4 configs, 3 seeds -- 12 jobs).
bash scripts/run_hypermodel_looped_lora_adapter_n2_loop4_noskip.sh
```

## Reading results

Compare `val_query_exact_match` / `test_query_exact_match` (mean ± std across 3 seeds), per
task and averaged, against each variant's own `arc1d_hypermodel_looped_mix11` sibling on the
same 11 tasks -- `mix11_td_lora_adapter_r*` against `mix11_td.yaml`, and
`mix11_n2_loop4_noskip_lora_adapter_r*` against `mix11_n2_loop4_noskip.yaml` -- not hyper-head
parameter count, which is a secondary consideration here (the old low-rank path's motivation
was shrinking the hyper-head; this experiment's motivation is whether the
additive+full-rank-preserving mechanism recovers accuracy, as a step toward eventually
supporting a real pretrained backbone). Comparing the two lora_adapter capacity settings
against each other is also informative in its own right: n_loops=8 gives the per-tensor
adapter more loop iterations to compound its correction through, at 2x the N_supervision.

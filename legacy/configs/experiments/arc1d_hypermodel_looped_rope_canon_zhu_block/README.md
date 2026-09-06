# arc1d_hypermodel_looped_rope_canon_zhu_block

## Goal

`arc1d_hypermodel_looped_rope_canon_capacity_baseline` found `layers=8`, no task descriptor,
8000-step training to be the strongest no-descriptor setting so far (0.738 mean exact-match).
This experiment tests three architecture "tricks" from the same research lineage as this
repo's existing Canon layers: RMSNorm, QK-norm, and a SwiGLU MLP, replacing LayerNorm/GELU in
the hypernetwork encoder's transformer block, while keeping RoPE and the existing Canon A/B/C/D
layers unchanged.

"Zhu" is Zeyuan Allen-Zhu, the author of the Canon Layers paper this repo's Canon mechanism is
already based on (`models/canon_layer.py` cites "Physics of Language Models: Part 4.1,
Architecture Design and the Magic of Canon Layers", NeurIPS 2025). That file's own position
docstring already anticipated a SwiGLU FFN for Canon-D (`"D — inside SwiGLU FFN, on
concatenated gate tensors [x1, x3]"`) but it was never actually built until now; `CanonMLP`
has always used plain GELU. QK-norm and RMSNorm are both specifically stability-oriented
(used in Gemma2/OLMo2/Qwen2 to prevent attention-logit blowup at depth), directly relevant to
this investigation's own repeated gradient-clip-sensitivity findings.

## New class: `RoPECanonZhuTransformer`

`models/rope_looped_transformer.py` now also defines `RoPECanonZhuTransformer` (plus its
supporting `RoPECanonZhuSelfAttention`/`RoPECanonZhuBlock`), structurally identical to the
existing `RoPECanonTransformer` but with three independent boolean flags:

- `use_rmsnorm`: `RMSNorm` (`models/transformer.py`) instead of `LayerNorm` for both
  pre-attention and pre-MLP norms, and the final norm.
- `use_qk_norm`: a per-head `RMSNorm(head_dim)` applied to query and key (separate learnable
  scale for each) right after the per-head reshape, before RoPE rotation.
- `use_swiglu`: `CanonZhuMLP` (`models/canon_transformer.py`) instead of `CanonMLP` -
  `gate_up_proj` → Canon-D on the concatenated `[gate, up]` tensor → SwiGLU → `down_proj`,
  `intermediate_dim` sized to keep total FFN parameters close to the GELU MLP's.

All three flags default to `False`, in which case `RoPECanonZhuTransformer` reproduces
`RoPECanonTransformer`'s exact behaviour (verified directly: byte-identical output given the
same weights). Registered in `HYPERNETWORK_REGISTRY` as `rope_canon_zhu_transformer`.

## Arms

Each sub-trick isolated alone, plus all three combined, per Fabio's direction (not just
tested as one bundled change), against the reused baseline (all three off):

| Arm | RMSNorm | QK-norm | SwiGLU | Δ params vs. baseline |
|---|:---:|:---:|:---:|---:|
| (reused baseline, not rerun) | off | off | off | 0 (reference) |
| `arm_rmsnorm` | **on** | off | off | -64 (-0.02%) |
| `arm_qknorm` | off | **on** | off | +256 (+0.06%) |
| `arm_swiglu` | off | off | **on** | -896 (-0.21%) |
| `arm_zhu_all` | **on** | **on** | **on** | -704 (-0.17%) |

3 seeds each, 12 new jobs. Parameter deltas measured directly at `hidden_dim=64,
num_heads=4, num_layers=8` (the encoder shape used throughout), confirming none of these
changes confound the comparison with a meaningfully different parameter budget. Optimizer/LR
stay at the original `RAdam`/`0.001` throughout (not combined with
`arc1d_hypermodel_looped_rope_canon_optimizer`'s change, so this stays a clean comparison
against the known baseline).

## Not built this round

The remaining 3 unexplored cells of the full 2x2x2 grid (e.g. RMSNorm+QK-norm without
SwiGLU); sandwich/post-norm placement; hypernetwork weight-sharing across layers.

## Running

```bash
bash scripts/run_hypermodel_looped_rope_canon_zhu_block.sh
```

No overfit smoke configs, matching every experiment since the followup; running directly on
the cluster.

## Reading results

Compare each arm's `val_query_exact_match` (mean ± std across 3 seeds) against the reused
baseline's 0.738 mean, per task and averaged. Particular attention to `1d_move_dp`, and to
which of the three sub-tricks (if any) individually moves the needle versus only helping in
combination (`arm_zhu_all` beating all three isolated arms would suggest a real interaction
effect, worth a follow-up covering the remaining grid cells).

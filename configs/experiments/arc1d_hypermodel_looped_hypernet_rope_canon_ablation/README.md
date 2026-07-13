# arc1d_hypermodel_looped_hypernet_rope_canon_ablation

## Goal

Isolate which architectural change to the hypernetwork **encoder** (not the target model,
which is held fixed) improves task-context reading: Canon convolutions, RoPE, or both. Every
`hyper_model.params` value is kept as close to today's shipped `transformer` convention
(`hidden_dim=64, num_heads=4, num_layers=2, output_dim=64`) as each class's constructor
allows — the goal is config-similarity to "current," not parameter- or block-count-matching.
This deliberately does **not** follow `arc1d_uniform_ablation`'s own precedent of bumping
`num_layers` to 3 for block/parameter parity with a looped-at-`n_loops=1` RoPE class; here
`num_layers` stays `2` everywhere it applies, and the RoPE arms use a genuinely flat
`num_layers`-controlled class instead (see "New class" below).

Target model: **fixed** across all 4 arms, exactly matching
`arc1d_hypermodel_looped_lora_adapter/base_n2_loop4_noskip.yaml`'s `target_model` block
(`rope_canon_looped_transformer`, `hidden_dim=16, num_heads=2, inner_dim=16,
inner_num_heads=2, n_loops=4, canon_set=ABCD`). Weight generation:
`hyper_head.lora_adapter=true, lora_adapter_rank=1` (fixed, reused as-is from
`arc1d_hypermodel_looped_lora_adapter`, not re-validated here). `N_supervision=2,
max_steps=2000, warmup_steps=200` — trains longer than that project's `max_steps=1000`. All
17 task categories (`TASK_CATEGORY_INDEX` minus `1d_padded_fill`) — up from that project's
11-task subset.

## New class: `RoPECanonTransformer`

No flat (non-looped, `num_layers`-controlled) RoPE-capable transformer existed in this repo
before this experiment. The only RoPE class was `RoPECanonLoopedTransformer`
(`models/rope_looped_transformer.py`), whose `pre_layer → middle_layer(×n_loops, weight
shared) → post_layer` structure gives 3 blocks at `n_loops=1`, not 2 layers.
`models/rope_looped_transformer.py` now also defines `RoPECanonTransformer`: structurally
identical to `models/canon_transformer.py`'s `CanonTransformer` (`input_projection → 
ModuleList of num_layers RoPECanonBlocks → final_norm → optional output_head`), reusing the
existing `RoPECanonBlock` (already used inside the looped class) unmodified. Confirmed by
direct instantiation: `RoPECanonTransformer(canon_set="ABCD")` and `CanonTransformer(canon_set="ABCD")`
have identical parameter counts at the same `hidden_dim`/`num_layers` (RoPE is
parameter-free), so `canon_set=""` vs. `"ABCD"` is the only thing that changes parameter
count between arms 2 and 4.

## Arms

| Arm | `hyper_model.name` | `canon_set` | `num_layers` | `use_sinusoidal_pe` | Config file |
|---|---|:---:|:---:|:---:|---|
| 1. Current | `transformer` | n/a | 2 | `true` | `hypernet_current.yaml` |
| 2. +RoPE | `rope_canon_transformer` | `""` | 2 | `false` | `hypernet_rope.yaml` |
| 3. +Canon ABCD | `canon_transformer` | `ABCD` | 2 | `true` | `hypernet_canon.yaml` |
| 4. +RoPE+Canon | `rope_canon_transformer` | `ABCD` | 2 | `false` | `hypernet_rope_canon.yaml` |

The 4 arms form a clean 2×2 grid (Canon on/off × RoPE on/off) with `num_layers=2` and
`hidden_dim=64/num_heads=4/output_dim=64` constant throughout.

## Background: `task_encoding.use_sinusoidal_pe`

`HyperModelLightning` previously never read `task_encoding.use_sinusoidal_pe` — every run
got additive sinusoidal PE unconditionally, including RoPE-based encoders/targets (whose own
docstrings say the input embedder should have `use_sinusoidal_pe=False` to avoid
double-encoding position). This is now wired through (mirroring
`DirectSupervisedLightning`/`LoopedSupervisedLightning`'s existing pattern), defaulting to
`true` so no existing config's behavior changes.

**Two caveats, disclosed rather than engineered around:**

- **PE-stacking on the fixed target model.** `HyperModelLightning` has exactly one shared
  `TaskTokenEmbedder` feeding both the encoder's input and the target model's input. The
  target model's RoPE is unconditional (applied regardless of `canon_set`), so its
  *architecture* is genuinely fixed across all 4 arms — but its *effective input positional
  treatment* isn't perfectly constant: arms 1/3 give the target both its own RoPE and
  redundant external sinusoidal PE; arms 2/4 give it only its own RoPE. This is a real,
  second-order effect (RoPE is generally understood to dominate over an embedding-layer-only
  additive PE, but that assumption hasn't been previously validated in this repo — no
  existing codepath has one embedder feeding two structurally different consumers like this).
  Not worth blocking on or building a per-consumer override for; noted here as an accepted
  caveat.
- **ReLU vs. GELU FFN activation.** `Transformer` (arm 1) has a configurable FFN activation
  (`activation: str = "relu"`, unset in every existing config → ReLU). `CanonTransformer` and
  `RoPECanonTransformer`'s FFN (`CanonMLP`) both hardcode `nn.GELU()` with no activation
  parameter at all. So arms 2–4 are unconditionally GELU-activated regardless of RoPE/Canon
  settings, while arm 1 is ReLU — not a pure RoPE/Canon-presence comparison. Per the "match
  current as closely as possible" goal, arm 1 is not changed to GELU to compensate.

## Running

```bash
# Smoke test first (CPU, single task, single batch), one per arm.
uv run python train.py --config configs/experiments/arc1d_hypermodel_looped_hypernet_rope_canon_ablation/overfit/hypernet_current_overfit.yaml
uv run python train.py --config configs/experiments/arc1d_hypermodel_looped_hypernet_rope_canon_ablation/overfit/hypernet_rope_overfit.yaml
uv run python train.py --config configs/experiments/arc1d_hypermodel_looped_hypernet_rope_canon_ablation/overfit/hypernet_canon_overfit.yaml
uv run python train.py --config configs/experiments/arc1d_hypermodel_looped_hypernet_rope_canon_ablation/overfit/hypernet_rope_canon_overfit.yaml

# Full sweep (4 configs x 3 seeds = 12 jobs).
bash scripts/run_hypermodel_looped_hypernet_rope_canon_ablation.sh
```

## Reading results

Compare `val_query_exact_match`/`test_query_exact_match` (mean ± std across 3 seeds) across
the 4 arms, per task and averaged; read both as a 4-way comparison and as two independent
main effects (Canon, RoPE). Not directly comparable to
`arc1d_hypermodel_looped_lora_adapter`'s existing results without accounting for two
differences: 17 tasks here vs. 11 there, and `max_steps=2000` here vs. `max_steps=1000`
there.

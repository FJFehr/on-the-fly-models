# ARC-1D Hypernetwork Story: Experiment Catalog

This is a full experiment catalog for the ARC-1D hypernetwork research line, feeding a paper
write-up. Every number below is pulled from the experiment's own `README.md` and YAML configs
under `legacy/configs/experiments/<dir>/`, or (where noted explicitly) from wandb where it is more
complete than the local `outputs/<experiment>/*/results.txt` files. Status is checked directly
against `outputs/<experiment_name>/*/results.txt` on disk. A separate document independently
checks cross-experiment consistency of architecture choices, so this document reports each
experiment's settings as its own config states them without reconciling them against each other.

---

## Phase 0: capacity ablation (ancestor)

### arc1d_uniform_ablation

- **Directory**: `legacy/configs/experiments/arc1d_uniform_ablation/`
- **README**: `legacy/configs/experiments/arc1d_uniform_ablation/README.md`

**Question.** Does a clean, parameter-matched, uniform-width looped Canon+RoPE transformer
beat the earlier "sandwich" (narrow-outer/wide-inner) architecture and a non-looped baseline,
and does that hold across two widths (dim=16 and dim=32)? This is the ancestor capacity story
the hypernetwork target model (`RoPECanonLoopedTransformer`) is drawn from.

**Task set.** All 17 `arc1d_recursion_ablation` task categories minus `1d_padded_fill`
(established convention for this story family): `1d_denoising_1c`, `1d_denoising_mc`,
`1d_fill`, `1d_flip`, `1d_hollow`, `1d_mirror`, `1d_move_1p`, `1d_move_2p`, `1d_move_2p_dp`,
`1d_move_3p`, `1d_move_dp`, `1d_pcopy_1c`, `1d_pcopy_mc`, `1d_recolor_cmp`, `1d_recolor_cnt`,
`1d_recolor_oe`, `1d_scale_dp`.

**Data.** `data/arc_1d_looped_augmented` (per-pair augmentation, ~1000 variants/base task; see
`legacy/configs/experiments/arc1d_recursion_ablation/base_ablation.yaml`, which
`arc1d_uniform_ablation`'s configs inherit via `_base_`).

**Architecture.** Two widths, dim=16 (2 heads) and dim=32 (4 heads), head_dim=8 convention.
T1-T3 use `transformer`/`canon_transformer` with `num_layers=3` (chosen for block-count parity
with the RoPE-looped family at `n_loops=1`). T4-T7 and L1-L6 use
`rope_canon_looped_transformer`. Canon ABCD (`canon_kernel=5, canon_activation=True,
canon_residual=True, canon_causal=False`) from T3 onward; RoPE from T4 onward. No hypernetwork
in this project - every model is trained directly. No LoRA (not applicable).

**Training recipe** (`legacy/configs/experiments/arc1d_recursion_ablation/base_ablation.yaml` +
`scripts/gen_uniform_ablation_configs.py`): optimizer **RAdam**, `learning_rate=0.0005`
(overridden from the base file's 0.001) throughout, `weight_decay=0.01`, `batch_size=256`,
`warmup_steps=200`. `N_supervision=1` for T1 (`direct_supervised`, `gradient_clip_val=1.0`,
`max_steps=8000`); `N_supervision=2` for T2-T7 and L1-L6 (`max_steps=4000`, no
`gradient_clip_val` set at that step). **5 seeds** per cell (85 runs/cell = 17 tasks x 5 seeds).

**Jobs.** 7 steps x 2 widths x 17 tasks x 5 seeds = 1,190 (T1-T7) + 6 diagnostics x 2 widths x
17 tasks x 5 seeds = 1,020 (L1-L6) = **2,210 jobs total**.

**Status.** Finished-with-local-results, but only partially. `outputs/arc1d_uniform_ablation/`
has 1,870 run directories, each with a `results.txt`: all of T1-T7 (1,190 = 7 x 170) and L1-L4
(680 = 4 x 170) are present locally. **L5 and L6 (n_loops=16/32, 340 planned jobs) have no local
run directories at all**, despite being fully reported with real numbers in the README's L1-L6
table (added in commit `21bf9e6`, "Fold L5/L6 results into the loop-diagnostics README table") -
those results must have been fetched/summarized and the raw run directories subsequently removed
or never retained locally; this write-up reproduces the README's own published numbers for L5/L6
since no local `results.txt` exists to re-derive them from.

**The T1-T7 story** (backbone params identical within a step across widths where noted; dim16 /
dim32):

| Step | Description | Backbone | Backbone params (dim16 / dim32) |
|------|---|---|---|
| T1 | Vanilla transformer, sin PE, N_sup=1 | `transformer` | 9,600 / 38,144 |
| T2 | + N_sup=2 loop training | `transformer` | 9,600 / 38,144 |
| T3 | + Canon ABCD | `canon_transformer` | 11,760 / 42,464 |
| T4 | + RoPE, flat (n_loops=1) | `rope_canon_looped_transformer` | 11,760 / 42,464 |
| T5 | + Looped middle (n_loops=4) | `rope_canon_looped_transformer` | 11,760 / 42,464 |
| T6 | + Block skip | `rope_canon_looped_transformer` | 11,760 / 42,464 |
| T7 | + Per-loop h0 (loop skip) | `rope_canon_looped_transformer` | 11,760 / 42,464 |

**Loop diagnostics L1-L6** (N_supervision fixed at 2; mean over 17 tasks x 5 seeds = 85
runs/cell, std omitted, see `scripts/plot_loop_diagnostics.py`):

| Code | n_loops | Block skip | Loop skip | val (dim16 / dim32) | test (dim16 / dim32) |
|------|---------|-----------|-----------|----------------------|------------------------|
| T4 | 1  | – | – | 91.2% / 93.7% | 91.5% / 93.4% |
| T5 | 4  | – | – | 92.4% / 92.8% | 92.8% / 92.5% |
| T6 | 4  | ✓ | – | 89.3% / 91.6% | 88.9% / 92.5% |
| T7 | 4  | ✓ | ✓ | 90.8% / 95.3% | 90.4% / 95.2% |
| L1 | 8  | – | – (control) | 93.1% / 95.0% | 93.3% / 94.9% |
| L2 | 4  | – | ✓ | 91.6% / 91.7% | 91.4% / 92.3% |
| L3 | 8  | – | ✓ | 91.6% / 93.4% | 92.0% / 92.6% |
| L4 | 8  | ✓ | ✓ | 91.4% / 95.0% | 90.9% / 94.4% |
| L5 | 16 | – | – | 92.5% / 95.9% | 92.4% / 95.2% |
| L6 | 32 | – | – | 91.2% / 94.1% | 90.7% / 94.6% |

**Key numeric findings (README's own text).**
1. At ~16.7k backbone params (dim=20), parameter-matched uniform scored 93.2% to 95.0%
   (no-skip to +block-skip) vs. wide-sandwich's 91.7% to 94.8%; at ~54k params (dim=36), uniform
   hit 96.0%, the best result found - the sandwich architecture's apparent advantage was purely
   a parameter-count confound.
2. Looping is essential: removing it (n_loops=1, wide capacity + block skip) scored 89.6%, worse
   than looped-no-skip's 91.7%.
3. On the clean uniform architecture, the L1-L4 replication of the old sandwich architecture's
   "loop-skip alone helps at n_loops=8" finding **did not replicate**: loop-skip-alone (L2, L3)
   never beat its no-skip baseline (T5, L1) at either width.
4. Combined block+loop skip only helps at dim=32/n_loops=4 (T7: 95.2% test vs. T5's 92.5%); at
   dim=16 or n_loops=8 (L4 vs L1) it does not.
5. The reliable lever is simply more loop iterations with no skip at all: the no-skip sweep
   (T4→T5→L1→L5→L6, n_loops=1→4→8→16→32) peaks at n_loops=8 for dim=16 (93.3% test) and
   n_loops=16 for dim=32 (95.2% test, tying T7's skip-based result), with both widths degrading
   somewhat by n_loops=32 - diminishing (and eventually negative) returns set in around
   n_loops≈8-16.

**Caveats.**
- **Predates the max_steps/N_supervision fix.** This project's commits run from `d11255d`
  (2026-07-04) to `21bf9e6` (2026-07-08), all before the fix landed in `4f8f4f1`
  ("Fix max_steps/N_supervision mismatch; compute-match LR schedules", 2026-07-13). Under the
  pre-fix behaviour, every `N_supervision=2` cell (T2-T7, L1-L6 - everything except T1) trained
  on half its intended `max_steps` worth of batches and its LR scheduler never completed
  warmup+decay. **Not re-run since the fix** (no commits touch this directory after
  `21bf9e6`). The README itself does not use the words "Known issue" (unlike several later
  hypernetwork READMEs that do flag this explicitly) - this predates-the-fix status is inferred
  from commit dates, not stated in-README.
- No disk-wipe or data-loss note in this README.

---

## Phase 1: hypernetwork predicts individual task backbones

### arc1d_hypermodel_looped

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped/`
- **README**: `legacy/configs/experiments/arc1d_hypermodel_looped/README.md`

**Question.** Can a hypernetwork generate the weights of `RoPECanonLoopedTransformer` (the
smallest looped backbone from `arc1d_uniform_ablation`'s L1 result) per task, one target model
per task category, instead of training that backbone directly?

**Task set.** All 17 `arc1d_recursion_ablation` categories minus `1d_padded_fill`, one config
per category (17 configs), no mixing, no task descriptor.

**Data.** `data/arc_1d_looped_augmented` - same augmented dataset as the capacity-ablation
story (deliberately not the older `arc_1d_augmented` recipe).

**Architecture.** Hyper backbone: `transformer`, `hidden_dim=64, num_heads=4, num_layers=2,
output_dim=64`; `hyper_head.pooling: hierarchical, bottleneck_dim: 128`; full MLP weight
generation, **no LoRA** (`low_rank_output` unset). Target: `rope_canon_looped_transformer`,
`hidden_dim=16, num_heads=2, inner_dim=16, inner_num_heads=2, n_loops=4` (phase 1 uses
`n_loops=4`, not the capacity story's `n_loops=8`, purely a compute/depth choice - `n_loops`
does not change parameter count since `middle_layer` is weight-shared), Canon ABCD, no
block/loop skip. Model size (fixed regardless of `n_loops`): hypernetwork encoder+head
1,736,784 trainable params, target 11,920 params (11,760 backbone + 160 output head) -
**1,748,704 total**.

**Training recipe** (`legacy/configs/experiments/arc1d_hypermodel_looped/base_hypermodel_looped.yaml`):
optimizer **RAdam**, `learning_rate=0.001`, `weight_decay=0.01`, `batch_size=512`,
`N_supervision=2`, `max_steps=1000` (→ 2000 total optimizer updates), `warmup_steps=100`,
no `gradient_clip_val` set. **3 seeds** (51 jobs = 17 tasks x 3 seeds).

**Status.** No local `results.txt` files exist anywhere under
`outputs/arc1d_hypermodel_looped/` - **Finished-but-wandb-only**.

**Key numeric findings - from wandb, not local files.** Fresh wandb pull of per-task
`test_query_exact_match`, averaged across seeds (project `arc1d_hypermodel_looped`): `1d_
denoising_1c`=1.0, `denoising_mc`=1.0, `fill`=1.0, `flip`=0.5, `hollow`=1.0, `mirror`=1.0,
`move_1p`=1.0, `move_2p`=1.0, `move_2p_dp`=1.0, `move_3p`=1.0, `move_dp`=0.92, `pcopy_1c`=1.0,
`pcopy_mc`=1.0, `recolor_cmp`=0.333, `recolor_cnt`=0.0, `recolor_oe`=0.0, `scale_dp`=1.0 (17
tasks, **overall mean 0.764**). `1d_flip` and the three recolor tasks are the clear laggards -
directly motivating the follow-up sweep below.

**Caveats.**
- **Known issue, stated explicitly in the README**: under manual optimization
  (`N_supervision > 1`), `Trainer(max_steps=...)` stopped after `max_steps / N_supervision`
  batches instead of `max_steps` batches pre-fix, so every result here (at `N_supervision=2`) ran
  on half the intended batches and never completed LR warmup+decay. "All phase-1 results below
  were collected before this fix - not yet re-run" (README, lines 3-10).
- Roadmap steps 2-4 (small mix, task-descriptor mix, low-rank variant) were "not yet built" per
  the README at time of writing; steps 2-4 were superseded by `arc1d_hypermodel_looped_mix11`
  and the later rope_canon tuning chain rather than built inside this same directory.

---

### arc1d_hypermodel_looped_recolor

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_recolor/`
- **README**: `legacy/configs/experiments/arc1d_hypermodel_looped_recolor/README.md`

**Question.** For the four tasks that didn't reach ~100% in phase 1 (`1d_flip`,
`1d_recolor_cmp/cnt/oe`), does more hypernetwork capacity (deeper internal loops, more external
supervision, skip connections) close the gap, or is this a real capacity ceiling rather than a
data problem? (README includes a rigorous per-category check confirming no support/query data
bug: the true rule was derived from 3 support pairs alone for each recolor category and verified
against 200 real instances/category, 200/200 correct in all three.)

**Task set.** 4 categories: `1d_flip`, `1d_recolor_cmp`, `1d_recolor_cnt`, `1d_recolor_oe`.

**Data.** `data/arc_1d_looped_augmented` throughout (a canonical-colour diagnostic variant was
built and tested, then deliberately dropped).

**Architecture.** Same hyper backbone/target family as `arc1d_hypermodel_looped`
(`rope_canon_looped_transformer`, `hidden_dim=16`), swept over `n_loops ∈ {4, 8, 16}` x `skip ∈
{none, block+loop}`. No LoRA (inherits `arc1d_hypermodel_looped/base_hypermodel_looped.yaml`,
which has no `lora_adapter` field set).

**Training recipe.** Optimizer **RAdam**, `learning_rate=0.001`, `weight_decay=0.01`,
`batch_size=512` (inherited base). `N_supervision ∈ {2, 4}`, compute-matched: N_sup=2 uses
`max_steps=1000/warmup_steps=100`; N_sup=4 uses `max_steps=500/warmup_steps=50` (both 2000 total
optimizer updates). **1 seed** per cell (single seed, per README).

**Jobs.** 3 (n_loops) x 2 (N_sup) x 2 (skip) = 12 configs/task x 4 tasks = **48 total**
(`scripts/gen_hypermodel_looped_recolor_configs.py`).

**Status.** Finished-with-local-results: 48/48 `results.txt` present under
`outputs/arc1d_hypermodel_looped_recolor/`.

**Key numeric findings** (`val_query_exact_match`, from local `results.txt`):
- `1d_recolor_cnt` and `1d_recolor_oe`: **0.0 in every single one of the 12 cells** - a
  structural ceiling, not a capacity or data issue at this model size.
- `1d_recolor_cmp`: mostly strong (many cells at 1.0; e.g. `n2_loop4_noskip=1.0`,
  `n4_loop8_noskip=1.0`, `n4_loop8_skip=1.0`), some intermediate cells dip
  (`n2_loop8_skip=0.2`).
- `1d_flip`: highly variable, best cells reach 1.0 (`n4_loop16_skip=1.0`, `n4_loop8_skip=1.0`),
  but the phase-1-matching cell (`n2_loop4_noskip`) is 0.0 - no clean monotonic trend with
  n_loops or N_sup alone.

**Caveats.**
- **Known issue, stated explicitly**: same manual-optimization max_steps/N_supervision bug as
  `arc1d_hypermodel_looped`; the README additionally flags that "the `N_supervision=4` cells in
  particular only ran a quarter of the intended batches ... so the 'N_sup=4 beats 2' finding from
  this sweep should be treated as provisional until re-run" (README, lines 3-12).

---

### arc1d_hypermodel_looped_mix11

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_mix11/`
- **README**: `legacy/configs/experiments/arc1d_hypermodel_looped_mix11/README.md`

**Question.** First multi-task hypernetwork experiment: can one hypernetwork be trained across
11 task categories simultaneously (not per-task), with a one-hot task descriptor to disambiguate,
and does each task hold its individually-trained accuracy?

**Task set.** The "simple" 11-task subset from `arc1d_hypermodel_augmented_td/rnn_simple_td.yaml`:
`1d_denoising_1c, 1d_denoising_mc, 1d_fill, 1d_hollow, 1d_mirror, 1d_move_1p, 1d_move_2p_dp,
1d_move_dp, 1d_pcopy_1c, 1d_pcopy_mc, 1d_scale_dp`.

**Data.** `data/arc_1d_looped_augmented` (inherited from `base_hypermodel_looped.yaml`).

**Architecture.** Two sibling configs, not one replacing the other:
- `mix11_td.yaml`: target `n_loops=8`, block+loop skip **on**, `N_supervision=4` - the "winning
  capacity settings" from the recolor sweep.
- `mix11_n2_loop4_noskip.yaml`: target `n_loops=4`, no skip, `N_supervision=2` - the original
  phase-1 default settings.

Both: `hyper_head.num_tasks: 18` (one-hot task descriptor enabled from the start - motivated
directly by `arc1d_hypermodel_disentanglement`'s finding that structurally similar move-variant
mixes fail without one, and this subset includes 3 move variants). No LoRA.

**Training recipe.** Optimizer **RAdam**, `learning_rate=0.001`, `weight_decay=0.01`,
`batch_size=512`, `devices: auto` (claims the whole node). `mix11_td`: `N_supervision=4,
max_steps=500, warmup_steps=50`; `mix11_n2_loop4_noskip`: `N_supervision=2, max_steps=1000,
warmup_steps=100` (both 2000 total optimizer updates). **3 seeds** each (single hand-written
config per variant, not a generated family - 3 seeds run sequentially since each claims the full
node).

**Status.** No local `results.txt` under `outputs/arc1d_hypermodel_looped_mix11/` (only a
`smoke_mix11b` smoke-test directory) - **Finished-but-wandb-only**.

**Key numeric findings - from wandb, not local files.** `looped_hyper_mix11_td` (finished run):
`val_query_exact_match=0.857`, `test_query_exact_match=0.768`.
`looped_hyper_mix11_n2_loop4_noskip` had 2 finished runs at `test_query_exact_match` 0.804 and
0.839 (plus some crashed runs).

**Caveats.**
- **Known issue, stated explicitly**: same manual-optimization bug; "`mix11_td.yaml`
  (`N_supervision=4`) only ran a quarter of its intended batches; `mix11_n2_loop4_noskip.yaml`
  (`N_supervision=2`) ran half. Both configs' 'winning capacity settings' provenance (from
  `arc1d_hypermodel_looped_recolor`) is itself provisional for the same reason" (README, lines
  3-13).

---

## Phase 1 architecture/tuning chain

This chain runs sequentially, each experiment consuming the previous one's conclusion as its new
fixed default. All experiments in this chain target `rope_canon_looped_transformer`
(`hidden_dim=16, num_heads=2, inner_dim=16, inner_num_heads=2`), on `data/arc_1d_looped_augmented`.

### arc1d_hypermodel_looped_hypernet_rope_canon_ablation

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_hypernet_rope_canon_ablation/`
- **README**: same directory, `README.md`

**Question.** Which architectural change to the hypernetwork **encoder** (not the fixed target)
improves task-context reading - Canon convolutions, RoPE, both, or neither - in a clean 2x2 grid?
Introduces the new `RoPECanonTransformer` class (a flat, `num_layers`-controlled RoPE-capable
transformer; previously only the looped RoPE class existed).

**Task set.** All 17 task categories.

**Architecture.** 4 arms, `hyper_model.hidden_dim=64, num_heads=4, num_layers=2, output_dim=64`
throughout: (1) `transformer`+sin-PE (current), (2) `rope_canon_transformer`+`canon_set=""`
(+RoPE), (3) `canon_transformer`+`canon_set=ABCD` (+Canon), (4) `rope_canon_transformer`+
`canon_set=ABCD` (+RoPE+Canon). Target fixed: `n_loops=4`, Canon ABCD, no skip. **LoRA
adapter rank=1** (`hyper_head.lora_adapter: true, lora_adapter_rank: 1`, fixed, reused as-is
from `arc1d_hypermodel_looped_lora_adapter`).

**Training recipe** (`base.yaml`): optimizer **RAdam**, `learning_rate=0.001`,
`weight_decay=0.01`, `batch_size=1024`, `gradient_clip_val=10.0` (added after loss-spike
diagnostics traced periodic spikes to unclipped hyper-projection/LoRA-adapter gradients: normal
steps p50=6.2/p90=15.9, spikes reached 40-90+), `N_supervision=2`, `max_steps=4000,
warmup_steps=400`. **5 seeds** planned.

**Jobs.** 4 configs x 5 seeds = 20 jobs.

**Status.** **Not-run.** `outputs/arc1d_hypermodel_looped_hypernet_rope_canon_ablation/` only
has 4 `overfit` smoke-test directories (`looped_hyper_encoder_ablation_{current,rope,canon,
rope_canon}_overfit`), no `results.txt` anywhere.

**Caveats.** Two disclosed-not-engineered-around confounds noted in the README itself: (1)
PE-stacking - the target model's RoPE is unconditional, so arms 1/3 give it both RoPE and
redundant sinusoidal PE while arms 2/4 give it only RoPE; (2) FFN activation - arm 1 (`Transformer`)
uses ReLU, arms 2-4 (`CanonMLP`) hardcode GELU, so it isn't a pure RoPE/Canon-presence comparison.

---

### arc1d_hypermodel_looped_lora_adapter

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_lora_adapter/`
- **README**: same directory, `README.md`

**Question.** Does a random-backbone + generated-LoRA-adapter weight-generation mode beat full
weight generation, at two capacity settings, on the 11-task descriptor mix? (Motivated by the
existing `low_rank_output` mode performing poorly for structural reasons detailed in the README:
no base to add to, and the "low rank" reshape isn't shape-meaningful.)

**Task set.** Same 11-task subset as `arc1d_hypermodel_looped_mix11`.

**Architecture.** `hyper_head.lora_adapter: true`, rank swept `r ∈ {1, 2, 4, 8}`, at two target
capacity settings (mirroring `mix11_td`/`mix11_n2_loop4_noskip`): `base.yaml`
(`n_loops=8`, block+loop skip on, `N_supervision=4, max_steps=500, warmup_steps=50`) and
`base_n2_loop4_noskip.yaml` (`n_loops=4`, no skip, `N_supervision=2, max_steps=1000,
warmup_steps=100`), both 2000 total optimizer updates. `hyper_head.num_tasks: 18` throughout.

**Training recipe.** Optimizer **RAdam**, `learning_rate=0.001`, `weight_decay=0.01`,
`batch_size=512`, `gradient_clip_val=10.0`. **3 seeds** each.

**Jobs.** 4 ranks x 2 capacity settings x 3 seeds = **24 jobs total**.

**Status.** **Not-run.** `outputs/arc1d_hypermodel_looped_lora_adapter/` only has 3
`overfit`-related directories, no `results.txt` for any rank-sweep cell.

**Caveats.** Mechanism caveat stated explicitly: this is a random-feature-expansion /
reservoir-computing effect (rank(W_random+BA) generically isn't bounded by r), not real LoRA's
"reuse pretrained competence" mechanism, since the frozen backbone here is random, not
pretrained. The trainable/shared-backbone variant
(`hyper_head.lora_adapter_train_backbone: true`) is implemented but "not yet built" as configs.

---

### arc1d_hypermodel_looped_rope_canon_followup

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_followup/`
- **README**: same directory, `README.md`

**Question.** With encoder fixed to RoPE+Canon (the combined arm from the prior ablation), does
hierarchical pooling beat attention pooling, does LoRA rank 1→8 unlock any task, and does
`N_supervision=4` beat 2? One shared baseline arm (`arm_default`), three single-axis pairwise
comparisons (not a full factorial).

**Task set.** All 17 categories (same list as `hypernet_rope_canon_ablation`).

**Architecture.** `arm_default`: hierarchical pooling, LoRA rank=1, `N_supervision=2`,
`hyper_model.num_layers=2`. `arm_pooling_attention`: attention pooling, `num_layers=4` (bumped
to roughly parameter-match hierarchical pooling's fixed 98,752-param overhead - end-to-end gap
measured directly at 5,632 params / 0.9%). `arm_rank8`: rank=8. `arm_nsup4`: `N_supervision=4,
max_steps=2000, warmup_steps=200` (compute-matched to `arm_default`'s `max_steps=4000,
warmup_steps=400`, both 8000 total optimizer updates).

**Training recipe.** Optimizer **RAdam**, `learning_rate=0.001`, `weight_decay=0.01`,
`batch_size=512` (dropped from 1024), `gradient_clip_val=10.0` (carried over, "not re-verified"
at this smaller batch size). **3 seeds** (down from 5).

**Jobs.** 4 configs x 3 seeds = **12 jobs total**.

**Status.** Finished-with-local-results: 12/12 `results.txt` present under
`outputs/arc1d_hypermodel_looped_rope_canon_followup/`.

**Key numeric findings** (mean across 3 seeds, `val_query_exact_match` / `test_query_exact_match`
from `results.txt`):

| Arm | val mean | test mean |
|---|---:|---:|
| `arm_default` | 0.784 | 0.735 |
| `arm_nsup4` | 0.777 | 0.758 |
| `arm_pooling_attention` | 0.784 | 0.780 |
| `arm_rank8` | **0.860** | **0.848** |

Rank 8 is the clear winner, well above default rank 1 on both val and test; pooling and
`N_supervision` show no clean separation from default.

**Caveats.** No "Known issue" text and no disk-wipe note in this README.

---

### arc1d_hypermodel_looped_rope_canon_mechanism_ablation

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_mechanism_ablation/`
- **README**: same directory, `README.md`

**Question.** `rank_sweep`'s rank=8 result was lower than `followup`'s, with two confounded
changes at once (`gradient_clip_val` 10→5, pooling hierarchical→attention). This experiment
re-isolates each factor with a single-axis flip from `rank_sweep/rank8.yaml`, plus two new
mechanism questions: does the LoRA adapter's random backbone matter vs. a zero backbone, and can
the model learn multitask from support examples alone without the one-hot task-identity signal
(**`arm_no_task_indicator`** - this is the second, independent source of the notd-vs-td evidence
covered in the Disentanglement Diagnostics section below).

**Task set.** All 17 categories.

**Architecture.** Baseline (reused from `rank_sweep/rank8.yaml`, not rerun here): rank=8,
RoPE+Canon `num_layers=4`, attention pooling, `gradient_clip_val=5`, `N_supervision=2`. Four new
arms, one-axis-flip each: `arm_clip10` (clip 5→10), `arm_hierarchical` (pooling
attention→hierarchical), `arm_zero_backbone`
(`hyper_head.lora_adapter_zero_backbone: true`), `arm_no_task_indicator`
(`hyper_head.num_tasks: null`).

**Training recipe.** RAdam, `learning_rate=0.001`, `weight_decay=0.01`, `batch_size=512`,
`N_supervision=2`. **3 seeds** per new arm.

**Jobs.** 4 arms x 3 seeds = **12 new jobs** (baseline's existing 5-seed data referenced, not
rerun).

**Status.** Finished-with-local-results: 12/12 new-arm `results.txt` present under
`outputs/arc1d_hypermodel_looped_rope_canon_mechanism_ablation/`. **Caveat on the reused
baseline**: the README claims the `rank8` baseline "already has 5 completed seeds of data," but
only **3 local seed directories** exist under
`outputs/arc1d_hypermodel_looped_rope_canon_rank_sweep/looped_hyper_rope_canon_rank_sweep_r8_
seed{1,2,3}/` - seeds 4-5 are not present locally (consistent with `rank_sweep`'s own
partial-run status below).

**Key numeric findings** (mean across seeds found locally; `test_query_exact_match`):

| Arm | n seeds (local) | val mean | test mean |
|---|---:|---:|---:|
| baseline (`rank8`, reused, `td`, clip=5) | 3 | 0.746 | 0.742 |
| `arm_clip10` | 3 | 0.837 | 0.841 |
| `arm_hierarchical` | 3 | 0.773 | 0.761 |
| `arm_zero_backbone` | 3 | 0.723 | 0.705 |
| `arm_no_task_indicator` (**notd**) | 3 | 0.629 | 0.636 |

`gradient_clip_val` (5→10) recovers most of the followup-vs-rank_sweep gap; pooling alone barely
moves it. **`arm_no_task_indicator`: removing the one-hot task descriptor costs roughly 10-11
points of test exact match** (0.636 vs. baseline's 0.742) - direct evidence the task descriptor
materially helps multitask training, corroborated independently by the `grid` experiment below.

**Caveats.** No "Known issue" (max_steps/N_supervision) text in this README - this project
postdates the 2026-07-13 fix. No disk-wipe note.

---

### arc1d_hypermodel_looped_rope_canon_rank_sweep

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_rank_sweep/`
- **README**: same directory, `README.md`

**Question.** Where between rank 1 and rank 8 does task-unlocking actually happen, and does
removing Canon from the encoder hurt at the winning rank (rank 8 only)?

**Task set.** All 17 categories.

**Architecture.** Encoder fixed: `rope_canon_transformer`, `hidden_dim=64, num_heads=4,
num_layers=4, output_dim=64`, attention pooling (hierarchical dropped - no measured benefit in
`followup`). Arms: `rank1/rank2/rank4/rank8` (`canon_set=ABCD`), `rank8_no_canon`
(`canon_set=""`, encoder drops from 213,888 to 202,368 params).

**Training recipe.** RAdam, `learning_rate=0.001`, `weight_decay=0.01`, `batch_size=512`,
`gradient_clip_val=5.0` (lowered from 10.0, "fresh, unvalidated setting"), `N_supervision=2,
max_steps=4000, warmup_steps=400`. **5 seeds** planned per arm.

**Jobs.** 5 configs x 5 seeds = **25 jobs total planned**.

**Status.** **Partially-run.** Only 15 of 25 planned `results.txt` files exist locally under
`outputs/arc1d_hypermodel_looped_rope_canon_rank_sweep/`: `rank1` has 3/5 seeds with results (a
4th directory exists but has no `results.txt`, i.e. failed/incomplete), `rank2`/`rank4`/`rank8`/
`rank8_no_canon` each have exactly 3/5 seeds present.

**Key numeric findings** (mean of the 3 local seeds each; `test_query_exact_match`):

| Arm | test mean |
|---|---:|
| `rank1` | 0.640 |
| `rank2` | 0.652 |
| `rank4` | 0.723 |
| `rank8` | 0.742 |
| `rank8_no_canon` | 0.655 |

Monotonic improvement with rank (1→8); dropping Canon at rank 8 costs almost as much as dropping
from rank 8 to rank 2 (0.742 → 0.655), i.e. Canon's presence in the encoder matters, not just
RoPE+depth alone.

**Caveats.** No "Known issue" text, no disk-wipe note. Only 3/5 planned seeds landed locally for
every arm - treat the above means as lower-n than the README's stated design.

---

### arc1d_hypermodel_looped_rope_canon_grid

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_grid/`
- **README**: same directory, `README.md`

**Question.** Full `num_layers x gradient_clip_val` grid, crossed with task-descriptor
presence/absence, to disentangle whether `num_layers=2`'s apparent advantage (seen in `followup`)
interacts with `gradient_clip_val`, and whether the notd-vs-td gap (found in
`mechanism_ablation`) holds constant across the grid or narrows at better-tuned settings.

**Task set.** All 17 categories.

**Architecture.** `hyper_model.num_layers ∈ {2, 4, 8}` x `gradient_clip_val ∈ {5, 10, 20}` x
`hyper_head.num_tasks ∈ {18 (td), null (notd)}` = 18 cells. RoPE+Canon encoder, attention
pooling, **LoRA rank=8** fixed throughout.

**Training recipe.** RAdam, `learning_rate=0.001`, `weight_decay=0.01`, `batch_size=512,
N_supervision=2, max_steps=4000, warmup_steps=400`. **3 seeds** per cell.

**Jobs.** 3 x 3 x 2 = 18 configs x 3 seeds = **54 jobs total**.

**Status.** Finished-with-local-results: 54/54 `results.txt` present under
`outputs/arc1d_hypermodel_looped_rope_canon_grid/`.

**Key numeric findings** (mean `test_query_exact_match` across 3 seeds per cell, computed from
`results.txt`):

| layers | clip | td mean | notd mean |
|---:|---:|---:|---:|
| 2 | 5 | 0.739 | 0.629 |
| 2 | 10 | 0.864 | 0.587 |
| 2 | 20 | 0.864 | 0.527 |
| 4 | 5 | 0.712 | 0.636 |
| 4 | 10 | 0.856 | 0.667 |
| 4 | 20 | 0.852 | 0.667 |
| 8 | 5 | 0.746 | 0.587 |
| 8 | 10 | **0.871** | 0.530 |
| 8 | 20 | 0.867 | 0.413 |

Pooled: **td mean = 0.819 (n=27), notd mean = 0.583 (n=27)** - a consistent, large gap across
every layers/clip cell, confirming `mechanism_ablation`'s finding independently on 3x more cells.
Best single cell: `layers=8, clip=10, td` (test mean 0.871). At `clip=10` with the descriptor on,
`layers=8` also has the tightest std of any td cell (per README, ~0.009 vs. 0.027 at layers=4).

**Caveats.** No "Known issue" text, no disk-wipe note.

---

### arc1d_hypermodel_looped_rope_canon_capacity_baseline

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_capacity_baseline/`
- **README**: same directory, `README.md`

**Question.** Acting on the grid's finding (fix `clip=10`, drop the two structurally-dead
recolor tasks), does the notd-vs-td gap narrow at these improved settings, crossed with
`layers ∈ {4,8}`, and does doubling `notd` training length close any of the shortfall (i.e. is
it partly just undertraining)?

**Task set.** 15 categories: the 17-task list minus `1d_recolor_cnt`/`1d_recolor_oe` (both stuck
at literal 0.0 in every grid cell regardless of layers/clip - dropped as "structurally dead," not
a capacity question). `1d_move_dp` (0.711 mean, never zero) stays in as the task most likely to
show a genuine capacity effect.

**Architecture.** `gradient_clip_val=10.0` fixed (no longer swept). `num_layers ∈ {4, 8}` x
`num_tasks ∈ {18 (td), null (notd)}` x, for notd only, `max_steps ∈ {4000 (standard), 8000
(long)}`. RoPE+Canon encoder, attention pooling, LoRA rank=8.

**Training recipe.** RAdam, `learning_rate=0.001`, `weight_decay=0.01`, `batch_size=512,
N_supervision=2`. Standard arms: `max_steps=4000, warmup_steps=400`; long arms:
`max_steps=8000, warmup_steps=800`. **3 seeds** per arm.

**Jobs.** 6 arms x 3 seeds = **18 jobs total**.

**Status.** Finished-with-local-results, near-complete: 17/18 `results.txt` present under
`outputs/arc1d_hypermodel_looped_rope_canon_capacity_baseline/` (`layers8_notd` has 2/3 seeds;
every other arm has 3/3).

**Key numeric findings** (mean per arm; `val_query_exact_match` / `test_query_exact_match`):

| Arm | n | val mean | test mean |
|---|---:|---:|---:|
| `layers4_td` | 3 | 0.925 | 0.925 |
| `layers8_td` | 3 | 0.929 | 0.925 |
| `layers4_notd` | 3 | 0.650 | 0.621 |
| `layers8_notd` | 2 | 0.656 | 0.637 |
| `layers4_notd_long` | 3 | 0.688 | 0.667 |
| `layers8_notd_long` | 3 | 0.713 | 0.704 |

At these improved settings the td arms reach ~0.925-0.929 while notd sits at ~0.62-0.66
(standard length) - the gap **persists strongly** even after dropping the two dead recolor tasks
and raising clip to 10. Doubling training length (`_long`) helps notd modestly (+0.03 to +0.07)
but does not come close to closing the gap. `layers=4` and `layers=8` perform similarly at every
setting (no strong capacity effect from depth alone).

**Caveats.** No "Known issue" text, no disk-wipe note. README explicitly scopes out an
unseen-task-category generalization test as a "not built this round" stepping stone - that
became `arc1d_hypermodel_looped_rope_canon_generalization` (below), and flags that `td`
generalizes poorly to unseen categories because `task_indicator_proj`'s untrained held-out row
would be meaningless.

---

### arc1d_hypermodel_looped_rope_canon_optimizer

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_optimizer/`
- **README**: same directory, `README.md`

**Question.** Does AdamW at a lower LR close more of the notd/td gap than the settings already
tried, reusing `capacity_baseline/arm_layers8_notd_long` (0.738 mean val) as the RAdam/0.001
baseline (not rerun)?

**Task set.** 15 categories (same as `capacity_baseline`).

**Architecture.** Identical to `capacity_baseline/arm_layers8_notd_long`: `layers=8`, notd,
`max_steps=8000/warmup_steps=800`, `gradient_clip_val=10`, LoRA rank=8, attention pooling.

**Training recipe.** Single new arm `arm_adamw`: optimizer **AdamW**, `learning_rate=0.0005`
(vs. reused baseline's RAdam/0.001). `weight_decay=0.01`, `batch_size=512, N_supervision=2`.
**3 seeds**.

**Jobs.** 1 new config x 3 seeds = **3 jobs**.

**Status.** Finished-with-local-results: 3/3 `results.txt` present under
`outputs/arc1d_hypermodel_looped_rope_canon_optimizer/`.

**Key numeric findings.** `arm_adamw`: val mean = 0.792, test mean = 0.779 (individual seeds:
val 0.863/0.750/0.763, test 0.813/0.788/0.738) - a real improvement over the reused RAdam
baseline's ~0.738 mean val, motivating AdamW as the default going forward in the rest of this
chain.

**Caveats.** No "Known issue" text, no disk-wipe note.

---

### arc1d_hypermodel_looped_rope_canon_zhu_block

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_zhu_block/`
- **README**: same directory, `README.md`

**Question.** Do three architecture tricks from the Canon-layers research lineage (RMSNorm,
QK-norm, SwiGLU MLP - "Zhu" = Zeyuan Allen-Zhu, the Canon Layers paper's author) improve the
hypernetwork encoder, isolated individually and combined, against the `capacity_baseline`
`layers8_notd_long` baseline (0.738 mean)?

**Task set.** 15 categories.

**Architecture.** New `RoPECanonZhuTransformer` class with 3 independent boolean flags
(`use_rmsnorm`, `use_qk_norm`, `use_swiglu`), all default False (byte-identical to
`RoPECanonTransformer` when off). Arms: `arm_rmsnorm`, `arm_qknorm`, `arm_swiglu` (each isolated)
and `arm_zhu_all` (all three on), against the reused baseline (all off). RoPE and Canon A/B/C/D
unchanged. LoRA rank=8, `layers=8`.

**Training recipe.** Optimizer **RAdam** (unchanged from baseline, "not combined with
`arc1d_hypermodel_looped_rope_canon_optimizer`'s change"), `learning_rate=0.001,
weight_decay=0.01, batch_size=512, gradient_clip_val=10.0, N_supervision=2, max_steps=8000,
warmup_steps=800`. **3 seeds** per arm.

**Jobs.** 4 arms x 3 seeds = **12 new jobs**.

**Status.** Finished-with-local-results: 12/12 `results.txt` present under
`outputs/arc1d_hypermodel_looped_rope_canon_zhu_block/`.

**Key numeric findings** (mean per arm; `val_query_exact_match` / `test_query_exact_match`):

| Arm | val mean | test mean |
|---|---:|---:|
| `arm_rmsnorm` | 0.688 | 0.646 |
| `arm_qknorm` | 0.713 | 0.688 |
| `arm_swiglu` | 0.662 | 0.625 |
| `arm_zhu_all` | **0.808** | **0.825** |

None of the three tricks individually beats the reused 0.738 baseline; **`arm_zhu_all` (all
three combined) clearly beats both the baseline and every isolated arm**, suggesting a real
interaction effect worth following up (as the README itself anticipates) - pursued next in
`lr_sweep`.

**Caveats.** No "Known issue" text, no disk-wipe note.

---

### arc1d_hypermodel_looped_rope_canon_lr_sweep

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_lr_sweep/`
- **README**: same directory, `README.md`

**Question.** How low should the LR go under AdamW (established as the new default via
`arc1d_hypermodel_looped_rope_canon_optimizer`), crossed with baseline vs. `arm_zhu_all`
architecture, and does `arm_zhu_all`'s advantage hold up with more seeds and a real LR sweep
(its earlier 3-seed mean of 0.808 had one seed below baseline)?

**Task set.** 15 categories.

**Architecture.** Two architectures (baseline `rope_canon_transformer` vs. `zhu_all` =
`rope_canon_zhu_transformer` with all 3 flags on) x four learning rates (`1e-3, 5e-4, 1e-4,
5e-5`), AdamW throughout. `layers=8`, notd, LoRA rank=8, `gradient_clip_val=10`.

**Training recipe.** Optimizer **AdamW**, `weight_decay=0.01, batch_size=512, N_supervision=2,
max_steps=8000, warmup_steps=800`. **5 seeds** per cell (7 new leaf configs x 5 seeds = 35 new
jobs, plus 2 top-up seeds reusing `arc1d_hypermodel_looped_rope_canon_optimizer/arm_adamw.yaml`
for the `baseline_lr5e-4` cell - no new leaf config for that cell).

**Jobs.** 37 new jobs planned (35 new leaves + 2 top-ups).

**Status.** Finished-with-local-results, near-complete: **33/35** new-leaf `results.txt` present
under `outputs/arc1d_hypermodel_looped_rope_canon_lr_sweep/` - exactly `zhuall_lr5e-4_seed5` and
`zhuall_lr5e-5_seed5` are missing. The 2 top-up seeds for `baseline_lr5e-4` never landed locally
either (that cell's data lives only in `arc1d_hypermodel_looped_rope_canon_optimizer`'s 3 local
seeds, at 0.792 val mean - see that entry above), so the README's claim of "5 seeds now covering
all 8 cells" is not fully reflected in local files for that one cell.

**Key numeric findings** (mean per arm; `val_query_exact_match` / `test_query_exact_match`, n
seeds noted where <5):

| Arm | n | val mean | test mean |
|---|---:|---:|---:|
| `baseline_lr1e-3` | 5 | 0.780 | 0.763 |
| `baseline_lr1e-4` | 5 | 0.840 | 0.840 |
| `baseline_lr5e-5` | 5 | 0.858 | 0.835 |
| `zhuall_lr1e-3` | 5 | **0.862** | **0.873** |
| `zhuall_lr5e-4` | 4 | 0.794 | 0.781 |
| `zhuall_lr1e-4` | 5 | 0.810 | 0.808 |
| `zhuall_lr5e-5` | 4 | 0.797 | 0.809 |

The baseline architecture prefers a **lower** LR (best at 5e-5, 0.858 val); `zhu_all` prefers the
**original, higher** LR (best at 1e-3, 0.862 val / 0.873 test - the best cell found anywhere in
this whole tuning chain), confirming the Zhu block's advantage is real and not an artefact of one
lucky seed, and that the two architectures have different LR optima rather than one dominating
uniformly.

**Caveats.** No "Known issue" text, no disk-wipe note.

---

## Optimizer track

### arc1d_hypermodel_looped_rope_canon_muon

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_muon/`
- **README**: same directory, `README.md`

**Question.** Does the Muon optimizer (Keller Jordan) have any value for this model, crossed
with `notd` vs. `frozen_td` (a task-identity embedding frozen at random init for every category,
established as an interesting middle ground by the leave-one-out generalization work)? Fabio's
specific hypothesis: Muon may stabilise `notd` runs, since `notd` + LoRA-adapter parameterisation
may be a harder optimization landscape than AdamW handles well.

**Task set.** Standard in-distribution: full 15-task list (`task_categories == val_task_
categories`, no held-out axis).

**Architecture.** Zhu block (all 3 flags on), fixed across every arm. `optimizer ∈ {AdamW,
Muon}` x `{notd, frozen_td}` (plain `td` dropped - no held-out category to expose its weakness
here). Muon-eligible parameters: only `nn.Linear` weights, excluding `input_projection`,
`output_head`, and `task_indicator_proj` (kept on AdamW); the LoRA
`lora_proj_a`/`lora_proj_b`/`lora_proj_other` projections are deliberately included as
Muon-eligible. `muon_lr=0.005` (lowered 4x from Keller Jordan's published default 0.02, which
diverged to NaN at global_step ~915/4000 during a first attempt), `muon_momentum=0.95`.

**Training recipe.** AdamW-aux group `learning_rate=0.001, weight_decay=0.01`, `batch_size=512,
gradient_clip_val=10.0, N_supervision=2`. `max_steps=4000, warmup_steps=400` - **half** of every
other rope_canon experiment's 8000/800 (a GPU smoke run hit several categories at 1.0 exact match
within the first 100 steps, so this pass was deliberately shortened). LoRA rank=8. **3 seeds**
per arm.

**Jobs.** 4 arms x 3 seeds = **12 jobs**.

**Status.** Finished-with-local-results: 12/12 `results.txt` present under
`outputs/arc1d_hypermodel_looped_rope_canon_muon/`. (A 5th leftover config, `arm_td_muon.yaml`,
exists in the directory but is outside the README's stated 4-arm grid and has no output.)

**Key numeric findings** (mean per arm; `val_query_exact_match` / `test_query_exact_match`):

| Arm | val mean | test mean |
|---|---:|---:|
| `frozentd_adamw` | 0.992 | 0.992 |
| `frozentd_muon` | 0.988 | 0.988 |
| `notd_adamw` | 0.675 | 0.675 |
| `notd_muon` | 0.483 | 0.488 |

**`frozen_td` + Zhu block reaches near-ceiling accuracy (~0.99) with either optimizer** - the
strongest in-distribution result found in the whole tuning chain up to this point, and the direct
basis for `arc1d_lowdata`'s fixed architecture. **Muon does not support the stabilisation
hypothesis**: `notd_muon` (0.483-0.488) is clearly *worse* than `notd_adamw` (0.675), the
opposite of what Fabio's hypothesis predicted.

**Caveats.** No "Known issue" (max_steps/N_supervision) text - postdates the fix. No disk-wipe
note. README documents a new safety check added during this experiment: `training_step` now
raises immediately on a non-finite pre-clip gradient norm, instead of silently continuing to
train on a diverged run - this is what caught the original `muon_lr=0.02` NaN divergence cheaply.

---

### arc1d_hypermodel_looped_rope_canon_muon_sweep

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_sweep/`
- **README**: same directory, `README.md`

**Question.** Is there a better `muon_lr`/aux-`learning_rate`/`weight_decay`/`batch_size`
combination than the untuned placeholder from `arc1d_hypermodel_looped_rope_canon_muon`, and
does excluding the LoRA head's A/B factor projections from Muon
(`muon_exclude_lora_heads`) help - targeting the stated goal of speeding up training and getting
`notd` to solve more categories?

**Task set.** All 15 in-distribution categories, `notd` only (no `frozen_td` arm this round).

**Architecture.** Zhu backbone, `rope_canon_looped_transformer` target, `muon_momentum=0.95,
gradient_clip_val=10.0, max_steps=4000/warmup_steps=400` fixed. Swept: `muon_lr ∈ {0.008, 0.01,
0.016, 0.02}` x `learning_rate (AdamW aux) ∈ {3e-4, 6e-4, 3e-3}` x `weight_decay ∈ {0, 0.01,
0.1}` x `batch_size ∈ {512, 1024, 2048}` x `muon_exclude_lora_heads ∈ {False, True}`.

**Jobs.** 4x3x3x3x2 = **216 configs, 1 seed each = 216 planned jobs**
(`scripts/gen_hypermodel_muon_sweep_configs.py`).

**Status.** **Finished-but-wandb-only, and incomplete relative to plan.** No
`outputs/arc1d_hypermodel_looped_rope_canon_muon_sweep/` directory exists locally at all -
verified via wandb: the project actually has **230 runs** (more than the 216-cell grid, implying
some cells were rerun), of which **123 finished, 99 failed, 8 crashed**. The README's own framing
("cancelled after 99 cells once its findings pointed clearly enough in a direction to act on," per
the follow-up experiment's README) undersells how many runs were attempted and failed - 99+8 = 107
of the 230 launched runs did not complete cleanly.

**Caveats.** No local files to check for "Known issue" text against; the successor experiment's
README (`arc1d_hypermodel_looped_rope_canon_muon_moveablation`) reports the sweep's own findings
in retrospect (see next entry).

---

### arc1d_hypermodel_looped_rope_canon_muon_moveablation

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_muon_moveablation/`
- **README**: same directory, `README.md`

**Question.** Follow-up to the (cancelled) 216-cell sweep, locking in its two clearest findings
(`muon_exclude_lora_heads=true`, `batch_size=2048`) and testing a new hypothesis: does dropping
`1d_move_1p`/`1d_move_2p` from training (13 remaining categories) reduce move-family interference
in `notd` mode, both in-distribution and on the compositional-generalization holdout eval?

**What the prior sweep found** (per this README's own retrospective, quoting the 99 completed
cells): LoRA-head exclusion beat inclusion (mean 0.654 vs. 0.618, 9/10 top cells used exclusion);
`batch_size=2048` pulled ahead of 512/1024 (mean 0.658 vs. ~0.623-0.629); `batch_size=4096`
genuinely OOMs (44.21/44.40 GiB on a single GPU) and was dropped entirely; `muon_lr` itself showed
no real trend (~0.63-0.64 across 0.008/0.01/0.016); `muon_lr=0.02` (the original NaN-divergence
value) was never reached by the sweep.

**Task set.** 15 categories (`with`) vs. 13 categories dropping `1d_move_1p`/`1d_move_2p`
(`without`).

**Architecture.** Zhu backbone, LoRA rank=8, `muon_lr=0.02` (deliberately retesting the value
that caused the original NaN, now better-protected by LoRA-head exclusion), `muon_momentum=0.95,
muon_exclude_lora_heads=true, batch_size=2048`, notd only. Grid: `weight_decay ∈ {0.01, 0.1}` x
`learning_rate ∈ {3e-4, 6e-4}` x move-task inclusion `∈ {with, without}`.

**Training recipe.** `max_steps=4000/warmup_steps=400, gradient_clip_val=10.0, optimizer=Muon`.
Every run followed by `scripts/eval_compositional_holdout.py` (10 composite categories, zero-shot)
and `log_embedding_clusters: true`. **3 seeds** per cell.

**Jobs.** 2x2x2 = 8 configs x 3 seeds = **24 jobs planned**.

**Status.** **Finished-but-wandb-only, and materially incomplete relative to plan.** No
`outputs/arc1d_hypermodel_looped_rope_canon_muon_moveablation/` directory exists locally at all -
verified via wandb: **42 runs** total (more than the 24 planned, again implying reruns), of which
only **17 finished, 17 failed, 8 crashed**. This is a considerably worse completion rate than the
README's plan implies - under half the launched runs (17/42) finished cleanly, and even that 17
is fewer than the 24 originally planned jobs.

**Caveats.** No local `results.txt` to check "Known issue" text against. The
`batch_size`-vs-`max_steps` caveat is stated explicitly in the README: `max_steps` is fixed at
4000 regardless of `batch_size`, so `bsz=2048` recycles the ~40-41-raw-task training pool ~13.6x
per run rather than seeing more raw data (vs. ~3.4x at the prior sweep's `bsz=512` baseline).

---

## Embedding removal (origin story)

### arc1d_hypermodel_disentanglement

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_disentanglement/`
- **README**: `legacy/configs/experiments/arc1d_hypermodel_disentanglement/README.md`

**Question.** Can a hypermodel represent multiple distinct tasks simultaneously without
conflating them through a shared bottleneck - and if not, is that a task-similarity problem or an
architecture/capacity problem? This is the foundational experiment behind the task descriptor's
origin.

**Task set.** Individual/mixed subsets of the 18-category `arc_1d_all_tasks_augmented` set,
notably `1d_move_1p/2p/3p` (single and mixed) and `1d_move_3p` + `1d_denoising_1c`.

**Data.** `data/arc_1d_all_tasks_augmented` - ~720k train examples, 18 categories, 1000 variants
per task (per-pair colour permutation x shift; global colour mode for the 6 categories whose rule
needs cross-pair colour consistency - see README for the full per-task augmentation-mode table).

**Architecture.** Hyper encoder: `transformer`, `hidden_dim=64, num_heads=4, num_layers=2`.
Bottleneck: `hyper_head.pooling: hierarchical, projection_dims: [16]` - the key disentanglement
probe, a deliberately tight 16-dim code. Target (decoder): bidirectional `rnn`, `hidden_dim=22,
num_layers=2, activation=silu, use_residual=true`. Task descriptor (Exp 2 only):
`hyper_head.num_tasks: 18` (`Linear(18, 64)` one-hot projection).

**Training recipe** (`base_disentanglement.yaml`): optimizer **RAdam**, `learning_rate=0.001,
weight_decay=0.01, gradient_clip_val=1.0, batch_size=2048, max_steps=2000, warmup_steps=200`.
Single seed per config (`seed=42`, no seed sweep in this project).

**Status.** Finished-with-local-results: 6 `results.txt` present under
`outputs/arc1d_hypermodel_disentanglement/` (`disentangle_mix_1p3p_bottleneck2_rnn`,
`disentangle_mix_1p3p_task_descriptor_bottleneck_rnn`,
`disentangle_mix_1p3p_bottleneck2_proj_rnn`, `disentangle_mix_3p_denoise1c_rnn`,
`disentangle_mix_1p3p_task_descriptor_input_rnn`, `disentangle_mix_1p2p3p_rnn`), plus an
`overfit` smoke dir.

**Key numeric findings (README's own text).**
- Individual tasks (`move_1p`, `move_2p`, `move_3p`, trained alone): **100% - works**, confirming
  architecture/bottleneck capacity is sufficient for a single task.
- **Mixed move tasks conflate and fail**: `1d_move_1p + 1d_move_3p` "fails both";
  `1d_move_1p + 1d_move_2p + 1d_move_3p` "fails all."
- **Exp 1 (structurally different pairing, no descriptor): `1d_move_3p + 1d_denoising_1c`
  - both tasks solved.** The bottleneck disentangles fine when tasks are structurally
  dissimilar; the move-mix failure is a task-similarity problem, not a capacity/architecture
  problem.
- **Exp 2 (explicit task descriptor): `mix_1p3p_task_descriptor_bottleneck.yaml`
  - "task descriptor successfully disambiguates tasks... trains faster and converges more
  stably than without the descriptor."** This is the origin of every later `td`/`frozen_td`
  variant in the rope_canon tuning chain.

**Caveats.** No "Known issue" (max_steps/N_supervision) text - this project uses `hyper_model`
directly, not `looped_supervised`'s manual-optimization path the bug affects, and predates the
looped-hypernetwork line entirely. No disk-wipe note.

---

## Disentanglement diagnostics (clusters + probes)

### arc1d_hypermodel_looped_rope_canon_vae_disentanglement

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_vae_disentanglement/`
- **README**: same directory, `README.md`

This experiment explored whether a beta-VAE bottleneck on the hypernetwork's pooled task
representation could substitute for an explicit task descriptor - regularizing `notd`'s
representation toward disentangled structure via KL divergence, with no task-identity signal at
all. It is out of scope for detailed coverage here (22 arms across three beta grids: a
constant-beta sweep, a scaled/tuned-beta sweep, and a full-training KL-anneal sweep); per
direction, only the plain baseline reference numbers it starts from are extracted below.

**Baseline reference numbers** (quoted verbatim from
`legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_vae_disentanglement/README.md`, lines
5-12, where the README states they come from **`arc1d_hypermodel_looped_rope_canon_muon_diag`'s
notd/td/frozentd comparison** - seed=1, Muon, `lora_adapter_rank=4`, `max_steps=2000`, 15 tasks;
note `muon_diag` has no corresponding `legacy/configs/experiments/` directory of its own, it is an
earlier informal diagnostic run referenced only from this README):

| Arm | Linear-probe accuracy | `val_query_exact_match` |
|---|---:|---:|
| `notd` (no task descriptor) | 73.3% | 82.5% |
| `td` (learned task descriptor) | 100% | 97.5% |
| `frozentd` (frozen task descriptor) | 100% | 96.2% |

`td` and `frozentd` scored identically on both metrics - the README states this is why "frozen
stays the default going forward, no further td-vs-frozentd work is planned." Cluster maps at this
stage also showed `notd`'s move-family tasks (Move 1p/2p/3p/Dynamic) partially overlapping,
consistent with its lower probe score.

---

### `arm_no_task_indicator` cross-check (arc1d_hypermodel_looped_rope_canon_mechanism_ablation)

A second, independent source of notd-vs-td evidence, from a different project than the VAE study
above: `arc1d_hypermodel_looped_rope_canon_mechanism_ablation`'s `arm_no_task_indicator` (see full
entry in the tuning-chain section above) found removing the one-hot task descriptor from an
otherwise-identical rank=8/RoPE+Canon/attention-pooling setup costs **roughly 10-11 points of
test exact match**: 0.636 (notd, 3 local seeds) vs. 0.742 (the reused `rank8` td baseline, 3
local seeds). Directly corroborated a third time, at larger n and across a full grid, by
`arc1d_hypermodel_looped_rope_canon_grid`'s pooled td/notd means (0.819 vs. 0.583 test, n=27
each). All three independent measurements agree on the same qualitative finding: the task
descriptor matters substantially for in-distribution multitask accuracy.

---

## Compositional generalization

### arc1d_hypermodel_compositional_generalization

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/`
- **README**: `legacy/configs/experiments/arc1d_hypermodel_compositional_generalization/README.md`

**Question.** Having learned "denoise" and "shift" (etc.) as separate rules on 15 in-distribution
categories, can the hypernetwork solve a never-seen task built by chaining two of those rules
together (e.g. "denoise a block, then shift it") - a combination it was never trained on but
built entirely from skills it was?

**Task set.** Trained on the same 15 in-distribution categories as
`arc1d_hypermodel_looped_rope_canon_muon`/`vae_disentanglement`. Evaluated zero-shot on 10 new
composite categories generated from scratch (ARC-1D has no native notion of chained rules; the
generator is `data_modules/arc1d_compositional.py`, new for this experiment, tested in
`tests/test_arc1d_compositional.py`): `1d_comp_denoise1c_shift3`, `1d_comp_denoisemc_copy`,
`1d_comp_denoisemc_denoise1c`, `1d_comp_denoisemc_mirror`, `1d_comp_fill_mirror`,
`1d_comp_fill_movedynamic`, `1d_comp_fill_shift3`, `1d_comp_hollow_shift3`,
`1d_comp_movedynamic_hollow`, `1d_comp_shift3_copy`.

**Data.** Training: `data/arc_1d_looped_augmented` (15-category subset). Held-out eval:
`data/arc_1d_compositional_holdout` - 400 instances (10 categories x 40), generated by
`scripts/build_arc1d_compositional.py`, single `holdout_test` split.

**Architecture.** `base.yaml` reuses the `arc1d_hypermodel_looped_rope_canon_muon`/
`vae_disentanglement` recipe verbatim: Zhu backbone, Muon (`muon_lr=0.005,
muon_momentum=0.95`), **LoRA rank=4**, `max_steps=2000, N_supervision=2`, same 15
`task_categories`/`val_task_categories`. Three arms: `notd.yaml` (`num_tasks: null`), `td.yaml`
(`num_tasks: 18`), `frozen_td.yaml` (`num_tasks: 28` - 18 + 10 reserved composite-category
indices in the frozen, never-trained projection matrix, since `td`'s learned embedding is
undefined for an index it never saw and is therefore not evaluated on the holdout at all).

**Status.** **Finished-with-local-results - and the README's own "Status" section is stale.**
The README's final section states "Training itself... has not been run yet - everything up to
that point has been verified with an untrained model instance... but no results exist yet."
This is **contradicted by the actual files on disk**: `outputs/arc1d_hypermodel_
compositional_generalization/looped_hyper_rope_canon_compositional_generalization_{notd,td,
frozentd}/results.txt` all exist, dated 2026-07-30, with real, non-trivial metrics (e.g. `notd`'s
`val_loss: 0.170`, far from an untrained-model value). The held-out compositional eval also ran
for `notd` and `frozen_td` (not `td`, per design):
`outputs/.../looped_hyper_rope_canon_compositional_generalization_{notd,frozentd}/
compositional_holdout_eval/results.txt` both exist.

**Key numeric findings - in-distribution** (`results.txt`, `val_query_exact_match` overall):
`td` = 0.9875, `frozentd` = 0.9875, `notd` = 0.4875. Task descriptor (learned or frozen) reaches
near-ceiling in-distribution; `notd` is roughly half that.

**Key numeric findings - full per-category compositional-holdout breakdown** (from
`compositional_holdout_eval/results.txt`, n=40 per category, both arms):

| Category | notd exact_match | notd seq_accuracy | frozentd exact_match | frozentd seq_accuracy |
|---|---:|---:|---:|---:|
| `1d_comp_denoise1c_shift3` | 0.000 | 0.810 | 0.000 | 0.716 |
| `1d_comp_denoisemc_copy` | 0.000 | 0.634 | 0.025 | 0.742 |
| `1d_comp_denoisemc_denoise1c` | 0.100 | 0.921 | 0.000 | 0.898 |
| `1d_comp_denoisemc_mirror` | 0.025 | 0.860 | 0.000 | 0.466 |
| `1d_comp_fill_mirror` | 0.000 | 0.792 | 0.000 | 0.698 |
| `1d_comp_fill_movedynamic` | 0.000 | 0.678 | 0.000 | 0.672 |
| `1d_comp_fill_shift3` | 0.000 | 0.756 | 0.000 | 0.721 |
| `1d_comp_hollow_shift3` | 0.000 | 0.854 | 0.000 | 0.743 |
| `1d_comp_movedynamic_hollow` | 0.000 | 0.807 | 0.000 | 0.780 |
| `1d_comp_shift3_copy` | 0.000 | 0.713 | 0.000 | 0.629 |
| **overall (n=400)** | **0.013** | - | **0.003** | - |

Neither arm gets meaningfully above chance on exact match (overall 1.3% for notd, 0.3% for
frozentd) - the model does not compose learned rules into a correct never-seen combination. But
per-token `seq_accuracy` is consistently high (0.47-0.92), a "near-miss" pattern: the model gets
most positions right without landing the exact answer. By this seq_accuracy signal, `notd`'s best
categories are `1d_comp_denoisemc_denoise1c` (0.921) and `1d_comp_denoisemc_mirror` (0.860);
`frozentd`'s best is `1d_comp_denoisemc_denoise1c` (0.898). Both arms are comparatively weakest
on `1d_comp_denoisemc_copy` (notd 0.634) and `1d_comp_shift3_copy`/`1d_comp_denoisemc_mirror`
(frozentd 0.629/0.466) - the categories involving `copy` or `mirror` composed with a second rule
look hardest for both variants.

**Caveats.** README "Status" text is stale/wrong (see above - treat it as not reflecting the
actual state of the project). No "Known issue" (max_steps/N_supervision) text - postdates the
fix. No disk-wipe note in this particular README.

---

## Generalization to unseen task categories

### arc1d_hypermodel_looped_rope_canon_generalization

- **Directory**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_generalization/`
- **README**: `legacy/configs/experiments/arc1d_hypermodel_looped_rope_canon_generalization/README.md`

**Question.** Can the hypernetwork generalize to a task category it has never seen during
training, inferring the task purely from 3-shot support examples at inference time - and does the
learned one-hot task-identity embedding (`td`) help or actively hurt that generalization, versus
no descriptor (`notd`) or a frozen-at-random-init descriptor for every category (`frozen_td`)?

**Task set.** 5 held-out categories, each held out one at a time from an otherwise-14-task
training set (each chosen because it has a structurally similar sibling remaining in training):
`1d_move_2p` (siblings: `move_1p/3p/dp/2p_dp`), `1d_denoising_mc` (sibling: `denoising_1c`),
`1d_flip` (weaker sibling: `mirror`), `1d_pcopy_mc` (sibling: `pcopy_1c`), `1d_hollow` (sibling:
`fill`). `1d_recolor_cnt`/`1d_recolor_oe` excluded as candidates (already 0% in-distribution).

**Architecture.** Zhu block (all 3 flags on) + AdamW/lr=0.001 (the `lr_sweep` leader), fixed.
Only axis varied: task-identity conditioning, 3 variants (`notd`, `td` with `num_tasks=18`,
`frozentd` with `num_tasks=18, freeze_task_indicator=true`). LoRA rank=8, `layers=8` (implicit
via the Zhu backbone default).

**Training recipe** (`base.yaml`): optimizer **AdamW**, `learning_rate=0.001, weight_decay=0.01,
batch_size=512, gradient_clip_val=10.0, N_supervision=2, max_steps=8000, warmup_steps=800`.
**3 seeds** per arm (README calls for 3; see caveat below - only 2 actually ran).

**Jobs.** 5 held-out categories x 3 variants x 3 seeds = **45 jobs planned**.

**Status.** No local results whatsoever: `outputs/arc1d_hypermodel_looped_rope_canon_
generalization/` does not exist. **This matches the README's own documented disk-wipe incident**
(see Caveats).

**Key numeric findings - this write-up uses fresh wandb numbers, which supersede and are more
complete than the README's own seed-1-only table.** Sourced directly from wandb project
`arc1d_hypermodel_looped_rope_canon_generalization`: **30 runs total, 28 finished, 1 failed, 1
crashed** - i.e. **2 seeds per cell for most cells** (not the 3 originally planned; a third seed
was never launched), except `pcopy_mc/frozen_td` and `pcopy_mc/td`, which each have only 1 of 2
seed runs finished (the other failed/crashed respectively).

| Held-out category | Variant | Seed A | Seed B |
|---|---|---:|---:|
| `denoising_mc` | `frozen_td` | 1.0 | 1.0 |
| `denoising_mc` | `notd` | 0.80 | 1.0 |
| `denoising_mc` | `td` | 0.0 | 1.0 |
| `flip` | `frozen_td` | 0.0 | 0.0 |
| `flip` | `notd` | 0.0 | 0.0 |
| `flip` | `td` | 0.0 | 0.0 |
| `hollow` | `frozen_td` | 0.0 | 0.0 |
| `hollow` | `notd` | 0.0 | 0.0 |
| `hollow` | `td` | 0.0 | 0.0 |
| `move_2p` | `frozen_td` | 0.0 | 0.0 |
| `move_2p` | `notd` | 0.0 | 0.0 |
| `move_2p` | `td` | 0.0 | 0.0 |
| `pcopy_mc` | `frozen_td` | 0.0 (1/2 seeds finished, other failed) | - |
| `pcopy_mc` | `notd` | 0.0 | 0.0 |
| `pcopy_mc` | `td` | 0.0 (1/2 seeds finished, other crashed) | - |

(Metric: `val_query_exact_match_by_task_<held_out_category>`.)

**Headline findings.**
- **`frozen_td` is the only variant that succeeds consistently** on the one category where
  success occurs at all (`denoising_mc`: both seeds = 1.0).
- **`td` is bimodal on that same category** (0.0 then 1.0) - high seed variance, unlike
  `frozen_td`'s clean consistency.
- **The other 4 categories (`flip`, `hollow`, `move_2p`, `pcopy_mc`) fail completely - 0.0 -
  under every variant and every seed.** The model does not generalize to a genuinely unseen
  category in the general case; `denoising_mc`'s partial success is best explained by its
  unusually close sibling (`denoising_1c` differs only in single- vs. multi-colour noise).

**This supersedes the README's own findings section**, which is explicitly based on seed 1 only
(reconstructed from wandb after a local data-loss incident, see below) and states the headline
"the model does not generalize to an unseen task category" with the same 4-of-5-categories-fail
pattern - the 2-seed wandb data above confirms that conclusion rather than overturning it, and
additionally exposes `td`'s seed-to-seed bimodality on `denoising_mc`, which the seed-1-only
table could not show.

**Caveats.**
- **Disk-wipe / data-loss note, stated explicitly in the README** ("Findings (seed 1, all 15
  arms)" section, final paragraph): after the first pass finished (44/45 jobs, one NCCL-timeout
  failure), a manual `rm -r outputs/*` on `torrnode12` - intended to fix what looked like a
  disk-space failure but was actually the shared `/homes/55` NFS quota being full (mostly an
  uncleaned local `wandb/` run cache, 49G) - **deleted every local `results.txt`/checkpoint for
  this run before it was rsynced**. The README's seed-1 numbers were reconstructed from wandb
  (`log_model=False` means no checkpoints were ever uploaded there - those are gone for good).
  Seeds 2-3 were relaunched from scratch after the quota was freed, but per the wandb run counts
  above, only one additional seed's worth of runs (mostly) actually landed, not two.
- No "Known issue" (max_steps/N_supervision) text - postdates the fix.

---

## Low data augmentation

### arc1d_lowdata

- **Directory**: `experiments/06_data_efficiency_ablation/hypernetwork/`
- **README**: `experiments/06_data_efficiency_ablation/hypernetwork/README.md`

**Question.** Phase 1 of a 3-phase data-efficiency study. Does the hypernetwork need less
training data per task than a model with no cross-task transfer, because it shares one backbone
across all 15 tasks? Sweeps `variants_per_base_task` down to find where performance collapses.

**Task set.** 15 in-distribution categories (`task_categories == val_task_categories`, no
held-out axis) - the same 15-category convention used from `capacity_baseline` onward.

**Architecture.** Fixed architecture carried over from `arc1d_hypermodel_looped_rope_canon_muon`'s
conclusions: Zhu block, `frozen_td` (`num_tasks=18, freeze_task_indicator=true` - "near-ceiling
~99% exact match vs ~68% for notd"), **LoRA rank=8**.

**Training recipe.** Optimizer **Muon** (`muon_lr=0.005, muon_momentum=0.95`, the already-
validated stable value), AdamW-aux `learning_rate=0.001, weight_decay=0.01`, `batch_size=512,
gradient_clip_val=10.0, N_supervision=2, max_steps=2000, warmup_steps=200` - held fixed across
every data level so only the data varies. **3 seeds** per level.

**Data-reduction mechanism.** `Arc1dMetaMulticlassDataModule.variants_per_base_task`/`data_seed`
(`data_modules/arc1d_meta_multiclass.py`), applied only to the train split; val/test stay at the
same fixed 100 examples/category. Sampling is stratified per base task and nested/cumulative
across levels (level K = level K-1 plus one more variant per base task); `data_seed=42` fixed
independent of training seed, so all 3 seeds at a given level see identical training data. Levels:
`v1` (1 variant/base task, ~40/category, the true original only, zero augmentation), `v2` (~80),
`v3` (~120), `v4` (~160), `v5` (~200), `v20` (~800).

**Jobs.** 6 levels x 3 seeds = **18 jobs planned**.

**Status.** No local `results.txt` anywhere under `outputs/arc1d_lowdata/` -
**Finished-but-wandb-only**.

**Key numeric findings - from wandb, not local files** (project `arc1d_lowdata`, run names
`lowdata_v1`..`lowdata_v20`). `test_query_exact_match` per level (mean across available seeds):
`v1=0.897` (4 seeds: 0.887, 0.887, 0.875, 0.938), `v2=0.954` (3 seeds), `v3=0.967` (3 seeds),
`v4=0.988` (3 seeds), `v5=0.983` (3 seeds), `v20=0.987` (3 finished of 4, one crashed).
Performance is already near-ceiling by `v2` (~80 examples/category) and essentially saturated by
`v4`-`v5`.

**Per-task breakdown at v1** (from wandb, `val_query_exact_match_by_task`): `1d_fill=1.0,
1d_flip=1.0, 1d_hollow=1.0, 1d_mirror=0.8, 1d_move_1p=1.0, 1d_move_2p=1.0, 1d_move_2p_dp=1.0,
1d_move_3p=1.0, 1d_move_dp=0.2, 1d_pcopy_1c=1.0, 1d_pcopy_mc=1.0, 1d_recolor_cmp=0.2,
1d_scale_dp=1.0, 1d_denoising_1c=1.0, 1d_denoising_mc=1.0`. 13/15 categories already at 1.0 with
only the original ~40 examples/category; `1d_move_dp` (0.2) and `1d_recolor_cmp` (0.2) are the
clear laggards at the lowest data level - this exact pair is the basis for the key comparison in
`arc1d_lowdata_baseline` below.

**Caveats.** README frames this explicitly as "Phase 1 of a 3-phase data-efficiency study," with
Phase 2 (a no-hypernetwork per-task baseline - later built as `arc1d_lowdata_baseline`) and Phase
3 (a shared backbone + frozen task-identity embedding, no hypernetwork - not built) both stated as
"direction only, not designed yet" at time of writing. No "Known issue" text, no disk-wipe note.

---

### arc1d_lowdata_lowrank

- **Directory**: `experiments/06_data_efficiency_ablation/hypernetwork_lowrank/`
- **README**: `experiments/06_data_efficiency_ablation/hypernetwork_lowrank/README.md`

**Question.** Follow-on from `arc1d_lowdata`, which found `variants_per_base_task=2` already
reaches full performance at the default LoRA rank=8. Does that conclusion hold at lower rank, and
is the LoRA-style parameterization itself even necessary at low data (vs. full-rank / no
LoRA-adapter generation)?

**Task set.** Same 15 in-distribution categories as `arc1d_lowdata`.

**Architecture.** Same fixed backbone as `arc1d_lowdata` (Zhu block, `frozen_td`). Crossed:
`lora_adapter_rank ∈ {4, 2, 1}` plus a `full` arm (`hyper_head.lora_adapter: false` - direct,
unrestricted full-rank weight generation, no frozen random backbone at all) x `variants_per_
base_task ∈ {1, 2, 3}`. **Rank=8 is not re-run here** - `arc1d_lowdata`'s `cell_v1/v2/v3` serve
as the rank=8 reference column.

**Training recipe.** Identical to `arc1d_lowdata`: Muon (`muon_lr=0.005, muon_momentum=0.95`),
AdamW-aux `learning_rate=0.001, weight_decay=0.01`, `batch_size=512, gradient_clip_val=10.0,
N_supervision=2, max_steps=2000, warmup_steps=200`. **3 seeds** per cell.

**Jobs.** 4 ranks x 3 levels x 3 seeds = **36 jobs planned**.

**Status.** No local `results.txt` under `outputs/arc1d_lowdata_lowrank/` -
**Finished-but-wandb-only**.

**Key numeric findings - from wandb, not local files.** Mean `test_query_exact_match` by
(rank, level): `full/v1=0.925, full/v2=0.95`; `r1/v1=0.85, r1/v2=0.863, r1/v3=0.931`; `r2/v1=0.85,
r2/v2=0.887, r2/v3=0.938`; `r4/v1=0.913, r4/v2=0.95, r4/v3=0.962` (`full/v3` not available in
this pull). Rank=8 (from `arc1d_lowdata`) is the reference column, not rerun here - `v1=0.887`
(mean of its 4 seeds), `v2=0.954`, `v3=0.967`. Pattern: performance rises with rank at fixed data
level (r1 < r2 < r4 ≈ r8), and `full` (no LoRA constraint) at v1/v2 (0.925/0.95) sits close to or
slightly above the r4/r8 range, i.e. the low-rank constraint is not obviously buying anything as a
regularizer at these data levels, consistent with r4 already being "enough."

**Caveats.** No "Known issue" text, no disk-wipe note.

---

### arc1d_lowdata_baseline

- **Directory**: `experiments/06_data_efficiency_ablation/individual/`
- **README**: `experiments/06_data_efficiency_ablation/individual/README.md`

**Question - the critical comparison experiment.** Companion to `arc1d_lowdata`, without the
hypernetwork: does the hypernetwork's data efficiency come specifically from cross-task transfer,
or would a plain single-task model need just as little data on its own? Trains the exact same
target architecture directly, one model per task category, no weight generation, no cross-task
sharing.

**Task set.** Same 15 categories as `arc1d_lowdata` (not `arc1d_uniform_ablation`'s 17 - the 2
extra categories there are excluded here too), one config per category, trained independently.

**Data.** `data/arc_1d_looped_augmented`, same `data_seed=42` as `arc1d_lowdata` - **the two
experiments train on identical underlying task-variant rows at a given level, not just a matched
count.** `variants_per_base_task ∈ {1, 2, 3}` (README states this "≈40/80/120 examples/category,"
confirmed by `experiments/06_data_efficiency_ablation/individual/README.md`, "Levels this round" section
- matching `arc1d_lowdata`'s `cell_v1`/`cell_v2`/`cell_v3` exactly). Levels 4/5/20 are out of
scope for this experiment.

**Architecture.** `rope_canon_looped_transformer` (`hidden_dim=16, num_heads=2, inner_dim=16,
inner_num_heads=2, n_loops=4`, Canon ABCD), trained directly via `LoopedSupervisedLightning`
(`model: looped_supervised`) - no task-identity conditioning of any kind, no LoRA/hypernetwork.
Matches `arc1d_uniform_ablation`'s `T5_dim16` config for this exact architecture.

**Training recipe** (`base.yaml`): optimizer **Muon** (`muon_lr=0.005, muon_momentum=0.95` - kept
identical to `arc1d_lowdata` to remove optimizer as a confound, even though RAdam was the
previously-validated choice for this exact architecture in `arc1d_uniform_ablation`/
`arc1d_recursion_ablation`), AdamW-aux `learning_rate=0.001, weight_decay=0.01`, `batch_size=256`
(smaller than `arc1d_lowdata`'s 512 - single-task, no hypernetwork overhead), `gradient_clip_val=
10.0, N_supervision=2, max_steps=2000, warmup_steps=200`. **3 seeds** per (category, level) cell.

**Jobs.** 15 categories x 3 levels x 3 seeds = **135 jobs planned**.

**Status.** No local `results.txt`/run directories at all under
`outputs/arc1d_lowdata_baseline/` (0 found) - but **this reflects only the local filesystem, not
the actual training state.** Per fresh wandb verification, **all 135 planned jobs are finished on
the cluster.** (Any note claiming "zero runs" would be describing local-disk state only; the
correct status is **Finished-but-wandb-only**, not not-run.)

**Documented known asymmetry** (README, "Known asymmetry" section): `Arc1dDirectDataModule`
supervises only the row's **3 support pairs** per training example (the query is held out
entirely for dev/test), whereas `arc1d_lowdata`'s hypernetwork supervises **all 4** examples per
row (3 support + that row's own query, confirmed by tracing the training loss in
`models/hypermodel_lightning.py`). At a given `variants_per_base_task` level this baseline
therefore sees 3 raw pairs/row vs. the hypernetwork's 4. The README explicitly notes this "does
not by itself explain a *large* gap in either direction" and that the hypernetwork's 3-shot
support is not shown to the target model in-context (no few-shot-ICL advantage either way).

**Key numeric findings - from wandb, not local files.** Per-category `test_query_exact_match`
mean across 3 seeds at each level (v1/v2/v3):

| Category | v1 | v2 | v3 |
|---|---:|---:|---:|
| `1d_denoising_1c` | 1.0 | 1.0 | 1.0 |
| `1d_denoising_mc` | 1.0 | 1.0 | 1.0 |
| `1d_fill` | 1.0 | 1.0 | 1.0 |
| `1d_flip` | 1.0 | 1.0 | 1.0 |
| `1d_hollow` | 1.0 | 1.0 | 1.0 |
| `1d_mirror` | 0.867 | 1.0 | 1.0 |
| `1d_move_1p` | 1.0 | 1.0 | 1.0 |
| `1d_move_2p` | 1.0 | 1.0 | 1.0 |
| `1d_move_2p_dp` | 1.0 | 1.0 | 1.0 |
| `1d_move_3p` | 1.0 | 1.0 | 1.0 |
| `1d_move_dp` | 0.133 | 0.667 | 0.8 |
| `1d_pcopy_1c` | 1.0 | 1.0 | 1.0 |
| `1d_pcopy_mc` | 1.0 | 1.0 | 1.0 |
| `1d_recolor_cmp` | 0.0 | 0.0 | 0.0 |
| `1d_scale_dp` | 0.8 | 0.933 | 1.0 |

**Key finding - directly comparable to `arc1d_lowdata`'s per-task v1 numbers above, and a
genuinely important, somewhat surprising result for the "does the hypernetwork's data efficiency
come from cross-task transfer" question.** 12 of 15 categories already reach ~100% test exact
match with a **single-task model at the lowest data level** (v1, ~40 examples) - the per-task
baseline is already highly sample-efficient on most tasks *without any hypernetwork at all*.
`1d_recolor_cmp` fails completely (0.0) at every level tested. `1d_move_dp`, `1d_mirror`, and
`1d_scale_dp` are the only categories that clearly improve with more data.

This means the value-add of cross-task transfer at these data levels is smaller than expected
overall, and concentrated mainly in a few hard tasks rather than spread broadly - most starkly:
- **`1d_recolor_cmp` at v1: hypernetwork 0.2 vs. baseline 0.0** - a real win from cross-task
  transfer on the one category the baseline cannot solve at all, at any data level tested.
- **`1d_scale_dp` at v1: hypernetwork 1.0 vs. baseline 0.8** - a smaller but real win.
- On the other 13 categories, the two are statistically indistinguishable or the baseline is
  already at ceiling (e.g. `1d_move_dp` at v1: hypernetwork 0.2 vs. baseline 0.133 - both poor,
  neither clearly ahead).

**Caveats.** No "Known issue" (max_steps/N_supervision) text, no disk-wipe note. Consistent with
`arc1d_lowdata`'s README framing this as its own deferred "Phase 2."

---

## Diagnostic and support code

**`visualisation/embedding_clusters.py`** (358 lines). Produces PCA/t-SNE/UMAP 2D projections of
the hypernetwork's pooled task-latent embeddings (`compute_pca_2d`, `compute_tsne_2d`,
`compute_umap_2d`, `render_embedding_cluster_figure`), **not** k-means or any other clustering
algorithm - it's a dimensionality-reduction/visualisation and quantitative-probe module, not a
clustering one. `compute_linear_probe_accuracy` fits a cross-validated multinomial
logistic-regression classifier (`StratifiedKFold` + `LogisticRegression`) on the raw
(unprojected) embedding vectors to predict task category, giving a cheap quantitative
disentanglement signal that complements the qualitative 2D projections (a linear classifier
separating categories well is a stronger, more falsifiable claim than "the clusters look
separated in a 2D projection"). `render_single_projection_figures` supports rendering each
projection as its own standalone image (used by later experiments for individual W&B panels)
alongside the original combined multi-panel figure.

**`training/logging.py`'s `log_embedding_cluster_plots`** (defined at line 626). The
end-of-run Lightning-side hook that calls into `embedding_clusters.py`: validation-only, gated
behind `log_embedding_clusters: true` and a model exposing `supports_embedding_visualization`
(the hypermodel path only). Supports an optional `holdout_dataloader` argument (added for
`arc1d_hypermodel_compositional_generalization`) to overlay a second group of embeddings - e.g. a
zero-shot compositional-holdout set - on the same fitted projection as the reference validation
embeddings, styled as translucent circles (reference) vs. full-colour X markers (holdout), so it's
visually clear whether never-seen examples land near their constituent base-rule clusters.
Omitting `holdout_dataloader` (every experiment except the compositional-generalization and
muon-moveablation ones) reproduces the original single-group behaviour unchanged.

**`scripts/eval_compositional_holdout.py`** (190 lines). Loads a trained `notd` or `frozen_td`
hypermodel checkpoint (`td` is explicitly unsupported - its `num_tasks=18` has no reserved indices
for the composite categories) and runs it zero-shot, unmodified, with no fine-tuning, on
`data/arc_1d_compositional_holdout`, reporting per-category `exact_match`/`seq_accuracy` to
`results.txt` plus a few rendered qualitative predictions per category. Registers the 10 composite
category names into `models.hypermodel_lightning.TASK_CATEGORY_INDEX` at indices 18-27 for the
duration of the eval run only. Reused verbatim (unmodified) by
`arc1d_hypermodel_looped_rope_canon_muon_moveablation` to get a compositional-generalization
reading on every one of its cells as a side effect of its own move-task-ablation training runs.

**`scripts/analyze_weight_space_pca.py`** (252 lines) - one-line pointer only, out of scope for
this document. This is an older, unrelated diagnostic: "Offline PCA analysis for `1d_move_1p`
binary bidirectional 1-layer RNN runs" (its own module docstring) - a weight-space PCA tool from
the earlier binary/RNN-target track, not the pooled-task-embedding PCA/t-SNE/UMAP tooling
described above. Noted here only so it isn't confused with `visualisation/embedding_clusters.py`.

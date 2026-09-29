# Experiment 7: task-identity ablation (frozen vs learned, latent vs input)

## Question

In experiments 2 to 5, the hypernetwork receives task identity (`frozen_td`) in one specific
way. The one-hot task id passes through a random `nn.Linear(18, 4)` that is **frozen**, and the
result is **added to the pooled task representation** (the latent) just before the weight
decoder. Two parts of this are untested assumptions:

1. **Frozen vs learned.** Does a learnable task embedding generalise better (experiments 4
   and 5)? Do its task representations cluster more cleanly by category?
2. **Latent vs input.** Should the embedding be in the latent, or added to every support-set
   token before the encoder, the way a positional embedding is?

The two choices are crossed (2x2), with `notd` as the reference:

| Arm | Placement | Task embedding | Runs |
|---|---|---|---|
| `notd` | none | none | experiment 2 / 5 |
| `frozentd_latent` | latent | frozen, random | experiment 2 (= its `frozen_td`); leave-one-out retrained here |
| `learnedtd_latent` | latent | learned | this experiment |
| `frozentd_input` | every support token | frozen, random | this experiment |
| `learnedtd_input` | every support token | learned | this experiment |

## Hypotheses

Written before any run. Each hypothesis is judged after the runs, against the s.d. across seeds.

| # | Comparison | Expected | Why | Measured by | Refuted if |
|---|---|---|---|---|---|
| H1 | learned latent vs frozen latent | **no difference** | In the latent, the projection only adds a per-task offset just before the decoder. A random 18→4 projection already gives each task a distinct offset, and the decoder can adapt to whatever offsets it is given, so learning them adds nothing | in-distribution, held-out and compositional accuracy; cluster metrics | the gap exceeds the seed s.d., in either direction |
| H2 | frozen input vs frozen latent | **better alignment in the latent space, possibly better generalisation** | At the input, the task ID passes through the encoder together with the support examples, so the encoder can learn to map examples of the same task to the same region. The representation is then organised by what the examples show, not by an offset added afterwards, and that should hold when the ID is zeroed or combined | clustering of the `support`, `support_no_id` and `weights` spaces; held-out (zero ID) and compositional (multi-hot) accuracy | clustering is no better, or held-out/compositional accuracy is lower |
| H3 | learned input vs frozen input | **better still** | The embedding can move towards what the encoder finds useful (for example, related tasks such as the move variants placed close together), and it has 72 more trainable parameters (see below) | the same metrics, plus the similarity of the learned embeddings | no gain beyond the seed s.d. |

**Alternative outcome to watch for.** With the ID at the input, the encoder could take a
shortcut: rely on the ID and ignore the support examples. It would then do well in
distribution but collapse when the ID is zeroed (leave-one-out). The drop from
in-distribution to held-out accuracy, reported per arm, separates this from H2 and H3.

### Parameter count

The arms are not all the same size, so every results table reports trainable and total
parameters. Total includes the frozen target, which is only a shape template.

| Arm | Task projection | Trainable | Total |
|---|---|---:|---:|
| `notd` | none | 10,156 | 11,360 |
| `frozentd_latent`, `frozentd_input` | 72, frozen | 10,156 | 11,432 |
| `learnedtd_latent`, `learnedtd_input` | 72, trained | 10,228 | 11,432 |

- Latent and input placement cost the same (the projection maps 18 tasks to 4 dimensions in
  both), so **H2 compares arms of equal size**.
- The learned arms have 72 more trainable parameters (+0.7%), so a small H3 gain could come
  from capacity rather than from learning the embedding. Under the zero-ID protocol, the
  held-out category never uses the learned table, so any held-out gain must come from how
  training shaped the encoder and decoder.
- If H3 holds by a small margin, a parameter-matched control follows: frozen input with 72
  trainable parameters added elsewhere. It is not part of the first runs.

## Setup

- **Model and training.**
  - In-distribution runs use experiment 2's dim-4 recipe, and leave-one-out runs use
    experiment 5's.
  - Every config inherits the matching experiment's `base.yaml` and changes only `hyper_head`.
  - Seeds match. With the same seed, all arms start from the same initial weights, including
    the same random task projection for frozen latent and frozen input
    (`tests/test_task_indicator_placement.py` checks this against experiments 2 and 5).
- **New config keys** (both off by default, so earlier experiments build the same models):
  - `hyper_head.placement: latent | input`.
  - `hyper_head.zero_task_categories`: categories scored with a zero task vector.
- **Learned arms** set `freeze_task_indicator: false`. The projection is then trained with
  AdamW, like the other input and output layers.
- **Unseen tasks**:
  - *Leave-one-out*: the held-out category gets a **zero** task vector in every arm.
    Experiment 5 scored `frozen_td` with the category's untrained random column instead, and
    kept no checkpoints to rescore, so `frozentd_latent` is retrained here as a fourth
    leave-one-out arm. Training is identical to experiment 5's `frozen_td` (same config and
    seed), so its in-distribution scores should match experiment 5's; only the held-out
    scoring differs. `notd` has no task vector, so experiment 5's own runs are used.
  - *Compositional*: a composite category gets **both of its components' embeddings** (a
    multi-hot task vector over its two base categories; `--mode mean` averages them instead).
    No new columns are needed, so unlike experiment 4 there is no padding. The component
    mapping is in `common.py`.
- **Cluster analysis** (`dump_representations.py`), four spaces per validation task:
  - `support`: the pooled encoder output, before any latent task identity;
  - `support_no_id`: the same with no task identity at all, i.e. what the encoder makes of
    the examples alone;
  - `task`: the representation the weights are generated from, the one experiment 2 plotted;
  - `weights`: the generated weights.

  Each space gets the 5-fold linear probe and the silhouette score. With latent placement,
  `task` contains the one-hot projection and clusters by construction, so arms are compared
  on the other three spaces.

## Run

```bash
GPUS=0,1,2 bash experiments/07_task_identity_ablation/run_indist.sh     # 15 jobs
bash experiments/07_task_identity_ablation/run_analysis.sh              # evaluation only
GPUS=all bash experiments/07_task_identity_ablation/run_loo.sh          # 280 jobs
PYTHONPATH=. uv run python experiments/07_task_identity_ablation/plot_all.py
```

- **Inputs from earlier experiments.** `run_analysis.sh` needs experiment 2's dim-4 runs, with
  their checkpoints, in `outputs/`, and the compositional data
  (`scripts/build_arc1d_compositional.py`). `plot_all.py` also reads experiment 5's `notd`
  runs.
- **Configs.** `gen_loo_configs.py` rewrites `configs/loo/` exactly.
- **Order.** Run the leave-one-out sweep last: it is the expensive step, so look at the
  in-distribution results first.

## Outputs

- `outputs/07_task_identity_ablation/`:
  - `indist_<arm>_seed<N>/` and `loo_<category>_<arm>_seed<N>/`: training runs;
  - `compositional/` and `representations/`: evaluations.
- `outputs/results/07_task_identity_ablation/*.csv` and
  `outputs/figures/07_task_identity_ablation/`, which contains:
  - held-out accuracy per category, all five arms;
  - a t-SNE of each space, one panel per arm;
  - the cosine similarity of the learned task embeddings.

## Result

Not run yet.

| | Verdict |
|---|---|
| H1 | |
| H2 | |
| H3 | |
| Shortcut | |

## Cost

Estimated, from experiments 2 and 5 on A40s:
- in-distribution: 15 runs × 24 min ≈ 6 GPU-hours;
- leave-one-out: 280 runs × 23 min ≈ 107 GPU-hours;
- evaluation (compositional and representations): under 1 GPU-hour.

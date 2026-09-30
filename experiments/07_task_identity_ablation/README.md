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

All 295 runs finished; 5 seeds per cell. Checks: `notd`'s compositional score (75.5% token
accuracy, 1.0% exact match) matches experiment 4, and the retrained `frozentd_latent`
leave-one-out runs match experiment 5's `frozen_td` in distribution (99.7%).

**In distribution** (experiment 2 setup, test query, mean ± s.d.):

| Arm | Trainable / total params | Token accuracy | Exact match |
|---|---:|---:|---:|
| `notd` | 10,156 / 11,360 | 95.2 ± 0.8% | 57.4 ± 5.1% |
| `frozentd_latent` | 10,156 / 11,432 | 99.6 ± 0.1% | 93.0 ± 2.4% |
| `learnedtd_latent` | 10,228 / 11,432 | 99.8 ± 0.1% | **96.5 ± 0.7%** |
| `frozentd_input` | 10,156 / 11,432 | 99.7 ± 0.1% | 95.1 ± 2.0% |
| `learnedtd_input` | 10,228 / 11,432 | 99.8 ± 0.1% | 96.2 ± 1.2% |

**Leave-one-out** (held-out category with a zero task vector, macro mean over 14 categories;
"drop" is in-distribution minus held-out token accuracy):

| Arm | Held-out token accuracy | Held-out exact match | Drop | Ahead of `notd` |
|---|---:|---:|---:|---:|
| `notd` | **81.6%** | **10.2%** | 14.9 | |
| `frozentd_latent` | 79.1% | 0.6% | 20.6 | 3 of 14 |
| `learnedtd_latent` | 77.3% | 1.1% | 22.5 | 3 of 14 |
| `frozentd_input` | 74.1% | 0.0% | 25.7 | 2 of 14 |
| `learnedtd_input` | 73.1% | 0.8% | 26.7 | 1 of 14 |

With a zero vector instead of experiment 5's untrained random column, `frozentd_latent`
rises from 67.5% to 79.1% held-out, but stays behind `notd`.

**Compositional** (10 composite categories, token accuracy, mean ± s.d.; exact match is at or
below 6% for every arm):

| Arm | Summed components (multi-hot) | Averaged components |
|---|---:|---:|
| `notd` | **75.5 ± 1.1%** | |
| `frozentd_latent` | 63.2 ± 7.4% | 67.8 ± 4.0% |
| `learnedtd_latent` | 66.4 ± 4.3% | 70.5 ± 3.3% |
| `frozentd_input` | 61.0 ± 2.7% | 60.9 ± 9.4% |
| `learnedtd_input` | 68.8 ± 2.6% | 68.0 ± 3.5% |

**Clustering** (validation set, 5-fold linear probe accuracy / silhouette, mean over seeds;
chance for the probe is 1/14 ≈ 7%):

| Arm | `support` | `support_no_id` | `weights` |
|---|---|---|---|
| `notd` | 0.61 / 0.22 | **0.61 / 0.22** | 0.75 / 0.24 |
| `frozentd_latent` | 0.25 / −0.07 | 0.25 / −0.07 | 1.00 / 0.93 |
| `learnedtd_latent` | 0.16 / −0.19 | 0.16 / −0.19 | 1.00 / 0.88 |
| `frozentd_input` | 0.94 / 0.83 | 0.20 / −0.27 | 0.99 / 0.84 |
| `learnedtd_input` | 0.93 / 0.79 | 0.22 / −0.24 | 0.96 / 0.80 |

`support_no_id` is the decisive column: it is what the encoder makes of the support examples
alone. Only `notd`'s encoder organises tasks by what the examples show. Every arm with a task
ID, wherever it is placed, has an encoder that barely separates the categories from the
examples: the task clusters in `support` (input placement) and `weights` come from the ID.

Figures: `loo_per_category`, `clusters_<space>_tsne_seed1` (one panel per arm) and
`task_embedding_similarity_seed1` in `outputs/figures/07_task_identity_ablation/`.

### Verdicts

| | Verdict |
|---|---|
| H1: learned latent = frozen latent | **Mostly supported.** Held-out (77.3% vs 79.1%, ahead in 3 of 14), compositional and clustering differences are within the seed s.d. In distribution, learned has higher exact match (96.5 ± 0.7% vs 93.0 ± 2.4%), so learning the offsets is not entirely free. The learned table does organise itself: the move variants become similar to each other, as do the two pattern-copy categories |
| H2: frozen input aligns the latent space, so it generalises better | **Refuted.** The encoder output clusters by task only while the ID is in it (probe 0.94); from the examples alone it is near chance (0.20, below `frozentd_latent`'s 0.25). Held-out (74.1% vs 79.1%, ahead in 2 of 14) and compositional accuracy are lower than latent placement |
| H3: learned input is better than frozen input | **Inconclusive.** Held-out is a tie (73.1% vs 74.1%, ahead in 7 of 14). Compositional with summed components is higher (68.8% vs 61.0%), but no better than `learnedtd_latent` with averaged ones. The 72 extra parameters therefore buy no clear gain, and the parameter-matched control is not needed |
| Shortcut alternative | **Supported.** Every task-ID arm lets the encoder rely on the ID instead of the examples, and input placement does this most: the largest in-distribution to held-out drops (25.7 and 26.7 points) and the least example-driven encoder. Task identity helps in distribution and costs generalisation, wherever it enters |

## Cost

Measured on A40s, from each run's `compute.json` (two leave-one-out jobs shared a GPU, and the
in-distribution runs shared their GPUs with those, so runs are slower than in experiments 2
and 5):

| Sweep | Runs | Median minutes per run | A40 GPU-hours |
|---|---:|---:|---:|
| In-distribution | 15 | 54.5 | 13 |
| Leave-one-out | 280 | 31.7 | 156 |
| Compositional and representations | evaluation only | | under 1 |

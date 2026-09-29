# Experiment 6, individual arm

See [`../README.md`](../README.md) for the question, data levels and results.

- **Model**: experiment 1's individual dim-4 recipe, unchanged: one direct Transformer per
  task category (the same architecture as the hypernetwork's target), no task identity.
- **Training**: as experiment 1 (Muon, `learning_rate` 0.0005, `embedding_dim` 10, 8,000
  steps), fixed across data levels. Evaluates the final weights.
- **Data**: each training row contributes its 3 support pairs (the hypernetwork is also
  supervised on the row's query, 4 pairs per row); validation and test are unchanged across
  levels.
- **Configs**: `configs/<category>/{t1..t20,v1,v2,v3,full}.yaml`, 14 categories x 9 levels =
  126, written by `gen_configs.py`; 5 seeds each = 630 jobs.

```bash
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/individual/run.sh
CATEGORY=1d_fill bash experiments/06_data_efficiency_ablation/individual/run.sh   # one category
```

Runs land in `outputs/06_data_efficiency_ablation_individual/`.

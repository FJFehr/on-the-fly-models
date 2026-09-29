# Experiment 6, joint arm

See [`../README.md`](../README.md) for the question, data levels and results.

- **Model**: experiment 1's joint dim-14 recipe, unchanged: one direct Transformer trained on
  all 14 categories, `td` (a learned per-category embedding added to every token) or `notd`.
  9,508 (`notd`) / 9,688 (`td`) parameters, sized to the hypernetwork's parameter budget
  rather than its dim-4 target, so that the comparison isolates weight generation rather than
  capacity.
- **Training**: as experiment 1 (Muon, `learning_rate` 0.0005, 8,000 steps, no gradient
  clipping), fixed across data levels. Evaluates the final weights.
- **Data**: each training row contributes its 3 support pairs; validation and test are
  unchanged across levels.
- **Configs**: `configs/cell_{td,notd}_{t1..t20,v1..v20,full}.yaml` (24), written by
  `gen_configs.py`; 5 seeds each = 120 jobs.

```bash
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/joint/run.sh
```

Runs land in `outputs/06_data_efficiency_ablation_joint/`.

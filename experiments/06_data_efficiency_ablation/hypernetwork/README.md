# Experiment 6, hypernetwork arm

See [`../README.md`](../README.md) for the question, data levels and results.

- **Model**: experiment 2's dim-4 hypernetwork, unchanged (`frozen_td`: `num_tasks` 18,
  `freeze_task_indicator: true`; `notd`: no task indicator). About 11.4K parameters.
- **Training**: as experiment 2 (Muon, 8,000 steps, batch 512, bf16), fixed across data
  levels. Evaluates the final weights.
- **Data**: the meta data module's training split is reduced per level
  (`variants_per_base_task`, `base_tasks_per_category`); validation and test are unchanged.
  The hypernetwork is supervised on all 4 examples of each training row (3 support pairs and
  the query), so it sees 4 pairs per row where the direct models see 3.
- **Configs**: `configs/cell_{frozentd,notd}_{t1..t20,v1..v20,full}.yaml` (24), written by
  `gen_configs.py`; 5 seeds each = 120 jobs.

```bash
GPUS=0,1,2,3 bash experiments/06_data_efficiency_ablation/hypernetwork/run.sh
```

Runs land in `outputs/06_data_efficiency_ablation_hypernetwork/`.

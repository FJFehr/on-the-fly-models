# Round 2, Phase 0.5: infra pilot

See `docs/arc1d_story/05_round2_plan.md`'s "Phase 0.5" section for the full
rationale. Five single-seed cells validating combinations that have never
been run before this round:

- **Direct-training + Muon** has only been used once historically
  (`arc1d_lowdata_baseline`), with `muon_lr` copied from the hypernetwork
  side, never independently validated for this smaller model.
- **`muon_lr=0.005` at `batch_size=2048` together** has never been run - 0.005
  was only validated at `batch_size=512` (`arc1d_hypermodel_looped_rope_canon_muon`);
  2048 was only validated at `muon_lr` in {0.008, 0.01, 0.016} (`muon_sweep`).
- **Full weight generation (no LoRA) + Muon** has never been run - every
  historical Muon experiment used the LoRA-adapter mode.

| Config | Lineage | Purpose |
|---|---|---|
| `direct_muon_lr005_nloops4.yaml` | direct (`arc1d_lowdata_baseline` base) | Baseline direct+Muon check |
| `direct_muon_lr02_nloops4.yaml` | direct | Does the much smaller direct model (11,920 params) tolerate Muon's un-lowered default LR (0.02) that diverged the 1.58M-param hypernetwork? |
| `hyper_muon_lr005_bsz2048_frozentd_full.yaml` | hypernetwork (`arc1d_hypermodel_looped_rope_canon_muon` base) | Full-generation mode at this (lr, batch_size) pairing, `frozen_td` |
| `hyper_muon_lr005_bsz2048_notd_full.yaml` | hypernetwork | Same, on the historically harder-to-stabilise `notd` arm |
| `hyper_muon_lr02_bsz2048_frozentd_full.yaml` | hypernetwork | Single-arm retest of `muon_moveablation`'s risky LR (17/42 = 40% finish rate historically) at n=1 |

Single category (`1d_fill`) for the direct pilots to keep them cheap; full
15-task set for the hypernetwork pilots (task count doesn't change per-step
wall clock materially, but training stability across the full task mix is
exactly what's being checked). `save_checkpoints: false` throughout - these
are throwaway, only the final metrics/logs matter.

Run each with `time` around the `train.py` call to get real wall-clock data
feeding Phase 1/2's job-count planning. Must show loss decreasing and no NaN
in all 5 cells before any other Round 2 phase commits to its full job count.

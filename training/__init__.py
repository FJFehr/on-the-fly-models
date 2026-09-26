"""Run mechanics shared by train.py and validate.py.

This package is about *how a run executes*, not what gets trained on what:
config loading and inheritance (`config.py`), Lightning trainer/callback
setup and checkpoint I/O (`trainer.py`), and everything that writes a run's
output artifacts -- model summaries, W&B logging, results.txt, hard-example
exports, task/embedding-cluster figure logging (`logging.py`).

No model or data-module definitions live here -- those are `models/`,
`lightning_modules/` and `data_modules/`. Submodules are imported directly (`from training.config
import load_config`, etc.); this file has no package-level re-exports of
its own.
"""

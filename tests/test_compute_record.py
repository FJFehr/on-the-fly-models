"""compute.json records a run's wall-clock time, adding up the segments of a resumed run.

scripts/compute_cost.py totals these files into each experiment's GPU-hours, so a resumed
run must add its time rather than overwrite what was already recorded.
"""

import json

from training.logging import write_compute_record


def test_resumed_run_adds_its_segment(tmp_path):
    config = {"devices": 1, "precision": "bf16-mixed"}
    write_compute_record(str(tmp_path), config, wall_clock_seconds=600)
    write_compute_record(str(tmp_path), config, wall_clock_seconds=300)

    record = json.loads((tmp_path / "compute.json").read_text())
    assert record["segments"] == [10.0, 5.0]
    assert record["wall_clock_minutes"] == 15.0
    assert record["precision"] == "bf16-mixed"
    assert record["devices"] == 1
    assert isinstance(record["gpu"], str)

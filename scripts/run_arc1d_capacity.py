"""Run all arc1d_capacity experiments across CPUs or one-job-per-GPU scheduling.

Discovers every architecture YAML under --config-dir (excluding any file whose
name starts with "base_") and launches one run per discovered config and seed.
GPU mode assigns each subprocess exactly one visible GPU via
``CUDA_VISIBLE_DEVICES=<gpu_id>``. CPU mode hides GPUs so Lightning falls back
to CPU.

Usage
-----
    uv run python scripts/run_arc1d_capacity.py \\
        --config-dir configs/experiments/arc1d_capacity_multiclass \\
        --gpus 0,1,2,3 \\
        --seeds 42,43,44

    uv run python scripts/run_arc1d_capacity.py \\
        --config-dir configs/experiments/arc1d_capacity_binary \\
        --max-parallel 8

    uv run python scripts/run_arc1d_capacity.py \\
        --config-dir configs/experiments/arc1d_capacity_binary \\
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from omegaconf import OmegaConf

DEFAULT_CPU_PARALLELISM = 4


@dataclass(frozen=True)
class ScheduledRun:
    """One seed-specific training job derived from one experiment config."""

    config_path: Path
    seed: int
    base_experiment_name: str

    @property
    def run_name(self) -> str:
        return f"{self.base_experiment_name}_seed_{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Root directory containing experiment YAML configs (searched recursively).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=("Directory for per-run log files. Defaults to outputs/<config-dir-name>/logs."),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of experiments to run concurrently.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids. When set, schedule one run per GPU at a time.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds to run for every discovered config (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without executing them.",
    )
    return parser.parse_args()


def discover_configs(config_root: Path) -> list[Path]:
    """Return all experiment YAML paths, excluding shared base configs."""
    return sorted(p for p in config_root.rglob("*.yaml") if not p.name.startswith("base_"))


def parse_csv_ints(raw: str | None, option_name: str) -> list[int]:
    """Parse a comma-separated integer list CLI value."""
    if raw is None:
        return []

    values = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            msg = f"{option_name} must be a comma-separated integer list, got {raw!r}."
            raise ValueError(msg) from exc

    if not values:
        msg = f"{option_name} must include at least one integer."
        raise ValueError(msg)
    return values


def load_config(config_path: Path):
    """Load and merge a config (resolving _base_), returning the OmegaConf object."""
    cfg = OmegaConf.load(config_path)
    if "_base_" in cfg:
        base_cfg = OmegaConf.load(cfg._base_)
        del cfg["_base_"]
        cfg = OmegaConf.merge(base_cfg, cfg)
    OmegaConf.resolve(cfg)
    return cfg


def load_experiment_name(config_path: Path) -> str:
    """Resolve the config's experiment_name after `_base_` merging."""
    cfg = load_config(config_path)
    experiment_name = cfg.get("experiment_name")
    if isinstance(experiment_name, str) and experiment_name:
        return experiment_name
    return f"{config_path.parent.name}_{config_path.stem}"


def resolve_output_root(config_path: Path) -> Path:
    """Return the output directory root for runs from this config."""
    cfg = load_config(config_path)
    output_dir = cfg.get("output_dir", "outputs")
    project_name = cfg.get("project_name", config_path.parent.name)
    return Path(output_dir) / project_name


def is_run_complete(run: ScheduledRun, output_root: Path) -> bool:
    """Return True if results.txt exists for this run.

    results.txt is written by train.py as the very last step after final
    validation and test passes, so it only exists if training completed
    successfully. best_model.ckpt alone is not sufficient — Lightning saves
    it during training, so it can exist even after a crash or kill.
    """
    return (output_root / run.run_name / "results.txt").exists()


def build_scheduled_runs(configs: Iterable[Path], seeds: list[int]) -> list[ScheduledRun]:
    """Expand every config into one scheduled run per seed."""
    runs = []
    for config_path in configs:
        base_experiment_name = load_experiment_name(config_path)
        for seed in seeds:
            runs.append(
                ScheduledRun(
                    config_path=config_path,
                    seed=seed,
                    base_experiment_name=base_experiment_name,
                )
            )
    return runs


def resolve_max_parallel(max_parallel: int | None, gpu_ids: list[int]) -> int:
    """Return the effective concurrency for CPU or GPU mode."""
    if max_parallel is not None and max_parallel < 1:
        msg = f"--max-parallel must be >= 1, got {max_parallel}."
        raise ValueError(msg)

    if gpu_ids:
        default_parallel = len(gpu_ids)
        return min(max_parallel or default_parallel, len(gpu_ids))
    return max_parallel or DEFAULT_CPU_PARALLELISM


def build_train_command(run: ScheduledRun) -> list[str]:
    """Build the seeded train.py command for one scheduled run."""
    return [
        "uv",
        "run",
        "python",
        "train.py",
        "--config",
        str(run.config_path),
        f"seed={run.seed}",
        f"experiment_name={run.run_name}",
    ]


def build_process_env(gpu_id: int | None) -> dict[str, str]:
    """Return the subprocess environment for CPU fallback or single-GPU mode."""
    env = dict(os.environ)
    if gpu_id is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def build_log_path(log_dir: Path, run: ScheduledRun) -> Path:
    """Return the per-run log file path."""
    return log_dir / f"{run.run_name}.log"


# Global registry of active child processes so Ctrl+C can terminate them.
_active_procs: list[subprocess.Popen] = []
_procs_lock = threading.Lock()


def _terminate_all() -> None:
    with _procs_lock:
        for proc in _active_procs:
            with suppress(ProcessLookupError):
                proc.terminate()


SKIP_RETURNCODE = -1  # sentinel: run was skipped (already complete)


def run_experiment(
    run: ScheduledRun,
    log_dir: Path,
    print_lock: threading.Lock,
    gpu_queue: Queue[int] | None,
    output_root: Path,
) -> tuple[str, int]:
    """Run one training job, streaming output to a log file.

    Skips the run (without consuming a GPU slot) if best_model.ckpt already
    exists in the expected output directory — meaning the run finished cleanly.
    Failed or incomplete runs (no checkpoint) are always retrained.
    """
    if is_run_complete(run, output_root):
        with print_lock:
            print(f"[SKIP ] {run.run_name}  (results.txt exists — already succeeded)")
        return run.run_name, SKIP_RETURNCODE

    log_path = build_log_path(log_dir, run)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_id = gpu_queue.get() if gpu_queue is not None else None
    cmd = build_train_command(run)
    env = build_process_env(gpu_id)
    device_label = f"gpu:{gpu_id}" if gpu_id is not None else "cpu"

    with print_lock:
        print(f"[START] {run.run_name}  ({device_label})")

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        with _procs_lock:
            _active_procs.append(proc)
        try:
            proc.wait()
        finally:
            with _procs_lock, suppress(ValueError):
                _active_procs.remove(proc)
            if gpu_queue is not None and gpu_id is not None:
                gpu_queue.put(gpu_id)

    with print_lock:
        status = "DONE " if proc.returncode == 0 else "FAIL "
        print(f"[{status}] {run.run_name}  ({device_label}, log: {log_path})")

    return run.run_name, proc.returncode


def main() -> None:
    args = parse_args()

    if not args.config_dir.is_dir():
        print(f"Config directory not found: {args.config_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        gpu_ids = parse_csv_ints(args.gpus, "--gpus")
        seeds = parse_csv_ints(args.seeds, "--seeds")
        max_parallel = resolve_max_parallel(args.max_parallel, gpu_ids)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    log_dir = args.log_dir or Path("outputs") / args.config_dir.name / "logs"

    configs = discover_configs(args.config_dir)
    if not configs:
        print(f"No experiment configs found under {args.config_dir}", file=sys.stderr)
        sys.exit(1)

    scheduled_runs = build_scheduled_runs(configs, seeds)
    # Resolve the output root from the first config (all configs in one dir share project_name).
    output_root = resolve_output_root(configs[0])
    gpu_mode = bool(gpu_ids)
    gpu_queue = Queue() if gpu_mode else None
    if gpu_queue is not None:
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)

    print(f"Found {len(configs)} experiment configs expanded to {len(scheduled_runs)} seeded runs")
    print(f"Config dir    : {args.config_dir}")
    print(f"Log dir       : {log_dir}")
    print(f"Seeds         : {seeds}")
    print(f"Execution mode: {'GPU' if gpu_mode else 'CPU'}")
    if gpu_mode:
        print(f"GPUs          : {gpu_ids}")
    print(f"Max parallel  : {max_parallel}")
    print()

    if args.dry_run:
        for index, run in enumerate(scheduled_runs):
            if is_run_complete(run, output_root):
                print(f"  [SKIP ] {run.run_name}  (results.txt exists — already succeeded)")
                continue
            cmd = build_train_command(run)
            device_label = f"gpu:{gpu_ids[index % len(gpu_ids)]}" if gpu_mode else "cpu"
            print(f"  [{run.run_name}] ({device_label}) {' '.join(cmd)}")
        return

    print_lock = threading.Lock()
    results: list[tuple[str, int]] = []

    def _sigint_handler(sig, frame) -> None:
        print("\nInterrupted — terminating all running experiments...", file=sys.stderr)
        _terminate_all()

    signal.signal(signal.SIGINT, _sigint_handler)

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(run_experiment, run, log_dir, print_lock, gpu_queue, output_root): run
            for run in scheduled_runs
        }
        for future in as_completed(futures):
            name, returncode = future.result()
            results.append((name, returncode))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    skipped = [(n, rc) for n, rc in sorted(results) if rc == SKIP_RETURNCODE]
    passed = [(n, rc) for n, rc in sorted(results) if rc == 0]
    failed = [(n, rc) for n, rc in sorted(results) if rc not in (0, SKIP_RETURNCODE)]
    for name, _ in skipped:
        print(f"  SKIP  {name}")
    for name, _ in passed:
        print(f"  PASS  {name}")
    for name, rc in failed:
        print(f"  FAIL  {name}  (exit {rc})")
    print()
    print(
        f"{len(passed)}/{len(results) - len(skipped)} seeded runs succeeded "
        f"({len(skipped)} skipped)."
    )

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

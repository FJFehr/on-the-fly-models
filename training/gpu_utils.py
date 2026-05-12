import math
import subprocess


def get_free_gpu_ids(max_memory_used_mb: int = 500) -> list[int]:
    """Return GPU IDs whose used memory is below threshold (considered free)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    free = []
    for line in result.stdout.strip().splitlines():
        idx, mem_used = line.split(", ")
        if int(mem_used) < max_memory_used_mb:
            free.append(int(idx))
    return free


def largest_power_of_2(n: int) -> int:
    """Largest power of 2 that is <= n, or 0 if n <= 0."""
    if n <= 0:
        return 0
    return 2 ** int(math.log2(n))


def resolve_free_gpus(max_memory_used_mb: int = 500) -> tuple[list[int], int]:
    """Return (selected_gpu_ids, count) — the largest power-of-2 subset of free GPUs."""
    free = get_free_gpu_ids(max_memory_used_mb)
    count = largest_power_of_2(len(free))
    return free[:count], count

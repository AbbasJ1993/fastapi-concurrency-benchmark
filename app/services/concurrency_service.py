"""Pure CPU-bound workload used by the benchmark."""

def cpu_bound_task(n: int) -> int:
    total = 0
    for i in range(n):
        total += (i * i) % 97
    return total

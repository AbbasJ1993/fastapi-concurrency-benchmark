# FastAPI Concurrency Benchmark

A small, reproducible FastAPI project comparing approaches to CPU-bound work: direct execution inside an async endpoint, offloading to a thread with `asyncio.to_thread`, and execution in a `ProcessPoolExecutor`.

**Scope:** This is a CPU-bound Python microbenchmark, **not** a computer-vision inference benchmark. The `/predict` endpoint validates uploaded images; it does not run a trained model.

## Reference environment

- **OS:** Windows 11
- **CPU:** Intel Core i7-11370H (4 physical cores, 8 logical threads)
- **Python:** 3.12.10
- **Process pool:** 1–8 workers, configured using `PROCESS_POOL_WORKERS`

Results are machine- and run-specific. The project was tested in this Windows environment; the benchmark runner's graceful shutdown implementation is Windows-specific.

## Setup (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

If activation is blocked, invoke `.\.venv\Scripts\python.exe` directly instead. `requirements.txt` contains the main dependencies; `requirements-lock.txt` is included as a snapshot of the development environment, but has not been independently verified on another machine.

## Run the API

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Available endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check |
| `POST /predict` | Uploaded-image validation (no model inference) |
| `GET /benchmark/direct` | CPU workload executed directly |
| `GET /benchmark/thread` | CPU workload offloaded to a thread |
| `GET /benchmark/process` | CPU workload submitted to the process pool |

## Run the process-pool benchmark

**Stop any manually running API instance on port 8000 first.** The benchmark script starts its own Uvicorn server for each worker configuration.

```powershell
python scripts/benchmark_process_pool.py
```

The script uses:

- Process-pool sizes: `1, 2, 3, 4, 5, 6, 7, 8`
- 8 concurrent HTTP requests per measured batch
- 5 measured batches per worker configuration
- 10,000,000 CPU-loop iterations per measured request
- 100,000 iterations per warm-up request
- One warm-up batch before each measured batch, containing as many concurrent requests as the configured worker count

The script records per-request client elapsed time and server-reported elapsed time, plus total elapsed time for each batch. **Batch wall time** is the time needed to complete all eight concurrent HTTP requests. The server-reported elapsed time includes time spent awaiting process-pool execution; it is not a pure CPU execution measurement.

`PROCESS_POOL_WORKERS` sets the number of `ProcessPoolExecutor` workers, **not** the number of Uvicorn server workers. On Windows, the runner starts Uvicorn in a new process group and requests graceful shutdown using `CTRL_BREAK_EVENT`. The reference run completed without observed leftover benchmark worker processes.

## Reference results (2026-11-10)

Mean of five measured batches per configuration:

| Process-pool workers | Mean batch wall time (s) | Mean request client time (s) |
| ---: | ---: | ---: |
| 1 | 4.5115 | 2.5470 |
| 2 | 2.6570 | 1.5689 |
| 3 | 2.0380 | 1.2837 |
| 4 | 1.6509 | 1.1665 |
| 5 | 1.5913 | **1.1566** |
| 6 | 1.6628 | 1.2017 |
| 7 | 1.7264 | 1.2591 |
| 8 | **1.4085** | 1.3943 |

Eight workers gave the shortest mean batch completion time in this run (approximately **3.20×** faster than one worker). Increasing worker count did **not** improve either metric monotonically.

**Worker-count interpretation:** In the recorded benchmark, 5 workers achieved the lowest mean client response time, while 8 workers achieved the shortest total batch completion time. The latency difference between 4 and 5 workers was less than 1%, so the results do not establish that 5 workers is the optimal configuration.

Reference CSVs:

- `benchmark_results/benchmark_summary_20261110_204043.csv`
- `benchmark_results/benchmark_details_20261110_204043.csv`

## Interpretation and limitations

- The CPU workload is a Python loop; results do not generalize automatically to native OpenCV operations, GPU inference, or I/O-bound APIs.
- A single Uvicorn server process is used; process-pool worker counts are varied separately.
- Worker configurations were run in ascending order, not randomized. Thermal state, CPU frequency, power settings, and background load were not controlled or recorded.
- Warm-up concurrency changes with worker count, and cold-start behavior is not measured separately.
- Client and server measurements include scheduling and queuing overhead; they are not isolated CPU execution times.
- Graceful shutdown and absence of leftover workers were checked on the reference Windows machine, but automated process-tree verification is not part of the script.
- No cross-platform behavior or exact third-party environment reproduction is claimed.

## Project structure

```text
app/                    FastAPI application, schemas, and services
scripts/                Concurrent and process-pool benchmark runners
tests/                  Automated tests
benchmark_results/      Reference benchmark CSVs
requirements.txt        Direct dependency specifications
requirements-lock.txt   Development environment snapshot
pytest.ini              Pytest configuration
```

## Reproducing results responsibly

Run tests before benchmarking, ensure port 8000 is free, close unnecessary CPU-intensive applications, and record your Python version, OS, processor, power mode, and benchmark configuration. Compare repeated runs rather than relying on one timing.

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
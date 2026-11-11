
import asyncio
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import signal
import httpx


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

WORKER_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]

NUM_REQUESTS = 8
NUM_RUNS = 5

ITERATIONS = 10_000_000
WARMUP_ITERATIONS = 100_000

REQUEST_TIMEOUT = 120.0
SERVER_START_TIMEOUT = 30.0

OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"


# --------------------------------------------------
# Server management
# --------------------------------------------------

def start_server(worker_count: int):
    """
    Start Uvicorn with the requested ProcessPool size.

    On Windows, create a new process group so that
    CTRL_BREAK_EVENT can be used for graceful shutdown.
    """

    env = os.environ.copy()
    env["PROCESS_POOL_WORKERS"] = str(worker_count)

    log_path = (
        OUTPUT_DIR / f"server_workers_{worker_count}.log"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_file = open(
        log_path,
        "w",
        encoding="utf-8",
    )

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                HOST,
                "--port",
                str(PORT),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    except Exception:
        log_file.close()
        raise

    print(
        f"Uvicorn started with PID={process.pid}, "
        f"ProcessPool workers={worker_count}"
    )

    return process, log_file


def stop_server(process, log_file):
    """
    Attempt graceful Uvicorn shutdown on Windows.
    """

    try:
        if process.poll() is not None:
            return

        print("Requesting graceful server shutdown...")

        process.send_signal(signal.CTRL_BREAK_EVENT)

        try:
            process.wait(timeout=30)

            print("Uvicorn exited gracefully.")

        except subprocess.TimeoutExpired:
            print(
                "Graceful shutdown timed out. "
                "Forcing server termination..."
            )

            process.kill()
            process.wait()

    finally:
        log_file.close()


async def wait_for_server(process):
    """
    Wait until the FastAPI health endpoint responds.
    """

    start_time = time.perf_counter()

    async with httpx.AsyncClient(timeout=2.0) as client:

        while time.perf_counter() - start_time < SERVER_START_TIMEOUT:

            if process.poll() is not None:
                raise RuntimeError(
                    "Uvicorn exited before the server became ready. "
                    "Check the corresponding server log."
                )

            try:
                response = await client.get(
                    f"{BASE_URL}/health"
                )

                if response.status_code == 200:
                    return

            except httpx.RequestError:
                pass

            await asyncio.sleep(0.5)

    raise TimeoutError(
        "Server did not become ready within the allowed time."
    )


# --------------------------------------------------
# Benchmark functions
# --------------------------------------------------

async def warmup_server(client, worker_count):
    """
    Send worker_count concurrent warm-up requests.

    Warm-up requests are excluded from benchmark measurements.
    """

    warmup_start = time.perf_counter()

    responses = await asyncio.gather(
        *(
            client.get(
                f"{BASE_URL}/benchmark/process",
                params={"n": WARMUP_ITERATIONS},
            )
            for _ in range(worker_count)
        ),
        return_exceptions=True,
    )

    errors = []

    for index, response in enumerate(responses, start=1):

        if isinstance(response, Exception):
            errors.append(
                f"Warm-up request {index}: {response}"
            )
            continue

        try:
            response.raise_for_status()

        except httpx.HTTPError as exc:
            errors.append(
                f"Warm-up request {index}: {exc}"
            )

    if errors:
        raise RuntimeError(
            "Warm-up failed:\n" + "\n".join(errors)
        )

    warmup_elapsed = time.perf_counter() - warmup_start

    return warmup_elapsed


async def run_single_batch(client, worker_count, run_number):
    """
    Send NUM_REQUESTS concurrent requests.
    """

    async def send_request(request_id):

        client_start = time.perf_counter()

        response = await client.get(
            f"{BASE_URL}/benchmark/process",
            params={"n": ITERATIONS},
        )

        client_elapsed = (
            time.perf_counter() - client_start
        )

        response.raise_for_status()
        data = response.json()

        return {
            "workers": worker_count,
            "run": run_number,
            "request_id": request_id,
            "client_elapsed": client_elapsed,
            "server_elapsed": data["elapsed_seconds"],
            "result": data["result"],
        }

    batch_start = time.perf_counter()

    results = await asyncio.gather(
        *(
            send_request(i)
            for i in range(1, NUM_REQUESTS + 1)
        )
    )

    total_elapsed = time.perf_counter() - batch_start

    return results, total_elapsed


# --------------------------------------------------
# CSV output
# --------------------------------------------------

def save_results(request_results, summary_results, timestamp):
    """
    Save detailed request results and summary results.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    details_path = (
        OUTPUT_DIR / f"benchmark_details_{timestamp}.csv"
    )

    summary_path = (
        OUTPUT_DIR / f"benchmark_summary_{timestamp}.csv"
    )

    detail_fields = [
        "workers",
        "run",
        "request_id",
        "client_elapsed",
        "server_elapsed",
        "result",
    ]

    with open(
        details_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=detail_fields,
        )

        writer.writeheader()
        writer.writerows(request_results)

    summary_fields = [
        "workers",
        "run",
        "total_elapsed",
        "average_client_elapsed",
        "average_server_elapsed",
    ]

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(summary_results)

    return details_path, summary_path


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

async def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_request_results = []
    all_summary_results = []

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    print("=" * 60)
    print("PROCESS POOL BENCHMARK")
    print("=" * 60)

    print(f"Iterations per request : {ITERATIONS:,}")
    print(f"Warm-up iterations     : {WARMUP_ITERATIONS:,}")
    print(f"Concurrent requests    : {NUM_REQUESTS}")
    print(f"Runs per configuration : {NUM_RUNS}")
    print(f"Worker configurations  : {WORKER_COUNTS}")

    for worker_count in WORKER_COUNTS:

        print("\n" + "=" * 60)
        print(f"TESTING WITH {worker_count} WORKER(S)")
        print("=" * 60)

        server_process = None
        log_file = None

        try:
            # Start server
            print("\nStarting FastAPI server...")

            server_process, log_file = start_server(
                worker_count
            )

            await wait_for_server(server_process)

            print("Server is ready.")

            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT
            ) as client:

                # Repeated benchmark runs
                for run_number in range(1, NUM_RUNS + 1):

                    print(
                        f"\nRun {run_number}/{NUM_RUNS}"
                    )

                    # ----------------------------------
                    # Warm-up
                    # ----------------------------------

                    print(
                        f"Warm-up: sending {worker_count} "
                        f"concurrent requests..."
                    )

                    warmup_elapsed = await warmup_server(
                        client,
                        worker_count,
                    )

                    print(
                        f"Warm-up completed in "
                        f"{warmup_elapsed:.4f} seconds."
                    )

                    # ----------------------------------
                    # Main benchmark
                    # ----------------------------------

                    results, total_elapsed = (
                        await run_single_batch(
                            client,
                            worker_count,
                            run_number,
                        )
                    )

                    all_request_results.extend(results)

                    average_client = sum(
                        item["client_elapsed"]
                        for item in results
                    ) / len(results)

                    average_server = sum(
                        item["server_elapsed"]
                        for item in results
                    ) / len(results)

                    summary = {
                        "workers": worker_count,
                        "run": run_number,
                        "total_elapsed": total_elapsed,
                        "average_client_elapsed": average_client,
                        "average_server_elapsed": average_server,
                    }

                    all_summary_results.append(summary)

                    print(
                        f"Total wall time : {total_elapsed:.4f} s"
                    )

                    print(
                        f"Avg client time : {average_client:.4f} s"
                    )

                    print(
                        f"Avg server time : {average_server:.4f} s"
                    )

                    for item in results:

                        print(
                            f"  Request {item['request_id']}: "
                            f"Client={item['client_elapsed']:.4f}s, "
                            f"Server={item['server_elapsed']:.4f}s"
                        )

        finally:

            if server_process is not None:

                print("\nStopping FastAPI server...")

                stop_server(
                    server_process,
                    log_file,
                )

                print("Server stopped.")

    # ----------------------------------------------
    # Save all results
    # ----------------------------------------------

    details_path, summary_path = save_results(
        all_request_results,
        all_summary_results,
        timestamp,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)

    print(f"\nDetailed results: {details_path}")
    print(f"Summary results : {summary_path}")

    print("\nAverage total time by worker count:")

    for worker_count in WORKER_COUNTS:

        times = [
            item["total_elapsed"]
            for item in all_summary_results
            if item["workers"] == worker_count
        ]

        average_time = sum(times) / len(times)

        print(
            f"{worker_count} worker(s): "
            f"{average_time:.4f} seconds"
        )


if __name__ == "__main__":
    asyncio.run(main())

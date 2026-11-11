
import asyncio
import time
import httpx


BASE_URL = "http://127.0.0.1:8000"

NUM_REQUESTS = 4
ITERATIONS = 10_000_000

METHODS = [
    "direct",
    "thread",
    "process",
]


async def run_batch(client, method):
    url = f"{BASE_URL}/benchmark/{method}"

    async def send_request(request_id):
        start = time.perf_counter()

        response = await client.get(
            url,
            params={"n": ITERATIONS},
        )

        elapsed = time.perf_counter() - start

        response.raise_for_status()

        data = response.json()

        return {
            "request_id": request_id,
            "client_elapsed": elapsed,
            "server_elapsed": data["elapsed_seconds"],
            "result": data["result"],
        }

    start = time.perf_counter()

    results = await asyncio.gather(
        *(
            send_request(i)
            for i in range(1, NUM_REQUESTS + 1)
        )
    )

    total_elapsed = time.perf_counter() - start

    print(f"\nMethod: {method}")
    print(f"Total wall time: {total_elapsed:.4f} seconds")

    for item in results:
        print(
            f"Request {item['request_id']}: "
            f"Client={item['client_elapsed']:.4f}s, "
            f"Server={item['server_elapsed']:.4f}s, "
            f"Result={item['result']}"
        )


async def main():
    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        # Warm up the Process Pool
        print("Warming up Process Pool...")

        response = await client.get(
            f"{BASE_URL}/benchmark/process",
            params={"n": 100_000},
        )

        response.raise_for_status()

        # Run each method separately
        for method in METHODS:
            await run_batch(client, method)


if __name__ == "__main__":
    asyncio.run(main())

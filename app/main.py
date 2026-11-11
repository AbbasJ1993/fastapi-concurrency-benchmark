
import os
import asyncio
import time
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.image_service import (
    MAX_FILE_SIZE,
    ImageDimensionsTooLargeError,
    ImageTooLargeError,
    InvalidImageError,
    inspect_image,
)

from app.schemas import PredictionResponse
from app.services.concurrency_service import cpu_bound_task


# --------------------------------------------------
# Process Pool lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    max_workers = int(
        os.getenv("PROCESS_POOL_WORKERS", "2")
    )

    if max_workers < 1:
        raise ValueError(
            "PROCESS_POOL_WORKERS must be >= 1"
        )

    process_pool = ProcessPoolExecutor(
        max_workers=max_workers
    )

    app.state.process_pool = process_pool

    print(
        f"Process Pool configured with "
        f"{max_workers} workers."
    )

    try:
        yield

    finally:
        print("Shutting down Process Pool...")

        await asyncio.to_thread(
            process_pool.shutdown,
            wait=True,
            cancel_futures=True,
        )

        print("Process Pool shutdown completed.")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

limiter = Limiter(
    key_func=get_remote_address
)

app = FastAPI(
    title="Computer Vision Inference API",
    description="An API for running computer vision models.",
    version="0.3.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


# --------------------------------------------------
# Existing configuration
# --------------------------------------------------

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_CONCURRENT_PREDICTIONS = 3

prediction_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_PREDICTIONS
)

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


# --------------------------------------------------
# Existing image upload service
# --------------------------------------------------

async def read_upload_limited(image: UploadFile) -> bytes:
    contents = bytearray()

    while True:
        chunk = await image.read(UPLOAD_CHUNK_SIZE)

        if not chunk:
            break

        if len(contents) + len(chunk) > MAX_FILE_SIZE:
            raise ImageTooLargeError(
                "Image size must not exceed 10 MiB."
            )

        contents.extend(chunk)

    return bytes(contents)


# --------------------------------------------------
# Existing endpoints
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    confidence_threshold: float = Form(
        0.5,
        ge=0.0,
        le=1.0,
    ),
):
    try:
        async with prediction_semaphore:

            if image.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported media type.",
                )

            image_bytes = await read_upload_limited(image)

            metadata = inspect_image(image_bytes)

            return {
                "status": "valid",
                "filename": image.filename,
                **metadata,
                "confidence_threshold": confidence_threshold,
            }

    except ImageTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc

    except ImageDimensionsTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc

    except InvalidImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        await image.close()


# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------

DEFAULT_ITERATIONS = 10_000_000


# --------------------------------------------------
# Benchmark endpoint 1: Direct execution
# --------------------------------------------------

@app.get("/benchmark/direct")
async def benchmark_direct(
    n: int = Query(
        DEFAULT_ITERATIONS,
        ge=1,
        le=50_000_000,
    ),
):
    start = time.perf_counter()

    result = cpu_bound_task(n)

    elapsed = time.perf_counter() - start

    return {
        "method": "direct",
        "iterations": n,
        "result": result,
        "elapsed_seconds": round(elapsed, 4),
    }


# --------------------------------------------------
# Benchmark endpoint 2: Thread Pool
# --------------------------------------------------

@app.get("/benchmark/thread")
async def benchmark_thread(
    n: int = Query(
        DEFAULT_ITERATIONS,
        ge=1,
        le=50_000_000,
    ),
):
    start = time.perf_counter()

    result = await asyncio.to_thread(
        cpu_bound_task,
        n,
    )

    elapsed = time.perf_counter() - start

    return {
        "method": "thread",
        "iterations": n,
        "result": result,
        "elapsed_seconds": round(elapsed, 4),
    }


# --------------------------------------------------
# Benchmark endpoint 3: Process Pool
# --------------------------------------------------

@app.get("/benchmark/process")
async def benchmark_process(
    request: Request,
    n: int = Query(
        DEFAULT_ITERATIONS,
        ge=1,
        le=50_000_000,
    ),
):
    loop = asyncio.get_running_loop()

    start = time.perf_counter()

    result = await loop.run_in_executor(
        request.app.state.process_pool,
        cpu_bound_task,
        n,
    )

    elapsed = time.perf_counter() - start

    return {
        "method": "process",
        "iterations": n,
        "result": result,
        "elapsed_seconds": round(elapsed, 4),
    }

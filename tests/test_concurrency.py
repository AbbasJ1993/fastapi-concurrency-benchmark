import asyncio
import pytest

from app.main import prediction_semaphore

from io import BytesIO

from httpx import ASGITransport, AsyncClient
from PIL import Image

import app.main as main

@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_operations():
    active_tasks = 0
    max_active_tasks = 0

    async def simulated_prediction():
        nonlocal active_tasks, max_active_tasks

        async with prediction_semaphore:
            active_tasks += 1

            max_active_tasks = max(
                max_active_tasks,
                active_tasks,
            )

            await asyncio.sleep(0.1)

            active_tasks -= 1

    await asyncio.gather(
        *(simulated_prediction() for _ in range(10))
    )

    assert max_active_tasks <= 3
    assert max_active_tasks == 3
    assert active_tasks == 0

def create_test_image():
    buffer = BytesIO()

    image = Image.new("RGB", (100, 100), color="red")
    image.save(buffer, format="JPEG")

    return buffer.getvalue()


def test_predict_limits_concurrent_requests(monkeypatch):

    async def run_test():
        # Create a fresh semaphore for this test's event loop.
        monkeypatch.setattr(
            main,
            "prediction_semaphore",
            asyncio.Semaphore(3),
        )

        active_requests = 0
        max_active_requests = 0

        original_read = main.read_upload_limited

        async def tracked_read(image):
            nonlocal active_requests, max_active_requests

            active_requests += 1

            max_active_requests = max(
                max_active_requests,
                active_requests,
            )

            try:
                # Simulate a short asynchronous operation.
                await asyncio.sleep(0.05)

                return await original_read(image)

            finally:
                active_requests -= 1

        monkeypatch.setattr(
            main,
            "read_upload_limited",
            tracked_read,
        )

        image_bytes = create_test_image()

        async with AsyncClient(
            transport=ASGITransport(app=main.app),
            base_url="http://test",
        ) as client:

            responses = await asyncio.gather(
                *[
                    client.post(
                        "/predict",
                        files={
                            "image": (
                                "test.jpg",
                                image_bytes,
                                "image/jpeg",
                            )
                        },
                        data={
                            "confidence_threshold": "0.5"
                        },
                    )
                    for _ in range(10)
                ]
            )

        assert all(
            response.status_code == 200
            for response in responses
        )

        assert max_active_requests == 3
        assert active_requests == 0

    asyncio.run(run_test())
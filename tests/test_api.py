from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

from app.main import limiter

client = TestClient(app)


def create_test_image(image_format="JPEG", size=(100, 100)):
    buffer = BytesIO()

    image = Image.new("RGB", size, color="red")
    image.save(buffer, format=image_format)

    return buffer.getvalue()


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image():
    image_bytes = create_test_image("JPEG")

    response = client.post(
        "/predict",
        files={
            "image": (
                "test.jpg",
                image_bytes,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "valid"
    assert response.json()["format"] == "JPEG"
    assert response.json()["width"] == 100
    assert response.json()["height"] == 100


def test_predict_empty_file():
    response = client.post(
        "/predict",
        files={
            "image": (
                "empty.jpg",
                b"",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_predict_invalid_image():
    response = client.post(
        "/predict",
        files={
            "image": (
                "invalid.jpg",
                b"This is not an image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_predict_large_file():
    large_bytes = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={
            "image": (
                "large.jpg",
                large_bytes,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413

def test_rate_limit():
    limiter.reset()
    responses = []

    for _ in range(11):
        response = client.post(
            "/predict",
            files={
                "image": (
                    "test.jpg",
                    create_test_image(),
                    "image/jpeg",
                )
            },
        )

        responses.append(response)

    assert all(
        response.status_code == 200
        for response in responses[:10]
    )

    assert responses[10].status_code == 429
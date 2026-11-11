
from io import BytesIO

import pytest
from PIL import Image

from app.image_service import (
    ImageDimensionsTooLargeError,
    ImageTooLargeError,
    InvalidImageError,
    inspect_image,
)


def create_test_image(
    image_format="JPEG",
    size=(100, 100),
):
    buffer = BytesIO()

    image = Image.new("RGB", size, color="red")
    image.save(buffer, format=image_format)

    return buffer.getvalue()


def test_valid_jpeg_image():
    image_bytes = create_test_image("JPEG")

    result = inspect_image(image_bytes)

    assert result["format"] == "JPEG"
    assert result["width"] == 100
    assert result["height"] == 100


def test_valid_png_image():
    image_bytes = create_test_image("PNG")

    result = inspect_image(image_bytes)

    assert result["format"] == "PNG"


def test_empty_file():
    with pytest.raises(InvalidImageError):
        inspect_image(b"")


def test_invalid_image():
    invalid_bytes = b"This is not an image"

    with pytest.raises(InvalidImageError):
        inspect_image(invalid_bytes)


def test_large_image_file():
    large_bytes = b"x" * (10 * 1024 * 1024 + 1)

    with pytest.raises(ImageTooLargeError):
        inspect_image(large_bytes)

def test_image_dimensions_too_large():
    image_bytes = create_test_image(
        image_format="JPEG",
        size=(5000, 5000),
    )

    with pytest.raises(ImageDimensionsTooLargeError):
        inspect_image(image_bytes)

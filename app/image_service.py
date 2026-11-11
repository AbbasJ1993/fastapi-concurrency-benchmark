
from io import BytesIO

from PIL import Image, UnidentifiedImageError


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 16_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImageError(Exception):
    """Raised when the uploaded file is not a valid supported image."""


class ImageTooLargeError(Exception):
    """Raised when the uploaded image exceeds the size limit."""

class ImageDimensionsTooLargeError(Exception):
    """Raised when image dimensions exceed the allowed limits."""


def inspect_image(image_bytes: bytes) -> dict:
    """
    Validate an image and return its basic metadata.
    """

    if not image_bytes:
        raise InvalidImageError("The uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise ImageTooLargeError(
            "Image size must not exceed 10 MiB."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            image_format = img.format

            if image_format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    "Only JPEG, PNG, and WEBP images are supported."
                )

            width, height = img.size

            if (
                width > MAX_IMAGE_WIDTH
                or height > MAX_IMAGE_HEIGHT
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ImageDimensionsTooLargeError(
                    "Image dimensions exceed the allowed limits."
                )

            img.verify()

        # verify() checks the image structure.
        # Reopen the image to read its dimensions.
        with Image.open(BytesIO(image_bytes)) as img:
            width, height = img.size

    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        ) from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "The image is corrupted or invalid."
        ) from exc

    return {
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(image_bytes),
    }

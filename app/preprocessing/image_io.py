from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError


async def load_upload_as_rgb(file: UploadFile) -> Image.Image:
    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents))
        return image.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError("Uploaded file is not a valid image.") from e


def load_image_as_rgb(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        return image.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"File is not a valid image: {image_path}") from e


def load_mask_as_l(image_path: str | Path) -> Image.Image:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Mask not found: {image_path}")

    try:
        image = Image.open(image_path)
        return image.convert("L")
    except UnidentifiedImageError as e:
        raise ValueError(f"File is not a valid mask image: {image_path}") from e

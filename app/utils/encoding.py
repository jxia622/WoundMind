import base64
from io import BytesIO

from PIL import Image


def pil_to_base64_png(image: Image.Image) -> dict:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "mime_type": "image/png",
        "data": encoded,
        "width": image.width,
        "height": image.height,
    }

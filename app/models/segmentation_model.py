from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import SEGMENTATION_CHECKPOINT_PATH


def build_segmentation_model(
    encoder_name: str = "efficientnet-b0",
    device: torch.device | None = None,
) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            "segmentation_models_pytorch is required for the segmentation model."
        ) from e

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(device)


def load_segmentation_model(
    checkpoint_path: Path = SEGMENTATION_CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_name = checkpoint.get("encoder_name", "efficientnet-b0")
    image_size = int(checkpoint.get("image_size", 512))

    model = build_segmentation_model(encoder_name=encoder_name, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata = {
        "encoder_name": encoder_name,
        "image_size": image_size,
        "checkpoint_path": str(checkpoint_path),
    }
    return model, metadata


class SegmentationModel:
    """
    Wound segmentation wrapper.
    """

    def __init__(
        self,
        checkpoint_path: Path = SEGMENTATION_CHECKPOINT_PATH,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.metadata = {"checkpoint_path": str(checkpoint_path), "placeholder": True}
        self.available = False
        self.warnings: list[str] = []
        self.image_size = 512
        self.transform = None

        try:
            self.model, self.metadata = load_segmentation_model(
                checkpoint_path=checkpoint_path,
                device=self.device,
            )
            self.available = True
            self.metadata["placeholder"] = False
            self.image_size = int(self.metadata.get("image_size", 512))
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        except Exception as e:
            self.warnings.append(
                f"Segmentation model unavailable; using placeholder mask. Reason: {e}"
            )

    def predict(self, image: Image.Image) -> dict:
        if self.model is None:
            mask = self._placeholder_mask(image)
            return {
                "mask": mask,
                "mask_available": True,
                "mask_source": "placeholder",
                "warnings": self.warnings,
                "metadata": self.metadata,
            }

        original_size = image.size
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        mask_array = (probability * 255).round().clip(0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_array, mode="L")
        mask = mask.resize(original_size, resample=Image.BILINEAR)

        return {
            "mask": mask,
            "mask_available": True,
            "mask_source": "model",
            "warnings": [],
            "metadata": self.metadata,
        }

    @staticmethod
    def _placeholder_mask(image: Image.Image) -> Image.Image:
        width, height = image.size
        y, x = np.ogrid[:height, :width]
        cx, cy = width / 2.0, height / 2.0
        rx, ry = width * 0.28, height * 0.28
        mask = (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0).astype(np.uint8) * 255
        return Image.fromarray(mask, mode="L")

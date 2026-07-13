from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.preprocessing.image_io import load_image_as_rgb


IMAGE_SIZE = 224
RESIZE_SIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


eval_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


aux_transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image_tensor = eval_transform(image.convert("RGB"))
    return image_tensor.unsqueeze(0)


def preprocess_image_path(image_path: str | Path) -> torch.Tensor:
    image = load_image_as_rgb(image_path)
    return preprocess_image(image)


def preprocess_rgb_mask_depth(
    image: Image.Image,
    mask: Image.Image | None = None,
    depth: Image.Image | None = None,
) -> torch.Tensor:
    rgb_tensor = eval_transform(image.convert("RGB"))
    channels = [rgb_tensor]

    # ConvNeXt DFU depth+mask notebooks trained channel 3 as depth and
    # channel 4 as segmentation mask.
    if depth is not None:
        channels.append(aux_transform(depth.convert("L")).clamp(0.0, 1.0))

    if mask is not None:
        channels.append(aux_transform(mask.convert("L")).clamp(0.0, 1.0))

    return torch.cat(channels, dim=0).unsqueeze(0)

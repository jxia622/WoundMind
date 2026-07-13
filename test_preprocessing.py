from pathlib import Path

from app.preprocessing import preprocess_image_path


def main():
    image_path = input("Enter path to a test image: ").strip()
    image_path = Path(image_path)

    tensor = preprocess_image_path(image_path)

    print(f"Preprocessed tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor min: {tensor.min().item():.4f}")
    print(f"Tensor max: {tensor.max().item():.4f}")

    expected_shape = (1, 3, 224, 224)

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tuple(tensor.shape)}")

    print("\nStep 5 preprocessing test passed.")


if __name__ == "__main__":
    main()

import torch

from app.model import load_model, load_class_map


def main():
    print("Loading class map...")
    idx_to_class = load_class_map()
    print(f"Class map loaded: {len(idx_to_class)} classes")

    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    print(f"Model loaded successfully on device: {device}")

    print("\nRunning dummy inference...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    expected_shape = (1, len(idx_to_class))
    actual_shape = tuple(output.shape)

    if actual_shape != expected_shape:
        raise ValueError(
            f"Expected output shape {expected_shape}, but got {actual_shape}"
        )

    print("\nStep 4 test passed.")


if __name__ == "__main__":
    main()

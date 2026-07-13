from pathlib import Path

from app.inference import load_inference_components, predict_image


def main():
    image_path = input("Enter path to a test image: ").strip().strip('"').strip("'")
    image_path = Path(image_path)

    print("\nLoading inference components...")
    model, idx_to_class, device, temperature = load_inference_components()
    print(f"Loaded model on device: {device}")
    print(f"Loaded class map with {len(idx_to_class)} classes")
    print(f"Temperature: {temperature}")

    print("\nRunning prediction...")
    result = predict_image(
        image_path=image_path,
        model=model,
        idx_to_class=idx_to_class,
        device=device,
        top_k=3,
        temperature=temperature,
    )

    print("\nPrediction result:")
    print(f"Predicted class index: {result['predicted_class_index']}")
    print(f"Predicted class name: {result['predicted_class_name']}")
    print(f"Predicted probability: {result['predicted_probability']:.4f}")
    print(f"Confidence level: {result['confidence_level']}")
    print(f"Prediction runtime: {result['prediction_time_seconds']:.4f} seconds")
    print(f"Top-2 probability margin: {result['top2_probability_margin']:.4f}")
    print(f"Normalized entropy: {result['normalized_entropy']:.4f}")

    print("\nTop predictions:")
    for item in result["top_predictions"]:
        print(
            f"{item['rank']}. "
            f"{item['class_index']}: "
            f"{item['class_name']} "
            f"({item['probability']:.4f})"
        )

    print("\nStep 6 inference test passed.")


if __name__ == "__main__":
    main()

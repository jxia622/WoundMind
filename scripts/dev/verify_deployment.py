from pathlib import Path
import json
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

checkpoint_path = PROJECT_ROOT / "checkpoints" / "convnext_tiny_best.pt"
class_map_path = PROJECT_ROOT / "app" / "condition_class_map.json"

print("Checking deployment files...\n")

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

if not class_map_path.exists():
    raise FileNotFoundError(f"Class map not found: {class_map_path}")

print(f"Checkpoint found: {checkpoint_path}")
print(f"Checkpoint size: {checkpoint_path.stat().st_size / (1024 * 1024):.2f} MB")

with open(class_map_path, "r") as f:
    class_map = json.load(f)

print(f"Class map found: {class_map_path}")
print(f"Number of classes in class map: {len(class_map)}")

print("\nFirst few class-map entries:")
for i, item in enumerate(class_map.items()):
    if i >= 5:
        break
    print(item)

print("\nTrying to load checkpoint with torch.load...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print("Checkpoint loaded successfully.")
print(f"Checkpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"- {key}")

print("\nStep 2 verification complete.")

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent
class_map_path = PROJECT_ROOT / "app" / "condition_class_map.json"

with open(class_map_path, "r") as f:
    class_map = json.load(f)

print("Raw class map:")
print(class_map)

print("\nChecking keys and values...")
for key, value in class_map.items():
    print(f"{key} -> {value}")

# Convert string keys to integer keys
idx_to_class = {int(k): v for k, v in class_map.items()}

print("\nConverted integer index map:")
for idx in sorted(idx_to_class.keys()):
    print(f"{idx} -> {idx_to_class[idx]}")

expected_indices = list(range(len(idx_to_class)))
actual_indices = sorted(idx_to_class.keys())

if actual_indices != expected_indices:
    raise ValueError(
        f"Class indices are not continuous from 0 to {len(idx_to_class)-1}. "
        f"Actual indices: {actual_indices}"
    )

print("\nClass map check passed.")

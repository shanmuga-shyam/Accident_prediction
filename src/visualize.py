import matplotlib.pyplot as plt
from train import train_models
from pathlib import Path

# Run training to get results
results = train_models()

plt.figure(figsize=(10,6))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("ML Model Accuracy Comparison")

plt.tight_layout()

# Ensure outputs directory exists (project root is parent of src)
project_root = Path(__file__).resolve().parent.parent
outputs_dir = project_root / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

out_path = outputs_dir / "model_accuracy.png"
plt.savefig(out_path)
plt.show()
import joblib
import pandas as pd
from pathlib import Path
import sys

# Resolve model paths relative to project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
models_dir = PROJECT_ROOT / "models"
model_path = models_dir / "best_model.pkl"
scaler_path = models_dir / "scaler.pkl"

def load_or_exit(path, desc):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"Required {desc} not found: {path}")
        print("Run `python src/train.py` from the project root to generate models.")
        sys.exit(1)

# Load model and scaler
model = load_or_exit(model_path, "model")
scaler = load_or_exit(scaler_path, "scaler")

# Example future driver data (replace with real features/order)
sample = pd.DataFrame([{
    'Age Group': 2,
    'Speed of Vehicle': 3,
    'Shift': 1,
    'Alcohol': 1,
    'Driving Experience': 4
}])

sample = scaler.transform(sample)

prediction = model.predict(sample)
prob = None
try:
    prob = model.predict_proba(sample)
except Exception:
    pass

print("Predicted Accident Type:", prediction)
print("Accident Risk Probability:", prob)
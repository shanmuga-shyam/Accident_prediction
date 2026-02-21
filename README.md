# Accident Risk Prediction — Preprocessing

This repository provides a small, production-ready preprocessing module used to
convert UI inputs into a weighted, normalized feature vector for downstream
machine-learning models. It focuses on safe, conservative encodings so that
unknown or malformed inputs do not understate risk.

Key behaviors:
- Domain-weighted encoding for categorical inputs (weather, road type, etc.).
- Normalization of numerical inputs to a 0–1 scale with sensible clamping.
- Feature importance multipliers applied to produce a weighted vector.
- A simple baseline risk score computed from the weighted features.

## Where to find things

- `src/feature_builder.py`: implementation of `build_feature_vector()` and
    `compute_baseline_risk()` (preprocessing only — no model training).

## Quick start

1. Create and activate a Python virtual environment (Python 3.8+ recommended):

```bash
python -m venv .venv
# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
# or cmd.exe:
.\.venv\Scripts\activate.bat
```

2. Install dependencies if any are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Create a `models/` directory to store trained model artifacts. This
     repository's `.gitignore` already excludes `/models/`, so you can safely
     keep large model files there.

```bash
mkdir models
```

## Example usage

```python
from src.feature_builder import build_feature_vector, compute_baseline_risk

input_data = {
        "weather": "Rain",
        "road_type": "Highway",
        "time_of_day": "Night",
        "road_condition": "Wet",
        "light_condition": "Night Lit",
        "traffic_density": 7,
        "speed_limit": 100,
        "vehicles_nearby": 5,
        "driver_age": 28,
        "driving_experience": 5,
        "alcohol": 0.0,
}

vec = build_feature_vector(input_data)
risk = compute_baseline_risk(vec)

print("Feature vector:", vec)
print("Baseline risk:", risk)
```

Returned vector order:

```
[weather, road_type, time_of_day, road_condition, light_condition,
 traffic, speed, vehicles, age, experience, alcohol]
```

## Notes and recommendations

- Unknown categorical values are treated conservatively (mapped to high risk)
    and emit a warning. This prevents accidental underestimation of risk.
- Numerical values outside expected ranges are clamped to reasonable bounds.
- Add unit tests for edge cases (missing keys, out-of-range numbers,
    unexpected categories) to ensure stability.

If you'd like, I can add the `src/feature_builder.py` implementation and
example unit tests next.


"""Train a regressor using the weighted feature vectors produced by
`src.feature_builder` and save the model to `models/accident_risk_model.joblib`.

This generates synthetic inputs (matching UI ranges), builds feature vectors
using the exact weighting rules, computes baseline risk for labels, and
trains a RandomForestRegressor.
"""
import os
import random
from typing import Dict, List

from joblib import dump
from sklearn.ensemble import RandomForestRegressor

from src.feature_builder import build_feature_vector, compute_baseline_risk

MODEL_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "accident_risk_model.joblib")


def _random_input() -> Dict:
    return {
        "weather": random.choice(["Clear", "Cloudy", "Rain", "Fog", "Storm"]),
        "road_type": random.choice(["Urban", "Rural", "Highway"]),
        "time_of_day": random.choice(["Morning", "Afternoon", "Evening", "Night"]),
        "road_condition": random.choice(["Dry", "Wet", "Snow", "Ice"]),
        "light_condition": random.choice(["Daylight", "Dawn/Dusk", "Night Lit", "No Street Light"]),
        "traffic_density": random.uniform(0, 10),
        "speed_limit": random.uniform(20, 120),
        "vehicles_nearby": random.uniform(0, 20),
        "driver_age": random.uniform(18, 75),
        "driving_experience": random.uniform(0, 40),
        "alcohol": random.uniform(0, 1),
    }


def build_dataset(n: int = 3000):
    X = []
    y = []
    for _ in range(n):
        inp = _random_input()
        vec = build_feature_vector(inp)
        label = compute_baseline_risk(vec)
        X.append(vec)
        y.append(label)
    return X, y


def train_and_save(n_samples: int = 3000):
    X, y = build_dataset(n_samples)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    print("Training regressor on", len(X), "samples...")
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    dump(model, MODEL_OUT)
    print("Saved model to", MODEL_OUT)


if __name__ == '__main__':
    train_and_save(4000)

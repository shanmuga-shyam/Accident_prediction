"""Feature engineering using domain-weighted encoding and importance multipliers.

This module implements the exact weighting scheme you provided and exposes:
- build_feature_vector(input_dict) -> List[float]
- compute_baseline_risk(feature_vector) -> float

All values are clamped to [0,1] before applying importance multipliers.
"""
from typing import Dict, List
import warnings

# --- Categorical risk weights (as specified) ---
WEATHER_WEIGHTS = {
    "Clear": 0.00,
    "Cloudy": 0.10,
    "Rain": 0.40,
    "Fog": 0.70,
    "Storm": 1.00,
}
ROAD_TYPE_WEIGHTS = {"Urban": 0.30, "Rural": 0.50, "Highway": 0.80}
TIME_WEIGHTS = {"Morning": 0.40, "Afternoon": 0.20, "Evening": 0.60, "Night": 1.00}
ROAD_COND_WEIGHTS = {"Dry": 0.00, "Wet": 0.50, "Snow": 0.80, "Ice": 1.00}
LIGHT_WEIGHTS = {
    "Daylight": 0.00,
    "Dawn/Dusk": 0.40,
    "Night Lit": 0.70,
    "No Street Light": 1.00,
}

# --- Importance multipliers (exact order provided) ---
IMPORTANCE = {
    "alcohol": 2.0,
    "speed": 1.7,
    "road_condition": 1.6,
    "light_condition": 1.5,
    "weather": 1.4,
    "time_of_day": 1.3,
    "traffic": 1.2,
    "vehicles": 1.1,
    "age": 1.0,
    "experience": 1.0,
    "road_type": 1.0,
}
SUM_IMPORTANCES = sum(IMPORTANCE.values())


def _clamp01(x: float) -> float:
    try:
        xf = x
    except Exception:
        return 1.0
    if xf != xf:
        return 1.0
    return 0.0 if xf < 0.0 else min(xf, 1.0)


def _lookup(mapping: Dict[str, float], key: object, name: str) -> float:
    if key is None:
        warnings.warn(f"Missing {name}; defaulting to max risk (1.0)")
        return 1.0
    k = str(key)
    if k in mapping:
        return _clamp01(mapping[k])
    warnings.warn(f"Unknown {name} '{k}'; defaulting to max risk (1.0)")
    return 1.0


def build_feature_vector(input_dict: Dict) -> List[float]:
    """Encode, normalize and apply importance multipliers.

    Returns vector in order:
    [weather, road_type, time_of_day, road_condition, light_condition,
     traffic, speed, vehicles, age, experience, alcohol]
    """
    required = [
        "weather",
        "road_type",
        "time_of_day",
        "road_condition",
        "light_condition",
        "traffic_density",
        "speed_limit",
        "vehicles_nearby",
        "driver_age",
        "driving_experience",
        "alcohol",
    ]
    if missing := [k for k in required if k not in input_dict]:
        raise ValueError(f"Missing required keys: {missing}")

    # categorical
    weather_w = _lookup(WEATHER_WEIGHTS, input_dict.get("weather"), "weather")
    road_type_w = _lookup(ROAD_TYPE_WEIGHTS, input_dict.get("road_type"), "road_type")
    time_w = _lookup(TIME_WEIGHTS, input_dict.get("time_of_day"), "time_of_day")
    road_cond_w = _lookup(ROAD_COND_WEIGHTS, input_dict.get("road_condition"), "road_condition")
    light_w = _lookup(LIGHT_WEIGHTS, input_dict.get("light_condition"), "light_condition")

    # numerical with clamping to expected ranges
    try:
        traffic_raw = float(input_dict.get("traffic_density", 0.0))
    except Exception:
        traffic_raw = 0.0
    traffic_raw = max(0.0, min(10.0, traffic_raw))
    traffic_w = _clamp01(traffic_raw / 10.0)

    try:
        speed_raw = float(input_dict.get("speed_limit", 20.0))
    except Exception:
        speed_raw = 20.0
    speed_raw = max(20.0, min(120.0, speed_raw))
    speed_w = _clamp01(speed_raw / 120.0)

    try:
        vehicles_raw = float(input_dict.get("vehicles_nearby", 0.0))
    except Exception:
        vehicles_raw = 0.0
    vehicles_raw = max(0.0, min(20.0, vehicles_raw))
    vehicles_w = _clamp01(vehicles_raw / 20.0)

    try:
        age_raw = float(input_dict.get("driver_age", 40.0))
    except Exception:
        age_raw = 40.0
    age_raw = max(18.0, min(75.0, age_raw))
    age_w = _clamp01(abs(age_raw - 40.0) / 40.0)

    try:
        exp_raw = float(input_dict.get("driving_experience", 0.0))
    except Exception:
        exp_raw = 0.0
    exp_raw = max(0.0, min(40.0, exp_raw))
    experience_w = _clamp01(1.0 - (exp_raw / 40.0))

    try:
        alcohol_raw = float(input_dict.get("alcohol", 0.0))
    except Exception:
        alcohol_raw = 0.0
    alcohol_raw = max(0.0, min(1.0, alcohol_raw))
    alcohol_w = _clamp01(alcohol_raw)

    # Apply importance multipliers
    vec = [
        weather_w * IMPORTANCE["weather"],
        road_type_w * IMPORTANCE["road_type"],
        time_w * IMPORTANCE["time_of_day"],
        road_cond_w * IMPORTANCE["road_condition"],
        light_w * IMPORTANCE["light_condition"],
        traffic_w * IMPORTANCE["traffic"],
        speed_w * IMPORTANCE["speed"],
        vehicles_w * IMPORTANCE["vehicles"],
        age_w * IMPORTANCE["age"],
        experience_w * IMPORTANCE["experience"],
        alcohol_w * IMPORTANCE["alcohol"],
    ]

    return [float(max(0.0, v)) for v in vec]


def build_normalized_vector(input_dict: Dict) -> List[float]:
    """Return the normalized (0-1) feature values before applying importance.

    Order matches `build_feature_vector`:
    [weather, road_type, time_of_day, road_condition, light_condition,
     traffic, speed, vehicles, age, experience, alcohol]
    """
    required = [
        "weather",
        "road_type",
        "time_of_day",
        "road_condition",
        "light_condition",
        "traffic_density",
        "speed_limit",
        "vehicles_nearby",
        "driver_age",
        "driving_experience",
        "alcohol",
    ]
    if missing := [k for k in required if k not in input_dict]:
        raise ValueError(f"Missing required keys: {missing}")

    weather_w = _lookup(WEATHER_WEIGHTS, input_dict.get("weather"), "weather")
    road_type_w = _lookup(ROAD_TYPE_WEIGHTS, input_dict.get("road_type"), "road_type")
    time_w = _lookup(TIME_WEIGHTS, input_dict.get("time_of_day"), "time_of_day")
    road_cond_w = _lookup(ROAD_COND_WEIGHTS, input_dict.get("road_condition"), "road_condition")
    light_w = _lookup(LIGHT_WEIGHTS, input_dict.get("light_condition"), "light_condition")

    try:
        traffic_raw = float(input_dict.get("traffic_density", 0.0))
    except Exception:
        traffic_raw = 0.0
    traffic_raw = max(0.0, min(10.0, traffic_raw))
    traffic_w = _clamp01(traffic_raw / 10.0)

    try:
        speed_raw = float(input_dict.get("speed_limit", 20.0))
    except Exception:
        speed_raw = 20.0
    speed_raw = max(20.0, min(120.0, speed_raw))
    speed_w = _clamp01(speed_raw / 120.0)

    try:
        vehicles_raw = float(input_dict.get("vehicles_nearby", 0.0))
    except Exception:
        vehicles_raw = 0.0
    vehicles_raw = max(0.0, min(20.0, vehicles_raw))
    vehicles_w = _clamp01(vehicles_raw / 20.0)

    try:
        age_raw = float(input_dict.get("driver_age", 40.0))
    except Exception:
        age_raw = 40.0
    age_raw = max(18.0, min(75.0, age_raw))
    age_w = _clamp01(abs(age_raw - 40.0) / 40.0)

    try:
        exp_raw = float(input_dict.get("driving_experience", 0.0))
    except Exception:
        exp_raw = 0.0
    exp_raw = max(0.0, min(40.0, exp_raw))
    experience_w = _clamp01(1.0 - (exp_raw / 40.0))

    try:
        alcohol_raw = float(input_dict.get("alcohol", 0.0))
    except Exception:
        alcohol_raw = 0.0
    alcohol_raw = max(0.0, min(1.0, alcohol_raw))
    alcohol_w = _clamp01(alcohol_raw)

    return [
        weather_w,
        road_type_w,
        time_w,
        road_cond_w,
        light_w,
        traffic_w,
        speed_w,
        vehicles_w,
        age_w,
        experience_w,
        alcohol_w,
    ]


def compute_baseline_risk(feature_vector: List[float]) -> float:
    if not isinstance(feature_vector, (list, tuple)):
        raise ValueError("feature_vector must be list/tuple of numeric values")
    if len(feature_vector) != 11:
        raise ValueError("feature_vector must have length 11")
    total = float(sum(float(x) for x in feature_vector))
    if SUM_IMPORTANCES <= 0:
        raise RuntimeError("Invalid IMPORTANCE configuration")
    risk = total / SUM_IMPORTANCES
    return _clamp01(risk)


__all__ = ["build_feature_vector", "compute_baseline_risk", "build_normalized_vector"]

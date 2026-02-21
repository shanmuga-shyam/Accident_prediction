"""Future prediction and prevention utilities for Road Accident Risk Assistant.

This module extends the existing predictor with:
- time-based future risk forecasting
- driver behavior prediction
- accident severity prediction
- safety recommendation generation
- traffic density forecasting
- emergency response time estimation
- a simple driver profile store

All functions are lightweight, use the existing preprocessing in
`src.feature_builder`, and simulate patterns where needed.
"""
from typing import Dict, List, Tuple, Optional
import math
import os
import json
import statistics
from datetime import datetime, timedelta

from src.feature_builder import build_feature_vector, compute_baseline_risk, build_normalized_vector


def _shift_time_of_day(current: str, hour_offset: int) -> str:
    """Shift categorical time_of_day by hour offset. Simple mapping.

    Categories: Morning (6-11), Afternoon (12-16), Evening (17-20), Night (21-5)
    """
    # approximate hour for category
    mapping_to_hour = {
        "Morning": 8,
        "Afternoon": 14,
        "Evening": 18,
        "Night": 23,
    }
    # find base hour
    base = mapping_to_hour.get(current, 12)
    new_hour = (base + hour_offset) % 24
    if 6 <= new_hour <= 11:
        return "Morning"
    if 12 <= new_hour <= 16:
        return "Afternoon"
    return "Evening" if 17 <= new_hour <= 20 else "Night"


def forecast_risk(current_features: Dict, hour_offset: int) -> Tuple[float, str]:
    """Forecast accident risk after `hour_offset` hours.

    - Simulates time shift and adjusts `light_condition` and `traffic_density`.
    - Increases fatigue risk for late-night forecasts.

    Returns (risk_score 0-1, label str).
    """
    # copy and adjust features
    f = dict(current_features)
    # shift time
    cur_time = f.get("time_of_day", "Afternoon")
    future_time = _shift_time_of_day(cur_time, hour_offset)
    f["time_of_day"] = future_time

    # set light_condition based on time_of_day
    if future_time == "Night":
        f["light_condition"] = "No Street Light"
    elif future_time == "Evening":
        f["light_condition"] = "Dawn/Dusk"
    else:
        f["light_condition"] = "Daylight"

    # predict traffic for that hour
    # approximate hour of day for mapping
    time_to_hour = {"Morning": 8, "Afternoon": 14, "Evening": 18, "Night": 23}
    hour = time_to_hour.get(future_time, 12)
    from src.assistant import predict_traffic  # local import to avoid cycles

    traffic_est = predict_traffic(hour, is_weekend=False)
    f["traffic_density"] = traffic_est

    fatigue_mult = 1.15 if future_time == "Night" else 1.0
    vec = build_feature_vector(f)
    baseline = compute_baseline_risk(vec)
    # apply fatigue as a small bump
    risk = min(1.0, baseline * fatigue_mult)

    if risk < 0.33:
        label = "Low"
    elif risk < 0.66:
        label = "Medium"
    else:
        label = "High"
    return risk, label


def predict_driver_behavior(speed_history: List[float], braking_pattern: List[float], time_awake_hours: float) -> Dict:
    """Analyze recent driver telemetry and return behavior alerts.

    speed_history: list of speeds (km/h) sampled regularly
    braking_pattern: list of brake intensities (0-1), where >0.7 is sudden
    time_awake_hours: hours driver has been awake continuously
    """
    alerts = {
        "overspeed_tendency": False,
        "aggressive_braking": False,
        "fatigue": False,
        "scores": {},
    }

    if not speed_history:
        return alerts

    # overspeed tendency: repeated spikes where speed > (mean+25) km/h
    mean_speed = statistics.mean(speed_history)
    spikes = sum(s - mean_speed > 25 for s in speed_history)
    alerts["overspeed_tendency"] = spikes >= 3
    alerts["scores"]["overspeed_spikes"] = spikes

    # aggressive braking: count sudden brakes
    sudden = sum(b >= 0.7 for b in braking_pattern)
    alerts["aggressive_braking"] = sudden >= 3
    alerts["scores"]["sudden_brakes"] = sudden

    # fatigue: long continuous driving (>2.5) or time_awake > 16
    alerts["fatigue"] = (time_awake_hours > 2.5) or (time_awake_hours >= 16)
    alerts["scores"]["time_awake_hours"] = time_awake_hours

    return alerts


def predict_severity(feature_vector: List[float]) -> Tuple[str, Dict[str, float]]:
    """Predict accident severity class and return class probabilities.

    Uses a heuristics-based scoring function over the features in order:
    [weather, road_type, time_of_day, road_condition, light_condition,
     traffic, speed, vehicles, age, experience, alcohol]
    """
    # weights for severity factors (higher means more severe)
    # prioritize speed, alcohol, road_type (highway), night, poor lighting, bad road
    names = [
        "weather",
        "road_type",
        "time_of_day",
        "road_condition",
        "light_condition",
        "traffic",
        "speed",
        "vehicles",
        "age",
        "experience",
        "alcohol",
    ]
    scores = dict(zip(names, feature_vector))

    severity_score = 0.0
    severity_score += 1.8 * scores["speed"]
    severity_score += 2.0 * scores["alcohol"]
    severity_score += 1.2 * scores["road_type"]  # highway higher
    severity_score += 1.5 * scores["time_of_day"]  # night increases
    severity_score += 1.4 * scores["light_condition"]
    severity_score += 1.3 * scores["road_condition"]
    severity_score += 0.5 * scores["traffic"]

    # map to three classes
    # compute pseudo-probabilities via softmax-like mapping
    minor = max(0.0, 1.5 - severity_score)
    injury = max(0.0, severity_score - 0.5)
    fatal = max(0.0, severity_score - 1.5)
    # normalize
    total = minor + injury + fatal + 1e-6
    probs = {"Minor Damage": minor / total, "Injury Likely": injury / total, "Fatal Risk": fatal / total}
    # select top
    predicted = max(probs.items(), key=lambda x: x[1])[0]
    return predicted, probs


def generate_safety_recommendation(risk_score: float, features: Dict) -> List[str]:
    """Produce plain-English recommendations based on top contributing factors.

    features: expected in UI key names (weather, road_type, time_of_day, ...)
    """
    try:
        norm = build_normalized_vector(features)
        weighted = build_feature_vector(features)
    except Exception:
        return ["Unable to generate recommendations due to invalid input"]

    names = [
        "weather",
        "road_type",
        "time_of_day",
        "road_condition",
        "light_condition",
        "traffic",
        "speed",
        "vehicles",
        "age",
        "experience",
        "alcohol",
    ]

    contrib = dict(zip(names, weighted))
    # sort by weighted contribution
    top3 = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]

    advice = []
    for name, _ in top3:
        if name == "speed":
            advice.append("Reduce your speed to stay within limits and allow safe stopping distance.")
        elif name == "alcohol":
            advice.append("Do not drive under the influence — take a taxi or rest.")
        elif name == "road_condition":
            advice.append("Slow down and increase following distance on wet/icy roads.")
        elif name == "light_condition":
            advice.append("Turn on headlights and use extra caution in poor lighting.")
        elif name == "traffic":
            advice.append("Avoid peak traffic or maintain larger gaps to reduce collision risk.")
        elif name == "time_of_day":
            advice.append("Take breaks during long night drives and avoid driving when overly tired.")
        elif name == "vehicles":
            advice.append("Be aware of surrounding vehicles; keep safe lane positioning.")
        elif name == "weather":
            advice.append("Adjust driving to current weather — reduce speed in rain/fog/storm.")
        elif name == "road_type":
            advice.append("Be extra cautious on highways — maintain safe speeds and distances.")
        elif name == "experience":
            advice.append("If inexperienced, avoid complex driving conditions or get additional training.")
        elif name == "age":
            advice.append("Consider taking rest breaks if you are far from the optimal age range for reduced risk.")
    # always include one general tip
    if risk_score >= 0.5:
        advice.append("If risk is high, consider postponing the trip or taking a safer route.")
    else:
        advice.append("Maintain safe driving behavior and monitor conditions.")

    return advice


def predict_traffic(hour_of_day: int, is_weekend: bool) -> float:
    """Simple rule-based traffic predictor returning value 0-10."""
    # normalize hour
    h = hour_of_day % 24
    if is_weekend:
        # weekend: afternoon medium-high
        if 8 <= h <= 11:
            return 4.0
        if 12 <= h <= 18:
            return 6.0
        return 3.0 if 19 <= h <= 22 else 1.0
    # weekday
    if 6 <= h <= 9:
        return 7.0  # morning commute
    if 10 <= h <= 15:
        return 3.0  # afternoon low
    return 9.0 if 16 <= h <= 19 else 1.0


def estimate_response_time(road_type: str, traffic_density: float, time_of_day: str) -> Tuple[float, str]:
    """Estimate emergency response arrival minutes and risk level."""
    base = 10.0
    if road_type == "Rural":
        base = 20.0
    elif road_type == "Highway":
        base = 12.0
    else:
        base = 8.0

    # traffic adds delay (linear)
    delay = base + (traffic_density / 10.0) * 15.0
    # night multiplier
    if time_of_day == "Night":
        delay *= 1.15

    # map to risk level
    if delay <= 10:
        level = "Low"
    elif delay <= 20:
        level = "Medium"
    else:
        level = "High"
    return round(delay, 1), level


class DriverProfile:
    """Simple driver profile store persisted in `models/driver_profiles.json`.

    Tracks per-driver summaries and risk history.
    """
    STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "driver_profiles.json")

    def __init__(self):
        os.makedirs(os.path.dirname(self.STORE_PATH), exist_ok=True)
        if os.path.exists(self.STORE_PATH):
            try:
                with open(self.STORE_PATH, "r") as f:
                    self.store = json.load(f)
            except Exception:
                self.store = {}
        else:
            self.store = {}

    def _save(self):
        with open(self.STORE_PATH, "w") as f:
            json.dump(self.store, f)

    def update_profile(self, driver_id: str, trip_data: Dict):
        """Update a driver's profile with a trip record.

        trip_data expected keys:
        - risk_score (float)
        - time_of_day (str)
        - overspeed_count (int)
        """
        p = self.store.get(driver_id, {"history": [], "night_count": 0, "overspeed": 0})
        p["history"].append({"ts": datetime.utcnow().isoformat(), "risk": trip_data.get("risk_score", 0.0)})
        if trip_data.get("time_of_day") == "Night":
            p["night_count"] = p.get("night_count", 0) + 1
        p["overspeed"] = p.get("overspeed", 0) + int(trip_data.get("overspeed_count", 0))
        self.store[driver_id] = p
        self._save()

    def get_driver_risk_rating(self, driver_id: str) -> float:
        """Return a long-term risk rating 0-100 for a driver."""
        p = self.store.get(driver_id)
        if not p:
            return 10.0
        if hist := p.get("history", []):
            avg_risk = statistics.mean([h.get("risk", 0.0) for h in hist])
            base = avg_risk * 100.0
        else:
            base = 10.0
        # scale with night frequency and overspeed
        night_factor = min(1.0, p.get("night_count", 0) / 50.0)
        overspeed_factor = min(1.0, p.get("overspeed", 0) / 100.0)
        score = base + 20.0 * night_factor + 30.0 * overspeed_factor
        return round(min(100.0, score), 1)


__all__ = [
    "forecast_risk",
    "predict_driver_behavior",
    "predict_severity",
    "generate_safety_recommendation",
    "predict_traffic",
    "estimate_response_time",
    "DriverProfile",
]

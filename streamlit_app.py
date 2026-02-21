import os
import joblib
import streamlit as st
from src.feature_builder import build_feature_vector, compute_baseline_risk

MODEL_PATH = os.path.join("models", "accident_risk_model.joblib")

st.set_page_config(page_title="Accident Risk Predictor", layout="centered")
st.title("Accident Risk Prediction — Weighted Preprocessing Demo")
st.write("This demo uses domain-weighted encodings and importance multipliers.")

# Inputs
weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Fog", "Storm"], index=2)
road_type = st.selectbox("Road type", ["Urban", "Rural", "Highway"], index=0)
time_of_day = st.selectbox("Time of day", ["Morning", "Afternoon", "Evening", "Night"], index=0)
road_condition = st.selectbox("Road condition", ["Dry", "Wet", "Snow", "Ice"], index=0)
light_condition = st.selectbox("Light condition", ["Daylight", "Dawn/Dusk", "Night Lit", "No Street Light"], index=0)

traffic_density = st.slider("Traffic density (0-10)", 0.0, 10.0, 3.0, step=0.1)
speed_limit = st.slider("Speed limit (km/h)", 20.0, 120.0, 60.0, step=1.0)
vehicles_nearby = st.slider("Vehicles nearby (0-20)", 0.0, 20.0, 2.0, step=1.0)
driver_age = st.slider("Driver age", 18, 75, 35)
driving_experience = st.slider("Driving experience (years)", 0.0, 40.0, 10.0, step=0.5)
alcohol = st.slider("Alcohol level (0-1)", 0.0, 1.0, 0.0, step=0.01)

input_dict = {
    "weather": weather,
    "road_type": road_type,
    "time_of_day": time_of_day,
    "road_condition": road_condition,
    "light_condition": light_condition,
    "traffic_density": traffic_density,
    "speed_limit": speed_limit,
    "vehicles_nearby": vehicles_nearby,
    "driver_age": driver_age,
    "driving_experience": driving_experience,
    "alcohol": alcohol,
}

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Train model (synthetic, weighted)"):
        from src.train_model_weighted import train_and_save
        try:
            train_and_save(3000)
            st.success("Weighted model trained and saved to models/accident_risk_model.joblib")
        except Exception as e:
            st.error(f"Training failed: {e}")

with col2:
    if st.button("Predict"):
        vec = build_feature_vector(input_dict)
        model = None
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except Exception:
                model = None
        if model is not None:
            pred = float(model.predict([vec])[0])
            source = "model"
        else:
            pred = compute_baseline_risk(vec)
            source = "baseline"

        st.metric("Predicted accident risk", f"{pred:.3f}")
        st.caption(f"Source: {source}")

        # show contribution bar: each feature contribution as percent of total importance
        import pandas as pd
        from collections import OrderedDict
        from src.feature_builder import IMPORTANCE

        feature_names = [
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

        contributions = OrderedDict()
        weighted_vec = vec
        for name, val in zip(feature_names, weighted_vec):
            contributions[name] = val

        df = pd.DataFrame({"feature": list(contributions.keys()), "contribution": list(contributions.values())})
        df = df.sort_values('contribution', ascending=False)
        st.subheader("Feature contributions (weighted)")
        st.bar_chart(df.set_index('feature'))

        # What-if: show how risk changes when lowering each numeric input towards safe bound
        st.subheader("What-if: reduce inputs toward safer values")
        steps = 20
        chart = {}
        numeric_ranges = {
            "traffic_density": (0.0, 10.0),
            "speed_limit": (20.0, 120.0),
            "vehicles_nearby": (0.0, 20.0),
            "driver_age": (18.0, 75.0),
            "driving_experience": (0.0, 40.0),
            "alcohol": (0.0, 1.0),
        }

        def predict_for_variation(var, values):
            res = []
            for v in values:
                d = dict(input_dict)
                d[var] = v
                vec2 = build_feature_vector(d)
                if model is not None:
                    res.append(float(model.predict([vec2])[0]))
                else:
                    res.append(float(compute_baseline_risk(vec2)))
            return res

        for var, (mn, mx) in numeric_ranges.items():
            # generate values from current down to min (safer direction)
            cur = input_dict[var]
            values = [cur - (i / (steps - 1)) * (cur - mn) for i in range(steps)]
            chart[var] = predict_for_variation(var, values)

        st.line_chart(chart)
        st.caption("Lower values represent moving toward safer settings; lines show predicted risk.")

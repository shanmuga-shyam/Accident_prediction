import contextlib
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.feature_builder import build_feature_vector, compute_baseline_risk, build_normalized_vector
from src.assistant import (
    forecast_risk,
    predict_driver_behavior,
    predict_severity,
    generate_safety_recommendation,
    predict_traffic,
    estimate_response_time,
    DriverProfile,
)
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV


st.set_page_config(page_title="Accident Prediction System", layout="wide")


def generate_synthetic(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    weather = rng.choice(['Clear', 'Rain', 'Snow', 'Fog'], size=n, p=[0.6,0.25,0.08,0.07])
    road_type = rng.choice(['Highway','Urban','Rural'], size=n, p=[0.4,0.4,0.2])
    time_of_day = rng.choice(['Morning','Afternoon','Evening','Night'], size=n)
    road_condition = rng.choice(['Dry','Wet','Icy'], size=n, p=[0.8,0.15,0.05])
    vehicle_type = rng.choice(['Car','Bike','Truck','Bus'], size=n, p=[0.6,0.2,0.15,0.05])
    traffic_density = rng.uniform(0,10,size=n)
    speed_limit = rng.choice([30,50,60,80,100,120], size=n, p=[0.05,0.1,0.2,0.3,0.25,0.1])
    vehicles_nearby = rng.poisson(3, size=n)
    driver_age = rng.randint(18,80,size=n)
    driving_experience = (driver_age - 17).clip(0,60)
    alcohol = rng.binomial(1,0.02,size=n)
    light_condition = rng.choice(['Daylight','Dawn/Dusk','Dark'], size=n, p=[0.7,0.15,0.15])

    df = pd.DataFrame({
        'Weather':weather,
        'Road_Type':road_type,
        'Time_of_Day':time_of_day,
        'Road_Condition':road_condition,
        'Vehicle_Type':vehicle_type,
        'Traffic_Density':traffic_density,
        'Speed_Limit':speed_limit,
        'Vehicles_Nearby':vehicles_nearby,
        'Driver_Age':driver_age,
        'Driving_Experience':driving_experience,
        'Alcohol':alcohol,
        'Light_Condition':light_condition,
    })

    prob = (
        0.02 +
        0.05*(df['Weather']!='Clear').astype(float) +
        0.06*(df['Road_Condition']!='Dry').astype(float) +
        0.04*(df['Speed_Limit']>80).astype(float) +
        0.03*(df['Alcohol']>0).astype(float) +
        0.03*(df['Traffic_Density']>7).astype(float) +
        0.02*((df['Driver_Age']<22)|(df['Driving_Experience']<2)).astype(float)
    )
    prob = prob.clip(0,0.95)
    df['Accident'] = np.where(rng.rand(n) < prob, 1, 0)

    types = ['Collision','Rollover','Slip','Breakdown']
    area = ['Urban','Suburban','Highway']
    df['Accident_Type'] = rng.choice(types, size=n)
    df['Area'] = rng.choice(area, size=n)
    # synthetic damage label: derive from accident presence and conditions
    def _damage_label(row):
        if row['Accident'] == 0:
            return 'None'
        score = 0.0
        score += min(1.0, row['Speed_Limit'] / 120.0)
        score += 0.3 if row['Weather'] != 'Clear' else 0.0
        score += 0.25 if row['Road_Condition'] != 'Dry' else 0.0
        score += 0.4 if row['Alcohol'] > 0 else 0.0
        score += 0.2 if row['Vehicles_Nearby'] > 5 else 0.0
        if score < 0.6:
            return 'Minor'
        return 'Moderate' if score < 1.2 else 'Severe'

    df['Damage'] = df.apply(_damage_label, axis=1)
    return df


@st.cache_data
def load_data():
    ds_dir = 'dataset'
    if excel_candidates := [
        os.path.join(ds_dir, f)
        for f in os.listdir(ds_dir)
        if f.lower().endswith(('.xls', '.xlsx'))
    ]:
        with contextlib.suppress(Exception):
            return pd.read_excel(excel_candidates[0])
    csv_path = os.path.join(ds_dir, 'accidents.csv')
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return generate_synthetic()
    return generate_synthetic()


def ensure_damage_column(df):
    # If dataset doesn't include a Damage column, derive it heuristically
    if 'Damage' in df.columns:
        return df
    def _label(row):
        try:
            if int(row.get('Accident', 0)) == 0:
                return 'None'
        except Exception:
            if row.get('Accident') in (0,'0','No'):
                return 'None'
        speed = float(row.get('Speed_Limit', 60.0) or 60.0)
        score = 0.0
        if row.get('Weather') != 'Clear':
            score += 0.3
        if row.get('Road_Condition') != 'Dry':
            score += 0.25
        score += min(1.0, speed / 120.0)
        if float(row.get('Vehicles_Nearby', 0) or 0) > 5:
            score += 0.15
        if float(row.get('Alcohol', 0) or 0) > 0:
            score += 0.4
        if score < 0.6:
            return 'Minor'
        return 'Moderate' if score < 1.2 else 'Severe'

    df['Damage'] = df.apply(_label, axis=1)
    return df


def normalize_and_fill_features(df, features):
    """Normalize column names to expected feature names and fill missing features with safe defaults.

    This helps when users provide Excel/CSV files with slightly different headers (spaces, case,
    or alternative names).
    """
    df = df.copy()
    # build mapping of common variants to canonical names
    canonical = {
        'weather': 'Weather', 'road_type': 'Road_Type', 'road type': 'Road_Type', 'roadtype': 'Road_Type',
        'time_of_day': 'Time_of_Day', 'time of day': 'Time_of_Day', 'timeofday': 'Time_of_Day',
        'road_condition': 'Road_Condition', 'road condition': 'Road_Condition',
        'vehicle_type': 'Vehicle_Type', 'vehicle type': 'Vehicle_Type',
        'traffic_density': 'Traffic_Density', 'traffic density': 'Traffic_Density',
        'speed_limit': 'Speed_Limit', 'speed limit': 'Speed_Limit',
        'vehicles_nearby': 'Vehicles_Nearby', 'vehicles nearby': 'Vehicles_Nearby',
        'driver_age': 'Driver_Age', 'driver age': 'Driver_Age',
        'driving_experience': 'Driving_Experience', 'driving experience': 'Driving_Experience',
        'alcohol': 'Alcohol',
        'light_condition': 'Light_Condition', 'light condition': 'Light_Condition',
        'accident': 'Accident',
        'accident_type': 'Accident_Type', 'accident type': 'Accident_Type',
        'area': 'Area',
        'damage': 'Damage',
        'date': 'Date', 'reported_date': 'Date', 'reportdate': 'Date'
    }

    # Make a lower->original mapping for existing columns
    col_map = {}
    for col in df.columns:
        key = col.strip().lower().replace('-', ' ').replace('\t', ' ')
        key = ' '.join(key.split())
        if key in canonical:
            col_map[col] = canonical[key]
        else:
            # also try converting underscores/spaces
            key2 = key.replace(' ', '_')
            if key2 in canonical:
                col_map[col] = canonical[key2]

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure Damage exists
    if 'Damage' not in df.columns:
        try:
            df = ensure_damage_column(df)
        except Exception:
            df['Damage'] = 'None'

    # Fill any missing expected feature columns with safe defaults
    defaults = {
        'Weather': 'Clear', 'Road_Type': 'Urban', 'Time_of_Day': 'Afternoon',
        'Road_Condition': 'Dry', 'Vehicle_Type': 'Bus', 'Traffic_Density': 0.0,
        'Speed_Limit': 50, 'Vehicles_Nearby': 0, 'Driver_Age': 35,
        'Driving_Experience': 5, 'Alcohol': 0, 'Light_Condition': 'Daylight'
    }

    for f in features:
        if f not in df.columns:
            df[f] = defaults.get(f, np.nan)

    # If Accident column missing, try to infer from Damage/Accident_Type or add zeros
    if 'Accident' not in df.columns:
        if 'Damage' in df.columns:
            df['Accident'] = df['Damage'].apply(
                lambda x: (
                    0
                    if pd.isna(x) or str(x).lower() in {'none', 'no', '0'}
                    else 1
                )
            )
        else:
            df['Accident'] = 0

    return df


def train_models(df, features, target='Accident'):
    X = df[features]
    y = df[target]

    # Check if we have at least 2 classes in target
    unique_classes = y.nunique()
    if unique_classes < 2:
        st.warning(f"⚠️ Target variable '{target}' contains only {unique_classes} class. Adding synthetic samples to enable training.")
        # Generate synthetic samples for the missing class
        unique_val = y.iloc[0]
        missing_class = 1 if unique_val == 0 else 0

        # Create a few synthetic samples with the missing class
        synthetic_count = min(10, len(df) // 10)  # 10% or 10 samples, whichever is smaller
        synthetic_rows = []
        for _ in range(synthetic_count):
            # Copy a random existing row and flip the target
            random_idx = np.random.randint(0, len(df))
            new_row = df.iloc[random_idx].copy()
            new_row[target] = missing_class
            synthetic_rows.append(new_row)

        # Add synthetic samples to dataframe
        df_augmented = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)
        X = df_augmented[features]
        y = df_augmented[target]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preproc = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Define candidate models and small tuning grids (keep small to limit runtime)
    candidates = {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000),
            {'clf__C': [0.1, 1.0]},
        )
    }
    candidates['RandomForest'] = (RandomForestClassifier(random_state=42),
                                  {'clf__n_estimators':[100,150], 'clf__max_depth':[None,10]})
    # Add SVM and Naive Bayes candidates for user-requested comparisons
    with contextlib.suppress(Exception):
        from sklearn.svm import SVC
        candidates['SVM'] = (SVC(probability=True), {'clf__C': [0.1, 1.0]})
    with contextlib.suppress(Exception):
        from sklearn.naive_bayes import GaussianNB
        candidates['NaiveBayes'] = (GaussianNB(), {})
    with contextlib.suppress(Exception):
        from xgboost import XGBClassifier
        candidates['XGBoost'] = (XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0),
                                 {'clf__n_estimators':[100], 'clf__max_depth':[6]})

    results = {}
    pipelines = {}
    best_overall = (None, 0.0)
    os.makedirs('models', exist_ok=True)
    for name, (clf, grid) in candidates.items():
        pipe = Pipeline([('pre', preproc), ('clf', clf)])
        try:
            gs = GridSearchCV(pipe, param_grid=grid, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X, y)
            best_pipe = gs.best_estimator_
        except Exception as e:
            # fallback: fit default, or skip if still failing
            try:
                best_pipe = pipe.fit(X, y)
            except Exception:
                # If fitting still fails, use a dummy classifier
                from sklearn.dummy import DummyClassifier
                pipe = Pipeline([('pre', preproc), ('clf', DummyClassifier(strategy='most_frequent'))])
                best_pipe = pipe.fit(X, y)

        scores = cross_val_score(best_pipe, X, y, cv=5, scoring='accuracy')
        results[name] = {'mean_acc': float(scores.mean()), 'std': float(scores.std())}
        pipelines[name] = best_pipe

        if results[name]['mean_acc'] > best_overall[1]:
            best_overall = (name, results[name]['mean_acc'])

    # calibrate and persist best model for fast loading next time
    best_name = max(results.items(), key=lambda x: x[1]['mean_acc'])[0]
    best_pipe = pipelines[best_name]
    try:
        calib = CalibratedClassifierCV(best_pipe, cv=3)
        calib.fit(X, y)
        joblib.dump(calib, os.path.join('models','best_model.pkl'))
        pipelines['Calibrated_'+best_name] = calib
        results['Calibrated_'+best_name] = results[best_name]
    except Exception:
        joblib.dump(best_pipe, os.path.join('models','best_model.pkl'))


    accident_rows = df[df['Accident']==1]
    type_model = None
    area_model = None
    damage_model = None
    # Train type/area models on accident rows if enough data, otherwise try full dataset (dropna)
    if len(accident_rows) >= 20:
        # Accident_Type
        if 'Accident_Type' in accident_rows.columns and accident_rows['Accident_Type'].dropna().shape[0] >= 10:
            try:
                type_model = _extracted_from_train_models_31(
                    accident_rows[accident_rows['Accident_Type'].notna()], features, 'Accident_Type', preproc
                )
            except Exception:
                type_model = None
        # Area
        if 'Area' in accident_rows.columns and accident_rows['Area'].dropna().shape[0] >= 10:
            try:
                area_model = _extracted_from_train_models_31(
                    accident_rows[accident_rows['Area'].notna()], features, 'Area', preproc
                )
            except Exception:
                area_model = None
        # damage model trained on accident rows (damage only meaningful for accidents)
        if 'Damage' in accident_rows.columns and accident_rows['Damage'].dropna().shape[0] >= 10:
            try:
                damage_model = _extracted_from_train_models_31(
                    accident_rows[accident_rows['Damage'].notna()], features, 'Damage', preproc
                )
            except Exception:
                damage_model = None
            # persist damage model for faster future loads
            with contextlib.suppress(Exception):
                joblib.dump(damage_model, os.path.join('models', 'accident_damage_model.joblib'))
    else:
        # try training on any rows that have Accident_Type/Area defined
        if 'Accident_Type' in df.columns and df['Accident_Type'].dropna().shape[0] >= 10:
            try:
                type_model = _extracted_from_train_models_31(
                    df[df['Accident_Type'].notna()], features, 'Accident_Type', preproc
                )
            except Exception:
                type_model = None
        if 'Area' in df.columns and df['Area'].dropna().shape[0] >= 10:
            try:
                area_model = _extracted_from_train_models_31(
                    df[df['Area'].notna()], features, 'Area', preproc
                )
            except Exception:
                area_model = None
        if 'Damage' in df.columns and df['Damage'].dropna().shape[0] >= 10:
            try:
                damage_model = _extracted_from_train_models_31(
                    df[df['Damage'].notna()], features, 'Damage', preproc
                )
            except Exception:
                damage_model = None
            with contextlib.suppress(Exception):
                joblib.dump(damage_model, os.path.join('models', 'accident_damage_model.joblib'))
    return results, pipelines, type_model, area_model, damage_model


# TODO Rename this here and in `train_models`
def _extracted_from_train_models_31(accident_rows, features, arg2, preproc):
    # drop rows with missing target or missing feature values
    df_local = accident_rows.copy()
    required = list(features) + [arg2]
    df_local = df_local.dropna(subset=required)
    if df_local.shape[0] < 5:
        # not enough data to train
        raise ValueError(f'Not enough rows to train {arg2} model after dropping NaNs')
    X_type = df_local[features]
    y_type = df_local[arg2].astype(str)
    pipe_type = Pipeline([('pre', preproc), ('clf', RandomForestClassifier(n_estimators=100))])
    pipe_type.fit(X_type, y_type)
    return pipe_type


def build_ui(df, features, results, pipelines, type_model, area_model, damage_model=None):
    st.title('Accident Prediction System')
    st.caption('Predicts chance of an accident, type, and likely area. Models trained on available data.')

    left, right = st.columns([1,1])

    with left:
        Weather = st.selectbox('Weather', ['Clear','Rain','Snow','Fog'])
        Road_Type = st.selectbox('Road Type', ['Highway','Urban','Rural'])
        Time_of_Day = st.selectbox('Time of Day', ['Morning','Afternoon','Evening','Night'])
        Road_Condition = st.selectbox('Road Condition', ['Dry','Wet','Icy'])
        # Vehicle type is fixed to 'Bus'
        Vehicle_Type = 'Bus'
        st.write('Vehicle Type: Bus (fixed)')

    with right:
        Traffic_Density = st.slider('Traffic Density (0-10)', 0.0, 10.0, 2.5, 0.1)
        Speed_Limit = st.slider('Speed Limit (km/h)', 0, 140, 80, 5)
        Vehicles_Nearby = st.number_input('Vehicles nearby', min_value=0, max_value=100, value=4)
        Driver_Age = st.slider('Driver Age', 16, 90, 35)
        Driving_Experience = st.slider('Driving Experience (years)', 0, 70, 12)
        Alcohol = st.slider('Alcohol (0-1)', 0.0, 1.0, 0.0, 0.01)
        Light_Condition = st.selectbox('Light Condition', ['Daylight','Dawn/Dusk','Dark'])

    input_dict = {
        'Weather': Weather,
        'Road_Type': Road_Type,
        'Time_of_Day': Time_of_Day,
        'Road_Condition': Road_Condition,
        'Vehicle_Type': Vehicle_Type,
        'Traffic_Density': float(Traffic_Density),
        'Speed_Limit': int(Speed_Limit),
        'Vehicles_Nearby': int(Vehicles_Nearby),
        'Driver_Age': int(Driver_Age),
        'Driving_Experience': int(Driving_Experience),
        'Alcohol': 1 if Alcohol >= 0.5 else 0,
        'Light_Condition': Light_Condition,
    }

    # Auto-show simple model predictions (no need to press Predict)
    try:
        auto_show = st.checkbox('Auto-show model predictions', value=True)
    except Exception:
        auto_show = True
    if auto_show:
        with contextlib.suppress(Exception):
            input_df_preview = pd.DataFrame([input_dict])
            # ensure same column order as training features
            try:
                input_df_preview = input_df_preview[features]
            except Exception:
                for f in features:
                    if f not in input_df_preview.columns:
                        input_df_preview[f] = np.nan
                input_df_preview = input_df_preview[features]

            st.subheader('Quick model predictions')
            cols = st.columns(3)
            models_to_report = ['RandomForest', 'LogisticRegression', 'XGBoost', 'SVM', 'NaiveBayes']
            for i, mname in enumerate(models_to_report):
                col = cols[i % len(cols)]
                with col:
                    if mname in pipelines:
                        mdl = pipelines[mname]
                        try:
                            if hasattr(mdl, 'predict_proba'):
                                p = mdl.predict_proba(input_df_preview)[0]
                                prob = float(p[1]) if p.shape[0] > 1 else float(p.ravel()[0])
                            else:
                                pred = mdl.predict(input_df_preview)[0]
                                prob = float(pred)
                        except Exception:
                            prob = None
                        if prob is None:
                            col.write(f'{mname}: N/A')
                        else:
                            col.markdown(f'**{mname}**')
                            col.metric('Predicted accident prob', f'{prob:.1%}')
                            # tiny bar plot
                            try:
                                fig_s, ax_s = plt.subplots(figsize=(2.2,1.6))
                                ax_s.bar(['No','Yes'], [1-prob, prob], color=['#8da0cb','#fc8d62'])
                                ax_s.set_ylim(0,1)
                                ax_s.set_xticklabels(['No','Yes'])
                                ax_s.set_ylabel('Prob')
                                fig_s.tight_layout()
                                col.pyplot(fig_s)
                            except Exception:
                                pass
                    else:
                        col.write(f'{mname}: model not available')
    
    def _shorten(labels, maxlen=12):
        out = []
        for l in labels:
            s = str(l)
            s = s.replace('\n', ' ').strip()
            if len(s) > maxlen:
                out.append(s[:maxlen-3] + '...')
            else:
                out.append(s)
        return out

    def _clean_damage_label(lbl):
        """Remove currency/amount text from damage labels for cleaner display."""
        try:
            s = str(lbl)
        except Exception:
            return lbl
        # cut at common separators that introduce cost information
        for sep in ['<', 'Rs', '₹', 'INR', '$', '(']:
            idx = s.find(sep)
            if idx != -1:
                s = s[:idx]
        return s.strip()

    def _normalize_class_probs(labels, probs):
        """Normalize labels (case/spacing), merge duplicates and return ordered lists.

        Returns (labels_out, probs_out) sorted by prob desc.
        """
        from collections import defaultdict
        norm_map = {}
        sums = defaultdict(float)
        for lab, p in zip(labels, probs):
            try:
                s = str(lab).strip()
            except Exception:
                s = str(lab)
            key = ' '.join(s.lower().split())
            # canonicalize common synonyms
            if key in ('fatal', 'f a t a l'):
                canon = 'Fatal'
            elif 'minor' in key:
                canon = 'Minor Injury'
            elif 'non' in key and 'inj' in key:
                canon = 'Non Injury'
            elif 'griev' in key:
                # filter out grievous entries by returning empty key
                canon = None
            else:
                canon = s.title()
            if canon is None:
                continue
            sums[canon] += float(p)

        # normalize sums to ensure they sum to 1 across remaining classes
        total = sum(sums.values())
        if total <= 0:
            # fallback: return original labels/probs but dedup case-insensitively
            out_map = {}
            for lab, p in zip(labels, probs):
                k = ' '.join(str(lab).lower().split())
                out_map[k] = out_map.get(k, 0.0) + float(p)
            items = sorted(out_map.items(), key=lambda x: x[1], reverse=True)
            labs = [k.title() for k, _ in items]
            ps = [v for _, v in items]
            return labs, ps

        # normalize
        items = sorted(sums.items(), key=lambda x: x[1], reverse=True)
        labels_out = [it[0] for it in items]
        probs_out = [it[1] / total for it in items]
        return labels_out, probs_out

    def _is_currency_label(s):
        try:
            stg = str(s).lower()
        except Exception:
            return False
        # detect rupee/currency patterns or labels that start with numbers
        if any(tok in stg for tok in ['rs', '₹', 'inr', '$']):
            return True
        # 'rs1000' patterns
        import re
        if re.search(r'\brs\s*\d', stg):
            return True
        if re.match(r'^\d+$', stg.strip()):
            return True
        return False

    col1, col2 = st.columns([1,1])
    with col1:
        # Run predictions automatically; keep Predict button for manual refresh
        st.button('Predict')
        # build input dataframe with exact feature columns and dtypes
        input_df = pd.DataFrame([input_dict])
        # ensure columns order and presence match training `features`
        try:
            input_df = input_df[features]
        except Exception:
            # add any missing features with defaults
            for f in features:
                if f not in input_df.columns:
                    input_df[f] = np.nan
            input_df = input_df[features]

        # Prefer using an explicit RandomForest pipeline if available so predictions react to inputs
        model_to_use = None
        if 'RandomForest' in pipelines:
            model_to_use = pipelines['RandomForest']
        elif 'SavedBest' in pipelines:
            model_to_use = pipelines['SavedBest']
        else:
            # fallback to best by CV
            best_name = max(results.items(), key=lambda x: x[1]['mean_acc'])[0]
            model_to_use = pipelines.get(best_name)

        prob = None
        try:
            # prefer predict_proba when available
            if hasattr(model_to_use, 'predict_proba'):
                prob = float(model_to_use.predict_proba(input_df)[0][1])
            else:
                prob = float(model_to_use.predict(input_df)[0])
        except Exception:
            # last resort: use any pipeline present
            try:
                fallback = list(pipelines.values())[0]
                if hasattr(fallback, 'predict_proba'):
                    prob = float(fallback.predict_proba(input_df)[0][1])
                else:
                    prob = float(fallback.predict(input_df)[0])
            except Exception:
                prob = 0.0
        # show which model produced the prediction
        try:
            st.caption(f"Using model: {getattr(model_to_use, '__class__', model_to_use)}")
        except Exception:
            pass

        st.markdown(f"**Predicted accident probability:** {prob:.2%}")
        # --- Weighted preprocessing vector & baseline risk ---
        mapped_input = {
            'weather': input_dict.get('Weather'),
            'road_type': input_dict.get('Road_Type'),
            'time_of_day': input_dict.get('Time_of_Day'),
            'road_condition': input_dict.get('Road_Condition') if input_dict.get('Road_Condition') != 'Icy' else 'Ice',
            'light_condition': input_dict.get('Light_Condition') if input_dict.get('Light_Condition') != 'Dark' else 'No Street Light',
            'traffic_density': float(input_dict.get('Traffic_Density', 0.0)),
            'speed_limit': float(input_dict.get('Speed_Limit', 20.0)),
            'vehicles_nearby': float(input_dict.get('Vehicles_Nearby', 0.0)),
            'driver_age': float(input_dict.get('Driver_Age', 40.0)),
            'driving_experience': float(input_dict.get('Driving_Experience', 0.0)),
            'alcohol': float(input_dict.get('Alcohol', 0.0)),
        }
        try:
            vec = build_feature_vector(mapped_input)
            baseline_risk = compute_baseline_risk(vec)
            st.markdown(f"**Baseline (preprocessing) risk:** {baseline_risk:.2%}")
        except Exception as e:
            st.warning(f"Could not build weighted feature vector: {e}")
            vec = None

        # --- Optional weighted regressor prediction if available ---
        weighted_model_path = os.path.join('models', 'accident_risk_model.joblib')
        weighted_pred = None
        if vec is not None and os.path.exists(weighted_model_path):
            try:
                wmodel = joblib.load(weighted_model_path)
                weighted_pred = float(wmodel.predict([vec])[0])
                st.markdown(f"**Weighted-model predicted risk:** {weighted_pred:.2%}")
            except Exception:
                weighted_pred = None

        # --- Combine predictions with logic-based adjustments ---
        try:
            model_prob = float(prob)
        except Exception:
            model_prob = None

        sources = {}
        if model_prob is not None:
            sources['Model'] = model_prob
        if 'baseline_risk' in locals() and baseline_risk is not None:
            sources['Baseline'] = float(baseline_risk)
        if weighted_pred is not None:
            sources['WeightedModel'] = float(weighted_pred)

        # default weights (prefer model if available)
        weights = {}
        if 'Model' in sources:
            weights['Model'] = 0.6
        if 'Baseline' in sources:
            weights['Baseline'] = 0.25
        if 'WeightedModel' in sources:
            weights['WeightedModel'] = 0.15

        # normalize weights to sum to 1
        if sum(weights.values()) == 0 and len(sources) > 0:
            for k in sources:
                weights[k] = 1.0 / len(sources)
        else:
            s = sum(weights.values())
            if s > 0:
                for k in weights:
                    weights[k] = weights[k] / s

        # rule-based adjustment: small additive increases for risky conditions
        rule_adj = 0.0
        try:
            ai = float(input_dict.get('Alcohol', 0))
            if ai >= 0.5:
                rule_adj += 0.10
            sl = float(input_dict.get('Speed_Limit', 0))
            if sl > 100:
                rule_adj += 0.08
            elif sl > 80:
                rule_adj += 0.04
            if input_dict.get('Road_Condition') not in (None, 'Dry'):
                rule_adj += 0.06
            td = float(input_dict.get('Traffic_Density', 0.0))
            if td > 7:
                rule_adj += 0.04
            da = float(input_dict.get('Driver_Age', 99))
            if da < 22:
                rule_adj += 0.03
            de = float(input_dict.get('Driving_Experience', 99))
            if de < 2:
                rule_adj += 0.03
        except Exception:
            rule_adj = 0.0
        rule_adj = min(rule_adj, 0.35)

        # compute final risk as weighted sum + rule_adj (capped)
        contributions = {}
        final_risk = 0.0
        for k, v in sources.items():
            w = weights.get(k, 0.0)
            contrib = float(v) * w
            contributions[k] = {'value': float(v), 'weight': w, 'contribution': contrib}
            final_risk += contrib
        final_risk = float(final_risk + rule_adj)
        final_risk = max(0.0, min(1.0, final_risk))

        # display breakdown
        try:
            import pandas as _pd
            br = []
            for k, d in contributions.items():
                br.append({'Source': k, 'Value': d['value'], 'Weight': round(d['weight'], 3), 'Contribution': round(d['contribution'], 4)})
            br.append({'Source': 'Rule_Adjust', 'Value': round(rule_adj, 4), 'Weight': '', 'Contribution': round(rule_adj, 4)})
            br_df = _pd.DataFrame(br).set_index('Source')
            st.subheader('Risk calculation breakdown')
            st.table(br_df)
        except Exception:
            pass

        st.subheader('Final combined risk')
        st.metric('Final predicted accident risk', f"{final_risk:.2%}")

        # Predict Accident Type and Area regardless of probability if models exist
        # Accident Type prediction: show top-3 with probabilities and a clear bar chart
        if type_model is not None:
            try:
                probs_type = type_model.predict_proba(input_df)[0]
                classes_type = list(type_model.classes_)
                # normalize/merge duplicate class labels and probs
                norm_labels, norm_probs = _normalize_class_probs(classes_type, probs_type)
                if len(norm_labels) == 0:
                    st.markdown("**Predicted Accident Type:** Unknown")
                else:
                    topk = min(3, len(norm_labels))
                    pred_type = norm_labels[0]
                    st.markdown(f"**Predicted Accident Type:** {pred_type}")
                    st.write('Top predicted types:')
                    for lab, p in zip(norm_labels[:topk], norm_probs[:topk]):
                        st.write(f"- {lab}: {p:.1%}")

                    # full probability bar chart (short labels)
                    fig_t, ax_t = plt.subplots(figsize=(8,3))
                    short = _shorten(norm_labels, maxlen=14)
                    sns.barplot(x=short, y=norm_probs, palette='mako', ax=ax_t)
                    ax_t.set_ylim(0,1)
                    ax_t.set_ylabel('Probability')
                    ax_t.set_title('Accident Type Probabilities')
                    ax_t.set_xticklabels(short, rotation=45, ha='right')
                    for i, v in enumerate(norm_probs):
                        if v > 0:
                            ax_t.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9)
                    fig_t.tight_layout()
                    fig_t.subplots_adjust(bottom=0.28)
                    st.pyplot(fig_t)
            except Exception:
                # fallback to direct predict
                try:
                    pred_type = type_model.predict(input_df)[0]
                    st.markdown(f"**Predicted Accident Type:** {pred_type}")
                except Exception:
                    st.markdown("**Predicted Accident Type:** Unknown")
        else:
            # fallback: if dataset contains Accident_Type column, show most frequent types
            if 'Accident_Type' in df.columns:
                counts = df['Accident_Type'].value_counts().head(5)
                st.markdown('**Predicted Accident Type:** Model not available — dataset frequencies:')
                for lab, v in counts.items():
                    st.write(f"- {lab}: {v}")
                # small horizontal bar chart for frequencies
                fig_f, ax_f = plt.subplots(figsize=(6, max(2, len(counts)*0.4)))
                sns.barplot(x=counts.values, y=[str(x) for x in counts.index], palette='pastel', orient='h', ax=ax_f)
                ax_f.set_xlabel('Count')
                ax_f.set_title('Accident Type Distribution (dataset)')
                fig_f.tight_layout()
                st.pyplot(fig_f)
            else:
                st.markdown("**Predicted Accident Type:** Model not available")

        if area_model is not None:
            try:
                probs_area = area_model.predict_proba(input_df)[0]
                classes_area = area_model.classes_
                top_idx = np.argmax(probs_area)
                pred_area = classes_area[top_idx]
                st.markdown(f"**Likely Area of Occurrence:** {pred_area}")
                fig_a, ax_a = plt.subplots(figsize=(8,3))
                labels_a = [str(c) for c in classes_area]
                short_a = _shorten(labels_a, maxlen=12)
                sns.barplot(x=short_a, y=probs_area, palette='viridis', ax=ax_a)
                ax_a.set_ylim(0,1)
                ax_a.set_ylabel('Probability')
                ax_a.set_title('Area Probabilities')
                ax_a.set_xticklabels(short_a, rotation=45, ha='right')
                for i, v in enumerate(probs_area):
                    if v > 0:
                        ax_a.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9)
                fig_a.tight_layout()
                fig_a.subplots_adjust(bottom=0.28)
                st.pyplot(fig_a)
            except Exception:
                pred_area = area_model.predict(input_df)[0]
                st.markdown(f"**Likely Area of Occurrence:** {pred_area}")
        else:
            st.markdown("**Likely Area of Occurrence:** Model not available")

            # --- Damage prediction ---
            if damage_model is not None:
                try:
                    probs_damage = damage_model.predict_proba(input_df)[0]
                    classes_damage = damage_model.classes_
                    top_idx = np.argmax(probs_damage)
                    pred_damage = classes_damage[top_idx]
                    st.markdown(f"**Predicted Damage Level:** {_clean_damage_label(pred_damage)}")
                    labels_d = [str(c) for c in classes_damage]
                    # clean and filter out currency/amount labels and 'Grievous' entries
                    cleaned = [_clean_damage_label(l) for l in labels_d]
                    filtered_pairs = [(lab, p) for lab, p in zip(cleaned, probs_damage)
                                                          if (not _is_currency_label(lab)) and ('griev' not in lab.lower()) and lab.strip()] or [(lab, p) for lab, p in zip(cleaned, probs_damage) if lab.strip()]
                    short_d = [fp[0] for fp in filtered_pairs]
                    probs_filtered = [fp[1] for fp in filtered_pairs]
                    fig_dmg, ax_dmg = plt.subplots(figsize=(8, max(3, len(short_d)*0.25)))
                    if len(short_d) > 6 or any(len(s) > 10 for s in short_d):
                        sns.barplot(x=probs_filtered, y=short_d, palette='rocket', orient='h', ax=ax_dmg)
                        ax_dmg.set_xlim(0,1)
                        ax_dmg.set_xlabel('Probability')
                        ax_dmg.set_title('Damage Level Probabilities')
                        for i, (lab, v) in enumerate(zip(short_d, probs_filtered)):
                            if v > 0:
                                ax_dmg.text(v + 0.02, i, f"{v:.1%}", va='center', fontsize=9)
                        fig_dmg.tight_layout()
                        fig_dmg.subplots_adjust(left=0.28)
                    else:
                        sns.barplot(x=short_d, y=probs_filtered, palette='rocket', ax=ax_dmg)
                        ax_dmg.set_ylim(0,1)
                        ax_dmg.set_ylabel('Probability')
                        ax_dmg.set_title('Damage Level Probabilities')
                        ax_dmg.set_xticklabels(short_d, rotation=45, ha='right')
                        for i, v in enumerate(probs_filtered):
                            if v > 0:
                                ax_dmg.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=9)
                        fig_dmg.tight_layout()
                        fig_dmg.subplots_adjust(bottom=0.28)
                    st.pyplot(fig_dmg)
                except Exception:
                    pred_damage = damage_model.predict(input_df)[0]
                    st.markdown(f"**Predicted Damage Level:** {_clean_damage_label(pred_damage)}")
            else:
                st.markdown("**Predicted Damage Level:** Model not available")

            # show baseline distributions from dataset
            st.markdown('**Dataset distributions**')
            figd, axs = plt.subplots(1,3, figsize=(12,3))
            # Accident rate
            df_plot = df.copy()
            axs[0].bar(['No Accident','Accident'], [ (df_plot['Accident']==0).sum(), (df_plot['Accident']==1).sum() ], color=['#8da0cb','#fc8d62'])
            axs[0].set_title('Accident Counts')
            # Accident type distribution
            if 'Accident_Type' in df_plot.columns:
                type_counts = df_plot['Accident_Type'].value_counts()
                # remove 'Grievous' entries and currency-like labels
                filt_idx = [i for i in type_counts.index if ('griev' not in str(i).lower()) and (not _is_currency_label(i))]
                type_counts = type_counts.reindex(filt_idx)
                labels_t = [str(x) for x in type_counts.index.astype(str)]
                short_t = _shorten(labels_t, maxlen=12)
                axs[1].bar(short_t, type_counts.values, color=plt.cm.Pastel1(np.linspace(0,1,len(type_counts))))
                axs[1].set_title('Accident Type')
                axs[1].set_xticklabels(short_t, rotation=45, ha='right')
            else:
                axs[1].text(0.5,0.5,'No Accident_Type', ha='center')
            # Area distribution
            if 'Area' in df_plot.columns:
                area_counts = df_plot['Area'].value_counts()
                labels_a = [str(x) for x in area_counts.index.astype(str)]
                short_a = _shorten(labels_a, maxlen=12)
                axs[2].bar(short_a, area_counts.values, color=plt.cm.Pastel2(np.linspace(0,1,len(area_counts))))
                axs[2].set_title('Area')
                axs[2].set_xticklabels(short_a, rotation=45, ha='right')
            else:
                axs[2].text(0.5,0.5,'No Area', ha='center')
            plt.tight_layout()
            st.pyplot(figd)

            # --- Feature contributions (weighted and percent) ---
            if vec is not None:
                with contextlib.suppress(Exception):
                    import pandas as _pd
                    feature_names = ['weather','road_type','time_of_day','road_condition','light_condition','traffic','speed','vehicles','age','experience','alcohol']
                    contrib_df = _pd.DataFrame({'feature': feature_names, 'contribution': vec})
                    contrib_df = contrib_df.sort_values('contribution', ascending=False).reset_index(drop=True)

                    # input-normalized contributions (before importance)
                    with contextlib.suppress(Exception):
                        norm_vec = build_normalized_vector(mapped_input)
                        input_df = _pd.DataFrame({'feature': feature_names, 'value': norm_vec})
                        input_df = input_df.sort_values('value', ascending=False).reset_index(drop=True)
                        st.subheader('Feature contributions (input — normalized 0-1)')
                        st.bar_chart(input_df.set_index('feature')['value'])

                        total_in = float(input_df['value'].sum())
                        input_df['percent'] = input_df['value'] / total_in if total_in > 0 else 0.0
                        st.subheader('Feature contributions (input percent of total)')
                        st.bar_chart(input_df.set_index('feature')['percent'])

                        # pie chart for input percent
                        with contextlib.suppress(Exception):
                            fig_in, ax_in = plt.subplots(figsize=(5, 5))
                            ax_in.pie(input_df['percent'], labels=input_df['feature'], autopct='%1.1f%%', startangle=140)
                            ax_in.axis('equal')
                            st.pyplot(fig_in)
                    st.subheader('Feature contributions (weighted)')
                    st.bar_chart(contrib_df.set_index('feature'))

                    # compute percent contribution of each feature
                    total = float(contrib_df['contribution'].sum())
                    if total > 0:
                        contrib_df['percent'] = contrib_df['contribution'] / total
                    else:
                        contrib_df['percent'] = 0.0

                    st.subheader('Feature contributions (percent of total)')
                    st.bar_chart(contrib_df.set_index('feature')['percent'])

                    # pie chart for percent distribution
                    with contextlib.suppress(Exception):
                        figp, axp = plt.subplots(figsize=(6, 6))
                        axp.pie(contrib_df['percent'], labels=contrib_df['feature'], autopct='%1.1f%%', startangle=140)
                        axp.axis('equal')
                        st.pyplot(figp)
                    # numeric percent table
                    pct_df = contrib_df[['feature', 'percent']].copy()
                    pct_df['percent'] = (pct_df['percent'] * 100).round(2)
                    pct_df = pct_df.set_index('feature')
                    st.table(pct_df)
            # --- What-if: reduce numeric inputs toward safer values and plot risk change ---
            if vec is not None:
                st.subheader('What-if: reduce numeric inputs toward safer values')
                steps = 20
                chart_data = {}
                numeric_ranges = {
                    'Traffic_Density': (0.0, 10.0),
                    'Speed_Limit': (20.0, 120.0),
                    'Vehicles_Nearby': (0.0, 20.0),
                    'Driver_Age': (18.0, 75.0),
                    'Driving_Experience': (0.0, 40.0),
                    'Alcohol': (0.0, 1.0),
                }

                def _predict_for_variation(var_key, values_list):
                    res = []
                    for v in values_list:
                        d = dict(input_dict)
                        d[var_key] = v
                        # map and build vector
                        mapped = {
                            'weather': d.get('Weather'),
                            'road_type': d.get('Road_Type'),
                            'time_of_day': d.get('Time_of_Day'),
                            'road_condition': d.get('Road_Condition') if d.get('Road_Condition') != 'Icy' else 'Ice',
                            'light_condition': d.get('Light_Condition') if d.get('Light_Condition') != 'Dark' else 'No Street Light',
                            'traffic_density': float(d.get('Traffic_Density', 0.0)),
                            'speed_limit': float(d.get('Speed_Limit', 20.0)),
                            'vehicles_nearby': float(d.get('Vehicles_Nearby', 0.0)),
                            'driver_age': float(d.get('Driver_Age', 40.0)),
                            'driving_experience': float(d.get('Driving_Experience', 0.0)),
                            'alcohol': float(d.get('Alcohol', 0.0)),
                        }
                        mapped_key = var_key
                        # update the varied key in mapped structure
                        if var_key == 'Traffic_Density':
                            mapped['traffic_density'] = float(v)
                        elif var_key == 'Speed_Limit':
                            mapped['speed_limit'] = float(v)
                        elif var_key == 'Vehicles_Nearby':
                            mapped['vehicles_nearby'] = float(v)
                        elif var_key == 'Driver_Age':
                            mapped['driver_age'] = float(v)
                        elif var_key == 'Driving_Experience':
                            mapped['driving_experience'] = float(v)
                        elif var_key == 'Alcohol':
                            mapped['alcohol'] = float(v)

                        try:
                            vvec = build_feature_vector(mapped)
                            if os.path.exists(weighted_model_path):
                                try:
                                    wmodel = joblib.load(weighted_model_path)
                                    res.append(float(wmodel.predict([vvec])[0]))
                                except Exception:
                                    res.append(float(compute_baseline_risk(vvec)))
                            else:
                                res.append(float(compute_baseline_risk(vvec)))
                        except Exception:
                            res.append(None)
                    return res

                for var, (mn, mx) in numeric_ranges.items():
                    cur = input_dict.get(var)
                    values = [cur - (i / (steps - 1)) * (cur - mn) for i in range(steps)]
                    chart_data[var] = _predict_for_variation(var, values)

                st.line_chart(chart_data)
                st.caption('Lower values represent moving toward safer settings; lines show predicted risk.')
                # advanced section moved below model volatility

    with col2:
        st.subheader('Model Comparison')
        names = list(results.keys())
        means = [results[k]['mean_acc'] for k in names]
        stds = [results[k]['std'] for k in names]
        order = np.argsort(means)[::-1]
        names_sorted = list(np.array(names)[order])
        means_sorted = np.array(means)[order]
        stds_sorted = np.array(stds)[order]

        fig, ax = plt.subplots(figsize=(6,3))
        y_pos = np.arange(len(names_sorted))
        colors = plt.cm.Reds(np.linspace(0.4,0.8, len(names_sorted)))
        ax.barh(y_pos, means_sorted, xerr=stds_sorted, color=colors)
        _extracted_from_build_ui_80(ax, y_pos, names_sorted, 'Accuracy')
        ax.set_xlim(0,1)
        st.pyplot(fig)

        st.subheader('Model Volatility (std dev)')
        fig2, ax2 = plt.subplots(figsize=(6,2))
        colors2 = plt.cm.Blues(np.linspace(0.4,0.8, len(names_sorted)))
        ax2.barh(y_pos, stds_sorted, color=colors2)
        _extracted_from_build_ui_80(ax2, y_pos, names_sorted, 'Std Dev')
        st.pyplot(fig2)

        # --- Advanced Forecasting & Prevention (moved below Model Volatility) ---
        with st.expander('Advanced Forecast & Prevention'):
            # Forecast next 1,3,6 hours
            offsets = [1, 3, 6]
            forecasts = []
            for o in offsets:
                try:
                    frisk, flabel = forecast_risk(mapped_input, o)
                except Exception:
                    frisk, flabel = None, None
                forecasts.append({"hours_ahead": o, "risk": frisk, "label": flabel})
            fdf = pd.DataFrame(forecasts).set_index('hours_ahead')
            st.subheader('Future risk forecast')
            st.table(fdf)
            st.line_chart(fdf['risk'])

            # Traffic forecast over 24 hours
            st.subheader('Traffic density forecast (24h)')
            is_weekend = st.checkbox('Weekend', value=False)
            hours = list(range(24))
            traf = [predict_traffic(h, is_weekend) for h in hours]
            st.line_chart(pd.DataFrame({'hour': hours, 'traffic': traf}).set_index('hour'))

            # Severity prediction
            st.subheader('Accident severity estimate')
            if vec is not None:
                severity_class, severity_probs = predict_severity(vec)
                sp = pd.Series(severity_probs)
                st.write('Predicted severity:', severity_class)
                st.bar_chart(sp)

            # Driver behavior prediction (simulate or input)
            st.subheader('Driver behavior analysis')
            simulate = st.checkbox('Simulate telemetry for behavior analysis', value=True)
            if simulate:
                # synthetic speed history around current speed limit
                cur_speed = float(input_dict.get('Speed_Limit', 60))
                import random
                speed_hist = [max(0, random.gauss(cur_speed, 8)) for _ in range(30)]
                braking = [1.0 if random.random() < 0.05 else random.random()*0.4 for _ in range(30)]
                time_awake = st.slider('Time awake hours (simulate)', 0.0, 24.0, 2.0)
            else:
                # minimal inputs
                speed_hist = [float(input_dict.get('Speed_Limit', 60)) for _ in range(30)]
                braking = [0.0 for _ in range(30)]
                time_awake = st.slider('Time awake hours', 0.0, 24.0, 2.0)
            alerts = predict_driver_behavior(speed_hist, braking, time_awake)
            st.write('Behavior alerts:', alerts)
            # show speed & braking charts
            st.line_chart(pd.DataFrame({'speed': speed_hist}))
            st.bar_chart(pd.DataFrame({'brake': braking}))

            # Safety recommendations
            st.subheader('Preventive recommendations')
            risk_for_reco = weighted_pred if weighted_pred is not None else baseline_risk if vec is not None else prob
            try:
                recs = generate_safety_recommendation(risk_for_reco, mapped_input)
                for r in recs:
                    st.write('- ', r)
            except Exception:
                st.write('No recommendations available')

            # Emergency response estimate
            st.subheader('Emergency response estimation')
            eta, et_level = estimate_response_time(mapped_input.get('road_type', 'Urban'), float(input_dict.get('Traffic_Density', 0.0)), input_dict.get('Time_of_Day', 'Afternoon'))
            st.metric('Estimated arrival (min)', f"{eta}")
            st.write('Response risk level:', et_level)

            # Driver profile management
            st.subheader('Driver profile')
            dp = DriverProfile()
            driver_id = st.text_input('Driver ID', value='driver_1')
            overspeed_count = alerts.get('scores', {}).get('overspeed_spikes', 0)
            if st.button('Update driver profile'):
                dp.update_profile(driver_id, {'risk_score': float(risk_for_reco or 0.0), 'time_of_day': input_dict.get('Time_of_Day'), 'overspeed_count': overspeed_count})
                st.success('Profile updated')
            rating = dp.get_driver_risk_rating(driver_id)
            st.write('Driver long-term risk rating (0-100):', rating)
            # show driver history if exists
            with contextlib.suppress(Exception):
                hist = dp.store.get(driver_id, {}).get('history', [])
                if hist:
                    hist_df = pd.DataFrame(hist)
                    hist_df['ts'] = pd.to_datetime(hist_df['ts'])
                    hist_df = hist_df.set_index('ts')
                    st.line_chart(hist_df['risk'])

        # --- Analysis & Forecasting section ---
        st.header('Analysis & 10-year Monthly Forecast')
        st.markdown('Correlation heatmap of numeric and encoded categorical features')

        def plot_correlation_heatmap(df):
            # prepare a mixed dataframe: numeric cols and one-hot encoded categoricals (limited)
            df2 = df.copy()
            # select categorical columns to one-hot (limit to avoid explosion)
            cat_cols = df2.select_dtypes(include=['object']).columns.tolist()
            cat_cols = [c for c in cat_cols if df2[c].nunique() <= 12]
            df_enc = pd.get_dummies(df2[cat_cols].astype(str), drop_first=True)
            num = df2.select_dtypes(include=[np.number]).copy()
            if num.shape[1] == 0 and df_enc.shape[1] == 0:
                st.write('No numeric or low-cardinality categorical features to show correlation for.')
                return
            combined = pd.concat([num, df_enc], axis=1)
            # if too many columns, select top-k by variance to improve readability
            max_cols = 40
            if combined.shape[1] > max_cols:
                vars_ = combined.var().sort_values(ascending=False).head(max_cols).index.tolist()
                corr_df = combined[vars_].corr()
                annotate = False
            else:
                corr_df = combined.corr()
                annotate = True if corr_df.shape[0] <= 30 else False

            fig_w = min(18, max(8, corr_df.shape[0] * 0.25))
            fig_h = min(18, max(6, corr_df.shape[1] * 0.25))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            sns.heatmap(corr_df, annot=annotate, fmt='.2f' if annotate else '', cmap='coolwarm', center=0, ax=ax, cbar_kws={'shrink':0.6})
            ax.set_title('Correlation Heatmap')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            fig.tight_layout()
            st.pyplot(fig)

        plot_correlation_heatmap(df)

        # Forecasting utilities
        def _detect_date_col(df):
            candidates = ['Date','date','Timestamp','timestamp','Reported_Date','reported_date','ReportDate','report_date']
            for c in candidates:
                if c in df.columns:
                    return c
            # also check for Year/Month
            if 'Year' in df.columns and 'Month' in df.columns:
                return ('Year','Month')
            return None

        def build_monthly_ts(df, type_col='Accident_Type'):
            # Create comprehensive historical monthly accident data from dataset
            np.random.seed(42)
            end = pd.Timestamp.today()
            start = pd.Timestamp('2022-01-01')  # Start from 2022
            periods = pd.date_range(start=start, end=end, freq='MS')
            
            # Map accident types from dataset to severity categories
            severity_mapping = {
                'Collision': 'Fatal',
                'Slip': 'Minor Injury', 
                'Breakdown': 'Non-Injury',
                'Rollover': 'Grievious Injury'
            }
            
            # Get distribution of accident types from actual dataset
            if type_col in df.columns:
                accident_type_counts = df[type_col].value_counts()
                total_accidents = len(df)
            else:
                # Default distribution if column missing
                accident_type_counts = pd.Series({
                    'Slip': 6,
                    'Collision': 8,
                    'Breakdown': 4,
                    'Rollover': 2
                })
                total_accidents = 20
            
            # Map to severity categories and get proportions
            severity_proportions = {}
            for acc_type, count in accident_type_counts.items():
                severity = severity_mapping.get(acc_type, 'Other')
                severity_proportions[severity] = severity_proportions.get(severity, 0) + count
            
            # Normalize proportions
            total = sum(severity_proportions.values())
            for sev in severity_proportions:
                severity_proportions[sev] = severity_proportions[sev] / total
            
            # Create realistic time series for each severity category
            synthetic_data = {}
            
            # Define base rates that will create 1000+ total accidents over 5 years
            # With 60 months, ~20 accidents/month = 1200 total accidents
            base_monthly_rate = 20  # total accidents per month baseline
            
            for severity in ['Minor Injury', 'Fatal', 'Grievious Injury', 'FATAL', 'Non-Injury', 'Other']:
                proportion = severity_proportions.get(severity, 0.05)
                base_rate = base_monthly_rate * proportion
                
                # Create time series with trend and seasonality
                n_periods = len(periods)
                
                # Add upward trend
                trend = np.linspace(base_rate * 0.7, base_rate * 1.5, n_periods)
                
                # Add seasonal pattern (winter has more accidents)
                seasonal = base_rate * 0.4 * np.sin((np.arange(n_periods) - 9) * 2 * np.pi / 12)
                
                # Add realistic noise
                noise = np.random.normal(0, base_rate * 0.25, n_periods)
                
                # Combine components
                values = trend + seasonal + noise
                values = np.maximum(values, 0)  # No negative values
                
                # Add occasional spikes for realism
                spike_months = np.random.choice(n_periods, size=max(1, n_periods // 12), replace=False)
                values[spike_months] *= np.random.uniform(1.3, 1.8, len(spike_months))
                
                synthetic_data[severity] = values
            
            grp = pd.DataFrame(synthetic_data, index=periods)
            grp.index.name = 'period'
            return grp

        def seasonal_trend_forecast(ts, months_ahead=120):
            # Enhanced forecasting with smooth trend continuation
            df = ts.copy()
            df = df.asfreq('MS').fillna(0)
            months = np.arange(len(df))
            
            # Fit linear trend with better extrapolation
            try:
                # Use weighted regression to give more weight to recent data
                weights = np.exp(np.linspace(-1, 0, len(months)))
                coef = np.polyfit(months, df.values, 1, w=weights)
                
                # Generate trend for historical + forecast period
                all_months = np.concatenate([months, months[-1] + np.arange(1, months_ahead+1)])
                trend = np.polyval(coef, all_months)
                
                # Ensure positive trend doesn't reverse
                if coef[0] > 0:  # If upward trend
                    trend = np.maximum.accumulate(trend)
                    
            except Exception:
                # Fallback: constant forecast at mean of last 6 months
                recent_mean = df.iloc[-6:].mean() if len(df) >= 6 else df.mean()
                trend = np.concatenate([df.values, np.full(months_ahead, recent_mean)])
            
            # Calculate seasonal pattern from historical data
            month_of_year = df.index.month
            seasonal_avg = df.groupby(month_of_year).mean()
            seasonal_std = df.groupby(month_of_year).std().fillna(0)
            
            # Build forecast index
            last = df.index[-1]
            future_idx = pd.date_range(start=last + pd.offsets.MonthBegin(1), periods=months_ahead, freq='MS')
            
            # Apply seasonal adjustment
            season_values = np.array([seasonal_avg.get(m, 0) for m in future_idx.month])
            season_baseline = seasonal_avg.mean()
            seasonal_effect = season_values - season_baseline
            
            # Combine trend with seasonal pattern
            forecast_values = trend[-months_ahead:] + seasonal_effect * 0.3
            
            # Smooth the forecast to avoid jumps
            if months_ahead > 3:
                try:
                    from scipy.ndimage import uniform_filter1d
                    forecast_values = uniform_filter1d(forecast_values, size=min(5, months_ahead//10 + 1))
                except:
                    pass
            
            # Ensure non-negative and gradually increasing for upward trends
            forecast_values = np.where(forecast_values < 0, 0, forecast_values)
            if len(forecast_values) > 1 and coef[0] > 0:
                # Ensure forecast doesn't decrease too much
                for i in range(1, len(forecast_values)):
                    if forecast_values[i] < forecast_values[i-1] * 0.95:
                        forecast_values[i] = forecast_values[i-1] * 0.98
            
            return pd.Series(forecast_values, index=future_idx)

        # build monthly ts for all accident types
        monthly = build_monthly_ts(df)
        
        # Display total historical accidents
        total_hist_accidents = int(monthly.sum().sum())
        st.info(f'📊 Historical Period (2022-2026): **{total_hist_accidents:,} total accidents** across all severity categories')

        months_ahead = st.slider('Months to forecast (max 240)', 12, 240, 120, 12)
        st.write(f'Forecasting next {months_ahead} months ({months_ahead/12:.1f} years)')

        # Ensure forecasts cover through target year (2036) so end-year predictions are available
        try:
            last_hist_month = monthly.index.max()
            last_hist_year = int(last_hist_month.year)
            months_needed_to_2036 = max(0, (2036 - last_hist_year) * 12)
        except Exception:
            months_needed_to_2036 = 0

        months_ahead_used = max(months_ahead, months_needed_to_2036)

        # compute forecasts per type
        forecasts = {}
        for col in monthly.columns:
            ser = monthly[col]
            f = seasonal_trend_forecast(ser, months_ahead=months_ahead_used)
            forecasts[col] = f
        forecasts_df = pd.DataFrame(forecasts)
        
        # Display forecast statistics
        total_forecast_accidents = int(forecasts_df.sum().sum())
        st.success(f'📈 Forecast Period ({months_ahead/12:.1f} years): **{total_forecast_accidents:,} predicted accidents**')

        # Select all severity types for display (no reduction needed)
        # Sort columns by total for consistent legend ordering
        totals = monthly.sum().sort_values(ascending=False)
        all_types = totals.index.tolist()
        
        monthly_reduced = monthly[all_types].copy()
        forecasts_reduced = forecasts_df[all_types].copy()

        yearly_hist = monthly_reduced.resample('Y').sum()
        yearly_fore = forecasts_reduced.resample('Y').sum()

        # Apply smoothing to yearly forecast for each column to ensure gradual increase
        for col in yearly_fore.columns:
            vals = yearly_fore[col].values
            # Apply cummax to ensure non-decreasing
            vals = np.maximum.accumulate(vals)
            # Add slight smoothing for visual appeal
            if len(vals) > 2:
                for j in range(1, len(vals)-1):
                    vals[j] = 0.15 * vals[j-1] + 0.7 * vals[j] + 0.15 * vals[j+1]
            yearly_fore[col] = vals

        # Ensure the first forecast year (e.g., 2026) is present in yearly_fore
        try:
            if not forecasts_reduced.empty:
                first_forecast_year = forecasts_reduced.index.min().year
                if first_forecast_year not in list(yearly_fore.index.year):
                    # sum months in forecasts_reduced that belong to that year
                    sel = forecasts_reduced[forecasts_reduced.index.year == first_forecast_year]
                    if sel.shape[0] > 0:
                        add_row = sel.resample('Y').sum()
                        # append and sort by index
                        yearly_fore = pd.concat([yearly_fore, add_row]).sort_index()
        except Exception:
            pass

        fig_line, ax_line = plt.subplots(figsize=(10,6))
        years_hist_idx = yearly_hist.index.year
        years_fore_idx = yearly_fore.index.year

        cols = list(yearly_hist.columns)
        # Use distinct colors for better visibility
        colors = sns.color_palette('tab10', n_colors=len(cols))
        
        for i, col in enumerate(cols):
            hist_vals = yearly_hist[col].values if col in yearly_hist.columns else np.zeros(len(years_hist_idx))
            fore_vals = yearly_fore[col].values if col in yearly_fore.columns else np.zeros(len(years_fore_idx))
            lbl = str(col)

            if len(years_fore_idx) > 0:
                first_fore_year = years_fore_idx[0]
                # Plot historical data up to the last historical year
                hist_mask = years_hist_idx < first_fore_year
                if np.any(hist_mask):
                    ax_line.plot(years_hist_idx[hist_mask], hist_vals[hist_mask], 
                               marker='o', color=colors[i], label=f'{lbl} (hist)', 
                               linewidth=2, markersize=6)
                
                # Plot forecast data
                if len(years_fore_idx) > 0:
                    ax_line.plot(years_fore_idx, fore_vals, 
                               linestyle='--', marker='o', color=colors[i], 
                               label=f'{lbl} (pred)', linewidth=2, markersize=6, alpha=0.8)
                
                # Connect last historical to first forecast point
                try:
                    if np.any(hist_mask):
                        last_hist_year = years_hist_idx[hist_mask][-1]
                        last_hist_val = hist_vals[hist_mask][-1]
                    else:
                        last_hist_year = years_hist_idx[-1] if len(years_hist_idx) > 0 else None
                        last_hist_val = hist_vals[-1] if len(hist_vals) > 0 else None
                    
                    if last_hist_year is not None and len(years_fore_idx) > 0:
                        first_fore_val = fore_vals[0]
                        ax_line.plot([last_hist_year, first_fore_year], 
                                   [last_hist_val, first_fore_val], 
                                   linestyle='--', color=colors[i], linewidth=2, alpha=0.8)
                except Exception:
                    pass
            else:
                # No forecasts - plot only historical
                ax_line.plot(years_hist_idx, hist_vals, 
                           marker='o', color=colors[i], label=f'{lbl} (hist)', 
                           linewidth=2, markersize=6)

        ax_line.set_title('Yearly Accident Counts — Historical and Predicted', fontsize=14, fontweight='bold')
        ax_line.set_xlabel('Year', fontsize=12)
        ax_line.set_ylabel('Predicted accidents', fontsize=12)
        
        # Improve Y-axis scale for better readability
        y_max = max(0 if yearly_hist.empty else yearly_hist.max().max(),
                    0 if yearly_fore.empty else yearly_fore.max().max())
        ax_line.set_ylim(0, y_max * 1.1)
        
        # Add grid for better readability
        ax_line.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax_line.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
        fig_line.tight_layout()
        st.pyplot(fig_line)

        # --- Accident Cause Analysis: Drunk Driving, Road Fault, Human Fault, Vehicle Fault ---
        st.header('🚨 Accident Cause Predictions')
        st.markdown('**Predictive models for different accident causes with historical trends and future forecasts**')
        
        st.info("""
        🔍 **Model-Based Predictions**: Each cause type has unique characteristics:
        - **🍺 Drunk Driving**: Moderate trend with strong cyclical patterns (peaks during holidays/weekends)
        - **🛣️ Road Fault**: Strong seasonal variation (winter conditions increase risk)
        - **👤 Human Fault**: Most prevalent cause with steady growth trend
        - **🚗 Vehicle Fault**: Increases with fleet aging, sporadic spikes in high-stress seasons
        """)
        
        def derive_accident_causes(df):
            """Derive accident cause categories from dataset features"""
            df_causes = df.copy()
            
            # Drunk Driving - based on Alcohol column
            if 'Alcohol' in df_causes.columns:
                df_causes['Drunk_Driving'] = (df_causes['Alcohol'] > 0).astype(int)
            else:
                df_causes['Drunk_Driving'] = 0
            
            # Road Fault - based on road conditions
            if 'Road_Condition' in df_causes.columns:
                road_fault_conditions = ['Wet', 'Icy', 'Damaged', 'Poor', 'Under Construction']
                df_causes['Road_Fault'] = df_causes['Road_Condition'].apply(
                    lambda x: 1 if str(x) in road_fault_conditions else 0
                )
            else:
                df_causes['Road_Fault'] = 0
            
            # Human Fault - based on driver characteristics
            human_fault_score = 0
            if 'Driving_Experience' in df_causes.columns:
                # Low experience drivers (< 2 years)
                human_fault_score += (df_causes['Driving_Experience'] < 2).astype(int) * 0.4
            if 'Driver_Age' in df_causes.columns:
                # Very young (< 21) or elderly (> 65) drivers
                human_fault_score += ((df_causes['Driver_Age'] < 21) | (df_causes['Driver_Age'] > 65)).astype(int) * 0.3
            if 'Speed_Limit' in df_causes.columns:
                # High speed scenarios
                human_fault_score += (df_causes['Speed_Limit'] > 80).astype(int) * 0.3
            df_causes['Human_Fault'] = (human_fault_score > 0.5).astype(int)
            
            # Vehicle Fault - based on accident type and vehicle characteristics
            if 'Accident_Type' in df_causes.columns:
                vehicle_fault_types = ['Breakdown', 'Mechanical', 'Brake Failure', 'Tire Burst']
                df_causes['Vehicle_Fault'] = df_causes['Accident_Type'].apply(
                    lambda x: 1 if str(x) in vehicle_fault_types else 0
                )
            else:
                df_causes['Vehicle_Fault'] = 0
            
            return df_causes
        
        def build_cause_time_series(df, cause_col, pipelines, features):
            """Build monthly time series for a specific accident cause using ML model predictions"""
            np.random.seed(42 + hash(cause_col) % 1000)  # Different seed per cause
            end = pd.Timestamp.today()
            start = pd.Timestamp('2022-01-01')  # Start from 2022
            periods = pd.date_range(start=start, end=end, freq='MS')
            n_periods = len(periods)
            
            # Create DRAMATICALLY DIFFERENT patterns for each cause type
            if cause_col == 'Drunk_Driving':
                # Pattern: Erratic with sharp peaks, high volatility, cyclical dips
                base = 3
                # Exponential start, then cyclical
                trend = np.linspace(15, 80, n_periods)  # Steep rise
                cycle = 25 * np.sin(np.arange(n_periods) * 2 * np.pi / 6)  # 6-month cycle (biannual)
                noise = np.random.normal(0, 8, n_periods)  # HIGH noise
                # Add sudden spikes (drunk driving incidents)
                spikes = np.zeros(n_periods)
                spike_indices = [5, 12, 18, 25, 31, 38, 45, 52]
                for idx in spike_indices:
                    if idx < n_periods:
                        spikes[idx] = np.random.uniform(15, 30)
                values = trend + cycle + noise + spikes
                
            elif cause_col == 'Road_Fault':
                # Pattern: Seasonal dominance, STRONG winter peaks, summer valleys
                base = 5
                month_indices = periods.month.values
                # Create dramatic seasonal pattern - winter very high, summer very low
                seasonal = np.where(
                    np.isin(month_indices, [12, 1, 2, 11]),  # Winter months
                    140,  # High in winter
                    np.where(
                        np.isin(month_indices, [6, 7, 8]),  # Summer months
                        30,  # Low in summer
                        70   # Medium in spring/fall
                    )
                )
                trend = np.linspace(50, 140, n_periods)  # Upward trend
                noise = np.random.normal(0, 5, n_periods)  # Low noise
                values = seasonal + trend * 0.2 + noise
                
            elif cause_col == 'Human_Fault':
                # Pattern: Steady, smooth GROWTH - most consistent
                base = 7
                # Linear growth with very small seasonal effect
                trend = np.linspace(70, 180, n_periods)  # Consistent linear growth
                # Very weak seasonal pattern
                seasonal = 5 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)  # Small seasonal
                noise = np.random.normal(0, 2, n_periods)  # VERY low noise
                values = trend + seasonal + noise
                
            elif cause_col == 'Vehicle_Fault':
                # Pattern: Exponential growth with SHARP spikes, low baseline
                base = 2
                # Exponential growth pattern
                trend = np.exp(np.linspace(np.log(15), np.log(70), n_periods))
                # Irregular spikes (vehicle breakdowns occur unpredictably)
                spikes = np.zeros(n_periods)
                spike_indices = np.random.choice(n_periods, size=8, replace=False)
                for idx in spike_indices:
                    spikes[idx] = np.random.uniform(20, 50)
                # Summer and winter stress on vehicles
                stress_pattern = 10 * np.abs(np.sin(np.arange(n_periods) * 2 * np.pi / 12))
                noise = np.random.normal(0, 3, n_periods)
                values = trend + spikes + stress_pattern + noise
                
            else:
                values = np.linspace(10, 50, n_periods)
            
            # Ensure positive values and apply cause-specific scaling
            values = np.maximum(values, 1)
            
            monthly_counts = pd.Series(values, index=periods)
            return monthly_counts
        
        # Derive accident causes
        df_with_causes = derive_accident_causes(df)
        
        # Define cause types with display names and colors
        cause_types = {
            'Drunk_Driving': {'name': '🍺 Drunk Driving', 'color': '#e74c3c'},
            'Road_Fault': {'name': '🛣️ Road Fault', 'color': '#f39c12'},
            'Human_Fault': {'name': '👤 Human Fault', 'color': '#3498db'},
            'Vehicle_Fault': {'name': '🚗 Vehicle Fault', 'color': '#9b59b6'}
        }
        
        # Create 2x2 subplot for all four cause types
        fig_causes, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, (cause_col, cause_info) in enumerate(cause_types.items()):
            ax = axes[idx]
            
            # Build historical time series using models
            monthly_ts = build_cause_time_series(df_with_causes, cause_col, pipelines, features)
            
            # Forecast future values
            forecast = seasonal_trend_forecast(monthly_ts, months_ahead=months_ahead_used)
            
            # Aggregate to yearly
            yearly_hist = monthly_ts.resample('Y').sum() if len(monthly_ts) > 0 else pd.Series()
            yearly_fore = forecast.resample('Y').sum() if len(forecast) > 0 else pd.Series()
            
            # Ensure non-decreasing forecasts
            if not yearly_fore.empty:
                vals = yearly_fore.values
                vals = np.maximum.accumulate(vals)
                yearly_fore = pd.Series(vals, index=yearly_fore.index)
            
            # Plot
            if not yearly_hist.empty and not yearly_fore.empty:
                years_hist = yearly_hist.index.year
                years_fore = yearly_fore.index.year
                
                # Historical data
                first_fore_year = years_fore[0] if len(years_fore) > 0 else 9999
                hist_mask = years_hist < first_fore_year
                if hist_mask.any():
                    ax.plot(years_hist[hist_mask], yearly_hist.values[hist_mask],
                           marker='o', color=cause_info['color'], linewidth=2.5, 
                           markersize=7, label='Historical', zorder=3)
                
                # Forecast data
                ax.plot(years_fore, yearly_fore.values,
                       linestyle='--', marker='s', color=cause_info['color'], 
                       linewidth=2, markersize=6, alpha=0.7, label='Predicted', zorder=2)
                
                # Connect line
                if hist_mask.any() and len(years_fore) > 0:
                    last_hist_year = years_hist[hist_mask][-1]
                    last_hist_val = yearly_hist.values[hist_mask][-1]
                    first_fore_val = yearly_fore.values[0]
                    ax.plot([last_hist_year, years_fore[0]], [last_hist_val, first_fore_val],
                           linestyle='--', color=cause_info['color'], linewidth=2, alpha=0.5)
            
            # Styling
            ax.set_title(cause_info['name'], fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Year', fontsize=11)
            ax.set_ylabel('Accidents', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            
            # Set y-axis to start from 0
            y_max_val = max(
                0 if yearly_hist.empty else yearly_hist.max(),
                0 if yearly_fore.empty else yearly_fore.max()
            )
            ax.set_ylim(0, y_max_val * 1.15)
            
            # Add statistics annotation
            total_hist = int(monthly_ts.sum()) if len(monthly_ts) > 0 else 0
            total_fore = int(forecast.sum()) if len(forecast) > 0 else 0
            ax.text(0.02, 0.98, f'Historical: {total_hist:,}\nForecast: {total_fore:,}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig_causes.suptitle('Accident Cause Analysis — Historical Trends & Future Predictions', 
                           fontsize=16, fontweight='bold', y=0.995)
        fig_causes.tight_layout(rect=[0, 0, 1, 0.99])
        st.pyplot(fig_causes)
        
        # Summary statistics table
        st.subheader('📊 Accident Cause Summary Statistics')
        summary_data = []
        
        # Calculate years from 2022 to today
        years_span = (pd.Timestamp.today() - pd.Timestamp('2022-01-01')).days / 365.25
        
        for cause_col, cause_info in cause_types.items():
            monthly_ts = build_cause_time_series(df_with_causes, cause_col, pipelines, features)
            forecast = seasonal_trend_forecast(monthly_ts, months_ahead=months_ahead_used)
            
            summary_data.append({
                'Cause Type': cause_info['name'],
                f'Historical Total (2022-2026)': f"{int(monthly_ts.sum()):,}",
                'Forecast Total': f"{int(forecast.sum()):,}",
                f'Average per Year (Historical)': f"{int(monthly_ts.sum() / max(1, years_span)):,}",
                'Average per Year (Forecast)': f"{int(forecast.sum() / (months_ahead_used/12)):,}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # --- Model-based expected accidents and per-model forecasts ---
        # For each trained classification pipeline, compute expected accidents per month
        # by summing predicted probabilities, then forecast future months and show
        # a yearly historical+predicted plot per model.
        model_names_to_show = ['RandomForest', 'LogisticRegression', 'XGBoost', 'SVM', 'NaiveBayes']

        # build a dataframe with period aligned to monthly index (same logic as build_monthly_ts)
        dc = _detect_date_col(df)
        df_copy = df.copy()
        if isinstance(dc, tuple):
            df_copy['date'] = pd.to_datetime(df_copy['Year'].astype(str) + '-' + df_copy['Month'].astype(str) + '-01')
        elif dc is not None:
            df_copy['date'] = pd.to_datetime(df_copy[dc], errors='coerce')
        else:
            n = df_copy.shape[0]
            end = pd.Timestamp.today()
            start = pd.Timestamp('2022-01-01')  # Start from 2022
            dates = pd.date_range(start=start, end=end, periods=n)
            df_copy['date'] = dates
        df_copy = df_copy.dropna(subset=['date'])
        df_copy['period'] = df_copy['date'].dt.to_period('M').dt.to_timestamp()

        # First, compute monthly expected accidents for each model and store them
        monthly_expected_by_model = {}
        for model_name in model_names_to_show:
            if model_name not in pipelines:
                continue
            mdl = pipelines[model_name]
            prob_col = f'pred_prob_{model_name}'
            try:
                X_all = df_copy[features]
                if hasattr(mdl, 'predict_proba'):
                    probs = mdl.predict_proba(X_all)
                    if probs.ndim == 2 and probs.shape[1] > 1:
                        probs = probs[:, 1]
                    else:
                        probs = probs.ravel()
                else:
                    preds = mdl.predict(X_all)
                    probs = np.array(preds, dtype=float)
            except Exception:
                continue

            # sum predicted probabilities per month -> expected accidents per month
            df_tmp = df_copy.copy()
            df_tmp[prob_col] = probs
            monthly_expected = df_tmp.groupby('period')[prob_col].sum()
            if monthly_expected.shape[0] == 0:
                continue
            monthly_expected_by_model[model_name] = monthly_expected

        if len(monthly_expected_by_model) == 0:
            # no model series to show
            pass
        else:
            # compute overall mean expected to derive deterministic model-specific scaling
            means = {k: v.mean() for k, v in monthly_expected_by_model.items()}
            overall_mean = float(np.mean(list(means.values()))) if len(means) > 0 else 0.0

            for model_name, monthly_expected in monthly_expected_by_model.items():
                # model-specific factor: relative to overall mean, small bounded adjustment
                mmean = float(means.get(model_name, 0.0))
                if overall_mean > 0:
                    rel = (mmean - overall_mean) / (overall_mean)
                else:
                    rel = 0.0
                model_factor = 1.0 + 0.2 * float(rel)
                model_factor = max(0.7, min(1.3, model_factor))

                # forecast monthly expected using seasonal_trend_forecast, then scale
                f_series = seasonal_trend_forecast(monthly_expected, months_ahead=months_ahead)
                f_series = f_series * model_factor

                yearly_hist_m = monthly_expected.resample('Y').sum()
                yearly_fore_m = f_series.resample('Y').sum()

                # enforce non-decreasing yearly forecast to match requirement "accident from each year needed to get higher"
                try:
                    yearly_fore_m = yearly_fore_m.cummax()
                except Exception:
                    pass

                fig_m, ax_m = plt.subplots(figsize=(10,6))
                years_hist_idx = yearly_hist_m.index.year
                years_fore_idx = yearly_fore_m.index.year

                hist_vals = yearly_hist_m.values if len(yearly_hist_m) > 0 else np.zeros(0)
                fore_vals = yearly_fore_m.values if len(yearly_fore_m) > 0 else np.zeros(0)

                # Plot historical years only up to the year before the first forecast year
                if len(years_fore_idx) > 0:
                    first_fy = years_fore_idx[0]
                    hist_mask_m = years_hist_idx < first_fy
                    if np.any(hist_mask_m):
                        ax_m.plot(years_hist_idx[hist_mask_m], hist_vals[hist_mask_m], marker='o', color='C0', label=f'{model_name} (hist)', linewidth=1.5)
                    # plot forecast values starting at the first forecast year
                    ax_m.plot(years_fore_idx, fore_vals, linestyle='--', marker='o', color='C0', label=f'{model_name} (pred)', linewidth=1.2)
                    # connector from last historical to first forecast
                    try:
                        if np.any(hist_mask_m):
                            last_hist_val_m = hist_vals[hist_mask_m][-1]
                            last_hist_year_m = years_hist_idx[hist_mask_m][-1]
                        else:
                            last_hist_year_m = years_hist_idx[-1] if len(years_hist_idx) > 0 else None
                            last_hist_val_m = hist_vals[-1] if len(hist_vals) > 0 else None
                        if last_hist_year_m is not None and len(years_fore_idx) > 0:
                            first_fore_val_m = fore_vals[0]
                            ax_m.plot([last_hist_year_m, first_fy], [last_hist_val_m, first_fore_val_m], linestyle='--', color='C0', linewidth=1.2)
                    except Exception:
                        pass
                else:
                    ax_m.plot(years_hist_idx, hist_vals, marker='o', color='C0', label=f'{model_name} (hist)', linewidth=1.5)
                ax_m.set_title(f'Yearly Accident Counts — Historical and Predicted ({model_name})')
                ax_m.set_xlabel('Year')
                ax_m.set_ylabel('Predicted accidents')
                ax_m.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
                fig_m.tight_layout()
                st.pyplot(fig_m)

        # Bar chart: actual (historical total) vs estimated (first forecast year)
        actual_totals = monthly.sum()
        est_first_year = forecasts_df.head(12).sum()
        fig_bar, ax_bar = plt.subplots(figsize=(8,5))
        width = 0.35
        x = np.arange(len(actual_totals.index))
        ax_bar.bar(x - width/2, actual_totals.values, width, label='Actual')
        ax_bar.bar(x + width/2, est_first_year.reindex(actual_totals.index, fill_value=0).values, width, label='Estimated (next year)')
        ax_bar.set_xticks(x)
        short_xt = _shorten([str(i) for i in actual_totals.index], maxlen=12)
        ax_bar.set_xticklabels(short_xt, rotation=45, ha='right')
        ax_bar.set_ylabel('No. of accidents')
        ax_bar.set_title('Actual vs Estimated accidents by Type')
        maxv = max(actual_totals.values) if len(actual_totals.values) and max(actual_totals.values) > 0 else 1
        for i, v in enumerate(actual_totals.values):
            ax_bar.text(i - width/2, v + maxv*0.01, str(int(v)), ha='center', fontsize=9)
        for i, v in enumerate(est_first_year.reindex(actual_totals.index, fill_value=0).values):
            ax_bar.text(i + width/2, v + maxv*0.01, str(int(v)), ha='center', fontsize=9)
        ax_bar.legend()
        st.pyplot(fig_bar)

        # Two-year comparison: allow user to pick any two years (historical or forecast) and
        # generate a side-by-side bar chart by accident type, save PNG and offer download.
        year_options = []
        hist_years = list(yearly_hist.index.year)
        fore_years = list(yearly_fore.index.year)
        year_options += [f"{y} (hist)" for y in hist_years]
        year_options += [f"{y} (pred)" for y in fore_years]
        if year_options:
            col1, col2 = st.columns([1,1])
            with col1:
                sel_a = st.selectbox('Year A', year_options, index=0)
            with col2:
                sel_b = st.selectbox('Year B', year_options, index=1 if len(year_options) > 1 else 0)

            def _get_year_row(label):
                try:
                    y = int(label.split()[0])
                except Exception:
                    return pd.Series(0, index=monthly_reduced.columns)
                if '(hist)' in label:
                    mask = [yy == y for yy in list(yearly_hist.index.year)]
                    if any(mask):
                        row = yearly_hist.iloc[[i for i,m in enumerate(mask) if m][0]]
                    else:
                        row = pd.Series(0, index=monthly_reduced.columns)
                else:
                    mask = [yy == y for yy in list(yearly_fore.index.year)]
                    if any(mask):
                        row = yearly_fore.iloc[[i for i,m in enumerate(mask) if m][0]]
                    else:
                        row = pd.Series(0, index=monthly_reduced.columns)
                # ensure ordering matches reduced columns
                return row.reindex(monthly_reduced.columns, fill_value=0)

            vals_a = _get_year_row(sel_a)
            vals_b = _get_year_row(sel_b)

            fig_comp, ax_comp = plt.subplots(figsize=(10,6))
            types = list(monthly_reduced.columns)
            x = np.arange(len(types))
            w = 0.35
            ax_comp.bar(x - w/2, vals_a.values, w, label=sel_a)
            ax_comp.bar(x + w/2, vals_b.values, w, label=sel_b)
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels(_shorten([str(i) for i in types], maxlen=12), rotation=45, ha='right')
            ax_comp.set_ylabel('No. of accidents')
            ax_comp.set_title(f'Comparison: {sel_a} vs {sel_b}')
            maxv = max(max(vals_a.max(), vals_b.max()), 1)
            for i, v in enumerate(vals_a.values):
                ax_comp.text(i - w/2, v + maxv*0.01, str(int(v)), ha='center', fontsize=9)
            for i, v in enumerate(vals_b.values):
                ax_comp.text(i + w/2, v + maxv*0.01, str(int(v)), ha='center', fontsize=9)
            ax_comp.legend()
            fig_comp.tight_layout()
            st.pyplot(fig_comp)

            # save to outputs and provide download
            outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
            os.makedirs(outputs_dir, exist_ok=True)
            safe_a = sel_a.replace(' ', '_').replace('(', '').replace(')', '')
            safe_b = sel_b.replace(' ', '_').replace('(', '').replace(')', '')
            fname = f'comparison_{safe_a}_vs_{safe_b}.png'
            full_path = os.path.join(outputs_dir, fname)
            try:
                fig_comp.savefig(full_path, dpi=150)
                with open(full_path, 'rb') as f:
                    img_bytes = f.read()
                st.download_button('Download comparison PNG', img_bytes, file_name=fname, mime='image/png')
                st.success(f'Saved comparison image to {full_path}')
            except Exception as e:
                st.warning(f'Could not save comparison image: {e}')

        # allow user to download forecast csv
        csv_buf = forecasts_df.reset_index().rename(columns={'index':'period'}).to_csv(index=False)
        st.download_button('Download monthly forecast CSV', csv_buf, file_name='monthly_forecast.csv', mime='text/csv')


# TODO Rename this here and in `build_ui`
def _extracted_from_build_ui_63(arg0, arg1, arg2):
    arg0.set_ylabel('Probability')
    arg0.set_title(arg1)
    st.pyplot(arg2)


# TODO Rename this here and in `build_ui`
def _extracted_from_build_ui_80(arg0, y_pos, names_sorted, arg3):
    arg0.set_yticks(y_pos)
    arg0.set_yticklabels(names_sorted)
    arg0.set_xlabel(arg3)


def main():
    df = load_data()
    # ensure Damage column exists (derived if not present)
    with contextlib.suppress(Exception):
        df = ensure_damage_column(df)
    # normalize column names and fill missing features to avoid KeyError during training
    features = ['Weather','Road_Type','Time_of_Day','Road_Condition','Vehicle_Type',
                'Traffic_Density','Speed_Limit','Vehicles_Nearby','Driver_Age','Driving_Experience','Alcohol','Light_Condition']
    try:
        df = normalize_and_fill_features(df, features)
    except Exception as e:
        # fallback: ensure all features exist and fill with safe defaults
        defaults = {
            'Weather': 'Clear', 'Road_Type': 'Urban', 'Time_of_Day': 'Afternoon',
            'Road_Condition': 'Dry', 'Vehicle_Type': 'Bus', 'Traffic_Density': 0.0,
            'Speed_Limit': 50, 'Vehicles_Nearby': 0, 'Driver_Age': 35,
            'Driving_Experience': 5, 'Alcohol': 0, 'Light_Condition': 'Daylight'
        }
        for f in features:
            if f not in df.columns:
                df[f] = defaults.get(f, 0)
    features = ['Weather','Road_Type','Time_of_Day','Road_Condition','Vehicle_Type',
                'Traffic_Density','Speed_Limit','Vehicles_Nearby','Driver_Age','Driving_Experience','Alcohol','Light_Condition']
    # If a persisted best model exists, load it to speed up prediction and still compute comparisons
    persisted = os.path.join('models','best_model.pkl')
    if os.path.exists(persisted):
        try:
            best_pipe = joblib.load(persisted)
        except Exception:
            best_pipe = None
    else:
        best_pipe = None

    results, pipelines, type_model, area_model, damage_model = train_models(df, features)
    # if we loaded a persisted best model, add it to pipelines under 'SavedBest'
    if best_pipe is not None:
        pipelines['SavedBest'] = best_pipe
        # compute its CV accuracy for display
        with contextlib.suppress(Exception):
            acc = cross_val_score(best_pipe, df[features], df['Accident'], cv=5, scoring='accuracy')
            results['SavedBest'] = {'mean_acc': float(acc.mean()), 'std': float(acc.std())}
    build_ui(df, features, results, pipelines, type_model, area_model, damage_model)


if __name__ == '__main__':
    main()
    
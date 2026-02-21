import contextlib
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
    return df


@st.cache_data
def load_data():
    csv_path = os.path.join('dataset','accidents.csv')
    return (
        pd.read_csv(csv_path)
        if os.path.exists(csv_path)
        else generate_synthetic()
    )


def train_models(df, features, target='Accident'):
    X = df[features]
    y = df[target]
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
        except Exception:
            # fallback: fit default
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
    # Train type/area models on accident rows if enough data, otherwise try full dataset (dropna)
    if len(accident_rows) >= 20:
        type_model = _extracted_from_train_models_31(
            accident_rows, features, 'Accident_Type', preproc
        )
        area_model = _extracted_from_train_models_31(
            accident_rows, features, 'Area', preproc
        )
    else:
        # try training on any rows that have Accident_Type/Area defined
        if df['Accident_Type'].dropna().shape[0] >= 10:
            type_model = _extracted_from_train_models_31(
                df[df['Accident_Type'].notna()], features, 'Accident_Type', preproc
            )
        if df['Area'].dropna().shape[0] >= 10:
            area_model = _extracted_from_train_models_31(
                df[df['Area'].notna()], features, 'Area', preproc
            )
    return results, pipelines, type_model, area_model


# TODO Rename this here and in `train_models`
def _extracted_from_train_models_31(accident_rows, features, arg2, preproc):
    X_type = accident_rows[features]
    y_type = accident_rows[arg2]
    pipe_type = Pipeline([('pre', preproc), ('clf', RandomForestClassifier(n_estimators=100))])
    pipe_type.fit(X_type, y_type)
    return pipe_type


def build_ui(df, features, results, pipelines, type_model, area_model):
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

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button('Predict'):
            input_df = pd.DataFrame([input_dict])
            best_name = max(results.items(), key=lambda x: x[1]['mean_acc'])[0]
            best_pipe = pipelines[best_name]
            prob = None
            try:
                prob = float(best_pipe.predict_proba(input_df)[0][1])
            except Exception:
                prob = float(best_pipe.predict(input_df)[0])

            st.markdown(f"**Predicted accident probability:** {prob:.2%}")
            # Predict Accident Type and Area regardless of probability if models exist
            if type_model is not None:
                try:
                    probs_type = type_model.predict_proba(input_df)[0]
                    classes_type = type_model.classes_
                    top_idx = np.argmax(probs_type)
                    pred_type = classes_type[top_idx]
                    st.markdown(f"**Predicted Accident Type:** {pred_type}")
                    fig_t, ax_t = plt.subplots()
                    ax_t.bar(classes_type, probs_type, color=plt.cm.Pastel1(np.linspace(0,1,len(classes_type))))
                    _extracted_from_build_ui_63(ax_t, 'Accident Type Probabilities', fig_t)
                except Exception:
                    pred_type = type_model.predict(input_df)[0]
                    st.markdown(f"**Predicted Accident Type:** {pred_type}")
            else:
                st.markdown("**Predicted Accident Type:** Model not available")

            if area_model is not None:
                try:
                    probs_area = area_model.predict_proba(input_df)[0]
                    classes_area = area_model.classes_
                    top_idx = np.argmax(probs_area)
                    pred_area = classes_area[top_idx]
                    st.markdown(f"**Likely Area of Occurrence:** {pred_area}")
                    fig_a, ax_a = plt.subplots()
                    ax_a.bar(classes_area, probs_area, color=plt.cm.Pastel2(np.linspace(0,1,len(classes_area))))
                    _extracted_from_build_ui_63(ax_a, 'Area Probabilities', fig_a)
                except Exception:
                    pred_area = area_model.predict(input_df)[0]
                    st.markdown(f"**Likely Area of Occurrence:** {pred_area}")
            else:
                st.markdown("**Likely Area of Occurrence:** Model not available")

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
                axs[1].bar(type_counts.index.astype(str), type_counts.values, color=plt.cm.Pastel1(np.linspace(0,1,len(type_counts))))
                axs[1].set_title('Accident Type')
            else:
                axs[1].text(0.5,0.5,'No Accident_Type', ha='center')
            # Area distribution
            if 'Area' in df_plot.columns:
                area_counts = df_plot['Area'].value_counts()
                axs[2].bar(area_counts.index.astype(str), area_counts.values, color=plt.cm.Pastel2(np.linspace(0,1,len(area_counts))))
                axs[2].set_title('Area')
            else:
                axs[2].text(0.5,0.5,'No Area', ha='center')
            plt.tight_layout()
            st.pyplot(figd)

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

    results, pipelines, type_model, area_model = train_models(df, features)
    # if we loaded a persisted best model, add it to pipelines under 'SavedBest'
    if best_pipe is not None:
        pipelines['SavedBest'] = best_pipe
        # compute its CV accuracy for display
        with contextlib.suppress(Exception):
            acc = cross_val_score(best_pipe, df[features], df['Accident'], cv=5, scoring='accuracy')
            results['SavedBest'] = {'mean_acc': float(acc.mean()), 'std': float(acc.std())}
    build_ui(df, features, results, pipelines, type_model, area_model)


if __name__ == '__main__':
    main()
    
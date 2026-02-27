import contextlib
import os
import pandas as pd
import numpy as np

def load_all_datasets(dataset_dir='dataset'):
    if not os.path.exists(dataset_dir):
        return pd.DataFrame()
    files = [
        os.path.join(dataset_dir, fn)
        for fn in os.listdir(dataset_dir)
        if fn.lower().endswith(('.csv', '.xls', '.xlsx'))
    ]
    parts = []
    for f in files:
        try:
            if f.lower().endswith('.csv'):
                parts.append(pd.read_csv(f))
            else:
                parts.append(pd.read_excel(f))
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _detect_date_col(df):
    candidates = ['Date','date','Timestamp','timestamp','Reported_Date','reported_date','ReportDate','report_date','Report_Date']
    return next(
        (c for c in candidates if c in df.columns),
        (
            ('Year', 'Month')
            if 'Year' in df.columns and 'Month' in df.columns
            else None
        ),
    )


def aggregate_monthly_counts(df):
    """Return monthly counts for total accidents and simple cause categories.

    Categories attempted: drunk_and_drive, road_fault, human_fault, vehicle_fault
    Uses heuristics based on available columns (Alcohol, Cause, Accident_Type, Road_Condition).
    """
    if df is None or df.shape[0] == 0:
        return pd.DataFrame()
    dfc = df.copy()
    dc = _detect_date_col(dfc)
    if isinstance(dc, tuple):
        dfc['date'] = pd.to_datetime(dfc['Year'].astype(str) + '-' + dfc['Month'].astype(str) + '-01', errors='coerce')
    elif dc is not None:
        dfc['date'] = pd.to_datetime(dfc[dc], errors='coerce')
    else:
        # assume each row represents an accident and create synthetic dates spread across recent years
        n = dfc.shape[0]
        end = pd.Timestamp.today()
        start = end - pd.DateOffset(years=3)
        dfc['date'] = pd.date_range(start=start, end=end, periods=n)

    dfc = dfc.dropna(subset=['date']).copy()
    # indicator for accident rows: if 'Accident' exists and is 0/1, use it; else assume each row is an accident
    if 'Accident' in dfc.columns:
        mask_acc = dfc['Accident'].astype(float) == 1
        dfc = dfc[mask_acc]

    # cause heuristics
    text_cols = []
    for c in ['Cause','Accident_Type','Damage']:
        if c in dfc.columns:
            text_cols.append(c)

    def contains_any(row, keywords):
        for c in text_cols:
            try:
                s = str(row.get(c, '')).lower()
            except Exception:
                s = ''
            for kw in keywords:
                if kw in s:
                    return True
        return False

    dfc['drunk_and_drive'] = 0
    if 'Alcohol' in dfc.columns:
        try:
            dfc['drunk_and_drive'] = (dfc['Alcohol'].astype(float) > 0).astype(int)
        except Exception:
            dfc['drunk_and_drive'] = dfc['drunk_and_drive']
    else:
        dfc['drunk_and_drive'] = dfc.apply(lambda r: 1 if contains_any(r, ['drunk','alcohol','drink']) else 0, axis=1)

    dfc['road_fault'] = dfc.apply(lambda r: 1 if contains_any(r, ['road','pothole','surface','maintenance','skid','slip']) else 0, axis=1)
    dfc['human_fault'] = dfc.apply(lambda r: 1 if contains_any(r, ['human','driver','error','fatigue','distract','overspeed','over speed','neglig']) else 0, axis=1)
    dfc['vehicle_fault'] = dfc.apply(lambda r: 1 if contains_any(r, ['vehicle','mechanic','brake','tyre','engine','electrical']) else 0, axis=1)

    dfc['period'] = dfc['date'].dt.to_period('M').dt.to_timestamp()
    grp = dfc.groupby('period').size().rename('Total')
    causes = ['drunk_and_drive','road_fault','human_fault','vehicle_fault']
    for c in causes:
        s = dfc.groupby('period')[c].sum()
        grp = pd.concat([grp, s.rename(c)], axis=1)
    grp = grp.fillna(0).astype(float)
    # ensure continuous monthly index
    if grp.shape[0] > 0:
        idx = pd.date_range(start=grp.index.min(), end=grp.index.max(), freq='MS')
        grp = grp.reindex(idx, fill_value=0)
        grp.index.name = 'period'
    return grp


def fit_and_forecast_regressors(series, months_ahead=120):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200, random_state=42
        )
    }
    models['Ridge'] = Ridge()
    models['GBRT'] = GradientBoostingRegressor(n_estimators=200)
    models['SVR'] = SVR()
    with contextlib.suppress(Exception):
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=200, verbosity=0)
    s = series.asfreq('MS').fillna(0)
    n = len(s)
    if n < 6:
        # not enough history, return empty
        return {}

    df_f = pd.DataFrame({'y': s.values})
    df_f['t'] = np.arange(n)
    df_f['month'] = s.index.month
    df_f['msin'] = np.sin(2 * np.pi * df_f['month'] / 12)
    df_f['mcos'] = np.cos(2 * np.pi * df_f['month'] / 12)
    for lag in (1,2,3):
        df_f[f'lag{lag}'] = df_f['y'].shift(lag).fillna(method='bfill')

    X = df_f.drop(columns=['y'])
    y = df_f['y']

    fitted = {}
    for name, mdl in models.items():
        try:
            mdl.fit(X, y)
            fitted[name] = mdl
        except Exception:
            continue

    last_t = X['t'].iloc[-1]
    start = s.index[-1] + pd.offsets.MonthBegin(1)
    future_idx = pd.date_range(start=start, periods=months_ahead, freq='MS')
    forecasts = {}
    for name, mdl in fitted.items():
        preds = []
        hist_vals = list(y.values)
        for i in range(months_ahead):
            tval = last_t + 1 + i
            mval = (future_idx[i].month)
            row = [tval, mval, np.sin(2 * np.pi * mval / 12), np.cos(2 * np.pi * mval / 12)]
            # lags
            row.extend(
                (
                    hist_vals[-lag]
                    if len(hist_vals) >= lag
                    else hist_vals[-1] if hist_vals else 0.0
                )
                for lag in (1, 2, 3)
            )
            xr = np.array(row).reshape(1, -1)
            try:
                p = float(mdl.predict(xr)[0])
            except Exception:
                p = float(np.mean(hist_vals[-3:]) if hist_vals else 0.0)
            p = max(0.0, p)
            preds.append(p)
            hist_vals.append(p)
        forecasts[name] = pd.Series(preds, index=future_idx)

    return forecasts


def forecast_all_counts(df, months_ahead=120):
    """Return (history_monthly, forecasts_per_category) where forecasts_per_category is dict:
    category -> { model_name: pd.Series }
    """
    monthly = aggregate_monthly_counts(df)
    if monthly.shape[0] == 0:
        return monthly, {}

    results = {}
    for col in monthly.columns:
        ser = monthly[col]
        res = fit_and_forecast_regressors(ser, months_ahead=months_ahead)
        results[col] = res

    return monthly, results

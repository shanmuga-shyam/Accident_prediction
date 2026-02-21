import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype


def preprocess_data(path):

    df = pd.read_excel(path)

    # Remove useless columns
    drop_cols = [
        'S.NO','Vehicle No','Staff Number','IRTHVLicNo',
        'Accident Date','Accident Time','Trip Time',
        'Place  Name','Road Name','Near Town'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Fill missing values
    df = df.fillna("Unknown")

    # Encode non-numeric (categorical) columns. Convert values to string
    # first to avoid mixed-type issues (e.g., ints and strings in same column).
    encoders = {}
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    return df, encoders
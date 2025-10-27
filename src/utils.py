# src/utils.py
import joblib
import os
import pandas as pd
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def save_model(obj, name):
    path = os.path.join(MODELS_DIR, name)
    joblib.dump(obj, path)
    print(f"Saved model: {name}")
    return path

def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def save_df(df, name):
    path = os.path.join(MODELS_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved CSV: {name}")
    return path

def parse_datetime_series(df, col):
    return pd.to_datetime(df[col], errors="coerce")

def ensure_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def clean_infinity(df, columns=None):
    import numpy as np
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


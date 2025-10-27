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

def save_model(obj, name):
    path = os.path.join(MODELS_DIR, name)
    joblib.dump(obj, path)
    return path

def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    return joblib.load(path)

def save_df(df, name):
    path = os.path.join(MODELS_DIR, name)
    df.to_csv(path, index=False)
    return path

def parse_datetime_series(df, col):
    return pd.to_datetime(df[col], errors="coerce")

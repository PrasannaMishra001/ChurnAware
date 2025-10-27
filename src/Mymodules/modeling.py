# src/Mymodules/modeling.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from ..utils import save_model
import joblib

def define_churn_label(features_df, as_of_date=None, churn_days=90):
    df = features_df.copy()
    if as_of_date is None:
        # estimate as_of_date from last_order_date if present
        if 'last_order_date' in df.columns:
            as_of_date = pd.to_datetime(df['last_order_date'].max()) + pd.Timedelta(days=1)
        else:
            raise ValueError("as_of_date required if last_order_date not present")
    else:
        as_of_date = pd.to_datetime(as_of_date)

    df['last_order_date'] = pd.to_datetime(df['last_order_date'], errors='coerce')
    df['days_since_last_order'] = (as_of_date - df['last_order_date']).dt.days.fillna(9999)
    # churn label: no order in last churn_days -> churned
    df['churn'] = (df['days_since_last_order'] > churn_days).astype(int)
    return df

def train_churn_model(df, feature_cols=None, random_state=42, save=True):
    df = df.copy()
    if feature_cols is None:
        feature_cols = [
            'recency_days','frequency','monetary','delivery_ratio',
            'avg_rating','negative_feedback_count','avg_sentiment','avg_order_value'
        ]
    X = df[feature_cols].fillna(0)
    y = df['churn'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique()>1 else None)

    model = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc_score': float(roc_auc_score(y_test, y_proba)) if y_proba is not None and len(set(y_test))>1 else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    if save:
        save_model(model, "churn_prediction_model.pkl")
        # save feature list as joblib
        joblib.dump(feature_cols, save_model_path("churn_feature_cols.joblib"))

    return model, metrics, (X_train, X_test, y_train, y_test)

def save_model_path(filename):
    # helper used above: returns path string using save_model but without saving again (works for small use)
    from ..utils import MODELS_DIR
    import os
    return os.path.join(MODELS_DIR, filename)

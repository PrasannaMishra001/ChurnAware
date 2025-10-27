# src/Mymodules/modeling.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

def define_churn_label(features_df, as_of_date=None, churn_days=90):
    df = features_df.copy()
    
    if as_of_date is None:
        if 'last_order_date' in df.columns:
            as_of_date = pd.to_datetime(df['last_order_date'].max()) + pd.Timedelta(days=1)
        else:
            raise ValueError("as_of_date required if last_order_date not present")
    else:
        as_of_date = pd.to_datetime(as_of_date)

    df['last_order_date'] = pd.to_datetime(df['last_order_date'], errors='coerce')
    df['days_since_last_order'] = (as_of_date - df['last_order_date']).dt.days.fillna(9999)
    
    df['churn'] = (df['days_since_last_order'] > churn_days).astype(int)
    
    return df


def train_churn_model(df, feature_cols=None, random_state=42, save=True):
    from ..utils import save_model, MODELS_DIR
    
    df = df.copy()
    
    if feature_cols is None:
        feature_cols = [
            'recency_days', 'frequency', 'monetary', 'delivery_ratio',
            'avg_rating', 'negative_feedback_count', 'avg_sentiment', 'avg_order_value'
        ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    y = df['churn'].astype(int)

    if y.nunique() == 1:
        print(f"Warning: All customers have churn status = {y.iloc[0]}. Cannot train model.")
        return None, {'error': 'Insufficient class variation'}, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, 
        stratify=y if y.nunique() > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state, 
        class_weight='balanced', 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    if y_proba is not None and len(set(y_test)) > 1:
        metrics['roc_auc_score'] = float(roc_auc_score(y_test, y_proba))
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        metrics['feature_importance'] = feature_importance.to_dict('records')

    if save:
        save_model(model, "churn_prediction_model.pkl")
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, "churn_feature_cols.joblib"))
        
        import json
        with open(os.path.join(MODELS_DIR, "churn_model_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)

    return model, metrics, (X_train, X_test, y_train, y_test)


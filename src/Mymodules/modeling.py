# src/Mymodules/modeling.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

CHURN_FEATURES = [
    'recency_days', 'frequency', 'monetary', 'on_time_ratio', 'avg_delay_score',
    'avg_rating', 'negative_feedback_count', 'avg_sentiment',
    'avg_text_sentiment', 'text_negative_count', 'avg_order_value',
    'orders_per_month', 'customer_lifespan_days', 'count_feedback'
]


def train_churn_model(train_df, test_df, feature_cols=None, random_state=42, save=True):
    from ..utils import save_model, MODELS_DIR

    train_df = train_df.copy()
    test_df = test_df.copy()

    if feature_cols is None:
        feature_cols = CHURN_FEATURES

    for df in (train_df, test_df):
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['churn'].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['churn'].astype(int)

    if y_train.nunique() == 1:
        print(f"Warning: All training customers have churn status = {y_train.iloc[0]}. Cannot train model.")
        return None, {'error': 'Insufficient class variation'}, None

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
        'evaluation': 'temporal holdout (features from past window, label from 90-day future window)',
        'train_snapshots': sorted(train_df['snapshot_date'].unique().tolist()) if 'snapshot_date' in train_df.columns else None,
        'test_snapshots': sorted(test_df['snapshot_date'].unique().tolist()) if 'snapshot_date' in test_df.columns else None,
        'train_rows': int(len(train_df)),
        'test_rows': int(len(test_df)),
        'train_churn_rate': float(y_train.mean()),
        'test_churn_rate': float(y_test.mean()),
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

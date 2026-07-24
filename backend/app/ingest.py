# backend/app/ingest.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sqlalchemy import text

from src.utils import MODELS_DIR
from backend.app.db import engine
from backend.app.services import score_churn, recommend_actions

CUSTOMER_COLUMNS = [
    'customer_id', 'customer_name', 'area', 'pincode', 'segment_name',
    'recency_days', 'frequency', 'monetary', 'avg_order_value',
    'on_time_ratio', 'avg_delay_score', 'avg_rating', 'avg_sentiment',
    'avg_text_sentiment', 'negative_feedback_count', 'count_feedback',
    'orders_per_month', 'customer_lifespan_days', 'text_negative_count',
]


def run_ingest():
    features_path = os.path.join(MODELS_DIR, "customer_feature_engineered_with_segments.csv")
    profiles_path = os.path.join(MODELS_DIR, "segment_profiles.csv")

    df = pd.read_csv(features_path)
    numeric_cols = [c for c in CUSTOMER_COLUMNS if c not in
                    ('customer_id', 'customer_name', 'area', 'pincode', 'segment_name')]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for col in ('customer_name', 'area', 'pincode'):
        if col in df.columns:
            df[col] = df[col].astype(str)

    df['churn_proba'] = score_churn(df)
    action_ids, action_names, q_values = recommend_actions(df)
    df['recommended_action_id'] = action_ids
    df['recommended_action'] = action_names
    df['expected_value'] = q_values.max(axis=1)

    out_cols = CUSTOMER_COLUMNS + [
        'churn_proba', 'recommended_action_id', 'recommended_action', 'expected_value'
    ]
    customers = df[[c for c in out_cols if c in df.columns]].copy()
    customers.to_sql('customers', engine, if_exists='replace', index=False)

    profiles = pd.read_csv(profiles_path)
    profiles.to_sql('segment_profiles', engine, if_exists='replace', index=False)

    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_customers_id ON customers (customer_id)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers (churn_proba)"
        ))

    print(f"Ingested {len(customers)} customers and {len(profiles)} segment profiles into SQLite")
    return len(customers)


if __name__ == "__main__":
    run_ingest()

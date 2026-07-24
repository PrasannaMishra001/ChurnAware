# src/Mymodules/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import timedelta

def compute_R_F_M_D(customers_df, orders_df, order_items_df, as_of_date=None, save_csv=True):
    from ..utils import save_df
    
    orders_df = orders_df.copy()
    order_items_df = order_items_df.copy()
    customers_df = customers_df.copy()

    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'], errors='coerce')
    
    if as_of_date is None:
        as_of_date = orders_df['order_date'].max() + pd.Timedelta(days=1)
    else:
        as_of_date = pd.to_datetime(as_of_date)
    
    print(f"Computing features with as_of_date: {as_of_date}")

    orders_df['order_total'] = pd.to_numeric(orders_df['order_total'], errors='coerce').fillna(0)
    
    agg = orders_df.groupby('customer_id').agg(
        last_order_date=('order_date', 'max'),
        first_order_date=('order_date', 'min'),
        frequency=('order_id', 'nunique'),
        monetary=('order_total', 'sum'),
        total_orders=('order_id', 'nunique')
    ).reset_index()

    agg['recency_days'] = (as_of_date - agg['last_order_date']).dt.days.clip(lower=0)

    delay_weights = {'On Time': 0.0, 'Slightly Delayed': 0.5, 'Significantly Delayed': 1.0}
    status = orders_df['delivery_status'].astype(str).str.strip()
    orders_df['on_time_flag'] = (status == 'On Time').astype(int)
    orders_df['delay_score'] = status.map(delay_weights).fillna(1.0)

    delivery = orders_df.groupby('customer_id').agg(
        on_time_orders=('on_time_flag', 'sum'),
        avg_delay_score=('delay_score', 'mean')
    ).reset_index()

    agg = agg.merge(delivery, on='customer_id', how='left')
    agg['on_time_orders'] = agg['on_time_orders'].fillna(0)
    agg['avg_delay_score'] = agg['avg_delay_score'].fillna(0)

    agg['on_time_ratio'] = np.where(
        agg['total_orders'] > 0,
        agg['on_time_orders'] / agg['total_orders'],
        0
    )
    agg['on_time_ratio'] = agg['on_time_ratio'].clip(0, 1)

    agg['avg_order_value'] = np.where(
        agg['total_orders'] > 0,
        agg['monetary'] / agg['total_orders'],
        0
    )
    
    agg['avg_order_value'] = agg['avg_order_value'].replace([np.inf, -np.inf], 0).fillna(0)

    agg['customer_lifespan_days'] = (agg['last_order_date'] - agg['first_order_date']).dt.days.fillna(0).clip(lower=0)
    agg['orders_per_month'] = np.where(
        agg['customer_lifespan_days'] > 0,
        agg['frequency'] / (agg['customer_lifespan_days'] / 30.0),
        agg['frequency']
    )

    out = customers_df.merge(agg, on='customer_id', how='left', suffixes=('_old', ''))
    
    if 'avg_order_value_old' in out.columns:
        out = out.drop(columns=['avg_order_value_old'])
    if 'total_orders_old' in out.columns:
        out = out.drop(columns=['total_orders_old'])

    numeric_cols = ['frequency', 'monetary', 'recency_days', 'on_time_ratio',
                    'avg_delay_score', 'avg_order_value', 'total_orders',
                    'on_time_orders', 'customer_lifespan_days', 'orders_per_month']
    
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    
    out = out.replace([np.inf, -np.inf], 0)
    
    print(f"Feature engineering complete: {len(out)} customers, {(out['frequency'] > 0).sum()} with orders")
    print(f"Avg order value stats: min={out['avg_order_value'].min():.2f}, max={out['avg_order_value'].max():.2f}, mean={out['avg_order_value'].mean():.2f}")

    if save_csv:
        save_df(out, "customer_feature_engineered.csv")

    return out, as_of_date


def aggregate_feedback_features(feedback_df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    df = feedback_df.copy()
    df['feedback_date'] = pd.to_datetime(df['feedback_date'], errors='coerce')

    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map).fillna(0)

    analyzer = SentimentIntensityAnalyzer()
    df['text_compound'] = df['feedback_text'].astype(str).apply(
        lambda t: analyzer.polarity_scores(t)['compound']
    )

    agg = df.groupby('customer_id').agg(
        avg_rating=('rating', 'mean'),
        count_feedback=('feedback_id', 'count'),
        negative_feedback_count=('sentiment', lambda s: (s == 'Negative').sum()),
        positive_feedback_count=('sentiment', lambda s: (s == 'Positive').sum()),
        avg_sentiment=('sentiment_score', 'mean'),
        avg_text_sentiment=('text_compound', 'mean'),
        text_negative_count=('text_compound', lambda s: int((s < -0.05).sum())),
        last_feedback_date=('feedback_date', 'max')
    ).reset_index()

    return agg


def build_customer_features(customers_df, orders_df, order_items_df, feedback_df, 
                           as_of_date=None, save_csv=True):
    from ..utils import save_df
    
    print("Building customer features...")
    
    features_df, as_of_date = compute_R_F_M_D(
        customers_df, orders_df, order_items_df, 
        as_of_date=as_of_date, save_csv=False
    )
    
    print("Aggregating feedback features...")
    feedback_agg = aggregate_feedback_features(feedback_df)
    
    merged = features_df.merge(feedback_agg, on='customer_id', how='left')
    
    fill_cols = ['avg_rating', 'count_feedback', 'negative_feedback_count',
                 'positive_feedback_count', 'avg_sentiment',
                 'avg_text_sentiment', 'text_negative_count']
    for c in fill_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
        else:
            merged[c] = 0

    merged['had_negative_feedback'] = (merged['negative_feedback_count'] > 0).astype(int)

    merged = merged.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"Final feature set: {merged.shape[0]} rows, {merged.shape[1]} columns")

    if save_csv:
        save_df(merged, "customer_feature_engineered.csv")

    return merged, as_of_date
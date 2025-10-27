# src/Mymodules/feature_engineering.py
import pandas as pd
import numpy as np
from datetime import timedelta
from ..utils import save_df, parse_datetime_series

def compute_R_F_M_D(customers_df, orders_df, order_items_df, as_of_date=None, save_csv=True):
    # Ensure correct dtypes
    orders_df = orders_df.copy()
    order_items_df = order_items_df.copy()
    customers_df = customers_df.copy()

    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'], errors='coerce')
    if as_of_date is None:
        # default as the max order_date + 1 day
        as_of_date = orders_df['order_date'].max() + pd.Timedelta(days=1)
    else:
        as_of_date = pd.to_datetime(as_of_date)

    # Monetary per order: use order_total from orders if present
    # Frequency: number of orders per customer
    agg = orders_df.groupby('customer_id').agg(
        last_order_date=('order_date', 'max'),
        first_order_date=('order_date', 'min'),
        frequency=('order_id', 'nunique'),
        monetary=('order_total', 'sum'),
        total_orders=('order_id', 'nunique')
    ).reset_index()

    # Recency in days
    agg['recency_days'] = (as_of_date - agg['last_order_date']).dt.days

    # Delivery ratio D = delivered_orders / total_orders
    # Determine delivered vs cancelled by delivery_status: consider 'On Time' and 'Slightly Delayed' as delivered
    delivered_status = set(['On Time', 'Slightly Delayed', 'Significantly Delayed'])
    orders_df['delivered_flag'] = orders_df['delivery_status'].fillna('').apply(
        lambda x: 1 if x in delivered_status else 0
    )
    delivered = orders_df.groupby('customer_id').agg(delivered_orders=('delivered_flag', 'sum')).reset_index()
    agg = agg.merge(delivered, on='customer_id', how='left')
    agg['delivered_orders'] = agg['delivered_orders'].fillna(0)
    agg['delivery_ratio'] = agg['delivered_orders'] / agg['total_orders']
    agg['delivery_ratio'] = agg['delivery_ratio'].fillna(0)

    # Average order value
    agg['avg_order_value'] = agg['monetary'] / agg['total_orders']
    agg['avg_order_value'] = agg['avg_order_value'].fillna(0)

    # Additional features: days_between_orders (approx)
    agg['customer_lifespan_days'] = (agg['last_order_date'] - agg['first_order_date']).dt.days.fillna(0)
    agg['orders_per_month'] = agg['frequency'] / (agg['customer_lifespan_days']/30 + 1e-9)

    # Merge with customers to keep other metadata
    out = customers_df.merge(agg, on='customer_id', how='left')

    # Ensure all columns exist before fillna
    for col in ['frequency','monetary','recency_days','delivery_ratio','avg_order_value']:
        if col not in out.columns:
            out[col] = 0

    out[['frequency','monetary','recency_days','delivery_ratio','avg_order_value']] = (
        out[['frequency','monetary','recency_days','delivery_ratio','avg_order_value']].fillna(0)
    )

    if save_csv:
        save_df(out, "customer_feature_engineered.csv")

    return out, as_of_date


def aggregate_feedback_features(feedback_df):
    df = feedback_df.copy()
    df['feedback_date'] = pd.to_datetime(df['feedback_date'], errors='coerce')

    # For each customer: avg_rating, negative_feedback_count, positive_feedback_count, avg_sentiment (map)
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map).fillna(0)
    agg = df.groupby('customer_id').agg(
        avg_rating = ('rating', 'mean'),
        count_feedback = ('feedback_id', 'count'),
        negative_feedback_count = ('sentiment', lambda s: (s=='Negative').sum()),
        positive_feedback_count = ('sentiment', lambda s: (s=='Positive').sum()),
        avg_sentiment = ('sentiment_score', 'mean'),
        last_feedback_date = ('feedback_date', 'max')
    ).reset_index()

    return agg

def build_customer_features(customers_df, orders_df, order_items_df, feedback_df, as_of_date=None, save_csv=True):
    features_df, as_of_date = compute_R_F_M_D(customers_df, orders_df, order_items_df, as_of_date=as_of_date, save_csv=False)
    feedback_agg = aggregate_feedback_features(feedback_df)
    merged = features_df.merge(feedback_agg, on='customer_id', how='left')
    # Fill missing feedback aggregates with 0s
    fill_cols = ['avg_rating','count_feedback','negative_feedback_count','positive_feedback_count','avg_sentiment']
    for c in fill_cols:
        merged[c] = merged[c].fillna(0)

    # Flags
    merged['had_negative_feedback'] = (merged['negative_feedback_count'] > 0).astype(int)

    if save_csv:
        save_df(merged, "customer_feature_engineered.csv")

    return merged, as_of_date

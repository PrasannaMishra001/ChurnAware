# src/Mymodules/labeling.py
import pandas as pd

DEFAULT_CUTOFFS = [
    '2023-09-01', '2023-11-01', '2024-01-01',
    '2024-03-01', '2024-05-01', '2024-07-01'
]


def build_snapshot_dataset(customers_df, orders_df, order_items_df, feedback_df,
                           cutoffs=None, horizon_days=90):
    from .feature_engineering import build_customer_features

    if cutoffs is None:
        cutoffs = DEFAULT_CUTOFFS

    customers = customers_df.copy()
    orders = orders_df.copy()
    feedback = feedback_df.copy()

    customers['registration_date'] = pd.to_datetime(customers['registration_date'], errors='coerce')
    orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')
    feedback['feedback_date'] = pd.to_datetime(feedback['feedback_date'], errors='coerce')

    snapshots = []
    for cutoff in cutoffs:
        T = pd.to_datetime(cutoff)
        horizon_end = T + pd.Timedelta(days=horizon_days)

        past_orders = orders[orders['order_date'] <= T]
        past_feedback = feedback[feedback['feedback_date'] <= T]
        eligible = customers[customers['registration_date'] <= T]

        if len(past_orders) == 0 or len(eligible) == 0:
            continue

        features, _ = build_customer_features(
            eligible, past_orders, order_items_df, past_feedback,
            as_of_date=T, save_csv=False
        )
        features = features[features['frequency'] > 0].copy()

        future_orders = orders[
            (orders['order_date'] > T) & (orders['order_date'] <= horizon_end)
        ]
        active_ids = set(future_orders['customer_id'].unique())
        features['churn'] = (~features['customer_id'].isin(active_ids)).astype(int)
        features['snapshot_date'] = T.strftime('%Y-%m-%d')
        snapshots.append(features)

    dataset = pd.concat(snapshots, ignore_index=True)
    return dataset


def temporal_train_test_split(dataset, test_snapshots=1):
    dates = sorted(dataset['snapshot_date'].unique())
    test_dates = set(dates[-test_snapshots:])
    train_df = dataset[~dataset['snapshot_date'].isin(test_dates)].copy()
    test_df = dataset[dataset['snapshot_date'].isin(test_dates)].copy()
    return train_df, test_df

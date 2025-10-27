# src/train_segmentation.py
"""
Script to create R+FMD features and run GMM segmentation.
Usage:
    python src/train_segmentation.py
"""
import os
from src.data_loader import load_all
from src.Mymodules.feature_engineering import build_customer_features
from src.Mymodules.segmentation import run_gmm_segmentation
from src.utils import save_df

def main():
    data = load_all()
    customers = data['customers']
    orders = data['orders']
    order_items = data['order_items']
    feedback = data['feedback']

    merged_features, as_of = build_customer_features(customers, orders, order_items, feedback, as_of_date=None, save_csv=True)

    # run segmentation
    segmented_df, gmm, scaler, profile_df = run_gmm_segmentation(merged_features, n_components=4, save=True)

    print("Segmentation done. Cluster profile:")
    print(profile_df)

if __name__ == "__main__":
    main()

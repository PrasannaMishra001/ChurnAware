# src/Mymodules/segmentation.py
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from ..utils import save_model, save_df

def run_gmm_segmentation(features_df, n_components=4, random_state=42, features_for_clustering=None, save=True):
    df = features_df.copy()
    if features_for_clustering is None:
        features_for_clustering = ['frequency', 'monetary', 'delivery_ratio']

    X = df[features_for_clustering].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(X_scaled)
    df['segment_label'] = labels

    # Score cluster centers in original feature space
    centers = gmm.means_
    # Assign business-friendly buckets by ranking cluster mean monetary*frequency*delivery_ratio
    cluster_profile = []
    for i in range(n_components):
        cluster_indices = (labels == i)
        cluster_profile.append({
            'cluster': i,
            'size': int(cluster_indices.sum()),
            'mean_frequency': float(df.loc[cluster_indices, 'frequency'].mean()),
            'mean_monetary': float(df.loc[cluster_indices, 'monetary'].mean()),
            'mean_delivery_ratio': float(df.loc[cluster_indices, 'delivery_ratio'].mean())
        })
    profile_df = pd.DataFrame(cluster_profile).sort_values(by=['mean_monetary','mean_frequency'], ascending=False).reset_index(drop=True)
    # Create mapping to human-friendly labels
    label_map = {}
    # Example heuristics: highest monetary & freq -> High-Value, smallest monetary & delivery_ratio -> At-Risk etc.
    sorted_clusters = profile_df['cluster'].tolist()
    # assign:
    order_map = {sorted_clusters[0]: 'High-Value'}
    if len(sorted_clusters) > 1:
        order_map[sorted_clusters[1]] = 'Mid-Value'
    if len(sorted_clusters) > 2:
        order_map[sorted_clusters[2]] = 'Low-Value'
    if len(sorted_clusters) > 3:
        order_map[sorted_clusters[3]] = 'At-Risk'

    df['segment_name'] = df['segment_label'].map(order_map).fillna('Other')

    # Save objects
    if save:
        save_model(gmm, "customer_segments_gmm.pkl")
        # save engineered features with segments
        save_df(df, "customer_feature_engineered_with_segments.csv")

    return df, gmm, scaler, profile_df

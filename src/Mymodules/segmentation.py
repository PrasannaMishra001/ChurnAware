# src/Mymodules/segmentation.py
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def run_gmm_segmentation(features_df, n_components=4, random_state=42, 
                         features_for_clustering=None, save=True):
    from ..utils import save_model, save_df
    
    df = features_df.copy()
    
    if features_for_clustering is None:
        features_for_clustering = ['frequency', 'monetary', 'delivery_ratio']

    X = df[features_for_clustering].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state, 
                          covariance_type='full', n_init=10)
    labels = gmm.fit_predict(X_scaled)
    df['segment_label'] = labels

    cluster_profile = []
    for i in range(n_components):
        cluster_mask = (labels == i)
        cluster_data = df.loc[cluster_mask]
        
        cluster_profile.append({
            'cluster': i,
            'size': int(cluster_mask.sum()),
            'mean_frequency': float(cluster_data['frequency'].mean()),
            'mean_monetary': float(cluster_data['monetary'].mean()),
            'mean_delivery_ratio': float(cluster_data['delivery_ratio'].mean()),
            'mean_avg_order_value': float(cluster_data['avg_order_value'].mean()),
            'mean_negative_feedback': float(cluster_data['negative_feedback_count'].mean())
        })
    
    profile_df = pd.DataFrame(cluster_profile)
    
    profile_df['composite_score'] = (
        profile_df['mean_monetary'] * 0.5 + 
        profile_df['mean_frequency'] * 0.3 + 
        profile_df['mean_delivery_ratio'] * 0.2
    )
    profile_df = profile_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    sorted_clusters = profile_df['cluster'].tolist()
    label_map = {}
    
    if len(sorted_clusters) >= 4:
        label_map[sorted_clusters[0]] = 'High-Value Champions'
        label_map[sorted_clusters[1]] = 'Promising Customers'
        label_map[sorted_clusters[2]] = 'Needs Attention'
        label_map[sorted_clusters[3]] = 'At-Risk'
    elif len(sorted_clusters) == 3:
        label_map[sorted_clusters[0]] = 'High-Value'
        label_map[sorted_clusters[1]] = 'Mid-Value'
        label_map[sorted_clusters[2]] = 'At-Risk'
    elif len(sorted_clusters) == 2:
        label_map[sorted_clusters[0]] = 'Active'
        label_map[sorted_clusters[1]] = 'At-Risk'
    else:
        label_map[sorted_clusters[0]] = 'All Customers'
    
    df['segment_name'] = df['segment_label'].map(label_map).fillna('Other')
    profile_df['segment_name'] = profile_df['cluster'].map(label_map)

    if save:
        save_model(gmm, "customer_segments_gmm.pkl")
        save_model(scaler, "customer_segments_scaler.pkl")
        save_df(df, "customer_feature_engineered_with_segments.csv")
        save_df(profile_df, "segment_profiles.csv")

    return df, gmm, scaler, profile_df


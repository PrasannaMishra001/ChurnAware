# src/train_segmentation.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_all
from src.Mymodules.feature_engineering import build_customer_features
from src.Mymodules.segmentation import run_gmm_segmentation
from src.utils import save_df

def main():
    print("BLINKIT CUSTOMER SEGMENTATION PIPELINE")
    
    print("\n[1/4] Loading datasets...")
    data = load_all()
    customers = data['customers']
    orders = data['orders']
    order_items = data['order_items']
    feedback = data['feedback']
    
    print(f"Loaded {len(customers)} customers")
    print(f"Loaded {len(orders)} orders")
    print(f"Loaded {len(feedback)} feedback records")
    
    print("\n[2/4] Engineering R+FMD features...")
    try:
        merged_features, as_of = build_customer_features(
            customers, orders, order_items, feedback, 
            as_of_date=None, save_csv=True
        )
        print(f"Created features for {len(merged_features)} customers")
        print(f"As-of date: {as_of}")
        
        customers_with_orders = (merged_features['frequency'] > 0).sum()
        print(f"Customers with orders: {customers_with_orders}")
        print(f"Avg order value range: Rs.{merged_features['avg_order_value'].min():.2f} - Rs.{merged_features['avg_order_value'].max():.2f}")
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[3/4] Running GMM segmentation...")
    try:
        segmented_df, gmm, scaler, profile_df = run_gmm_segmentation(
            merged_features, n_components=4, save=True
        )
        
        print(f"Segmentation complete")
        print(f"Created {len(profile_df)} segments")
        
    except Exception as e:
        print(f"Error in segmentation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n[4/4] Segment Profiles:")
    print(profile_df.to_string(index=False))
    
    print("\nSUMMARY STATISTICS")
    
    for _, row in profile_df.iterrows():
        pct = (row['size'] / len(segmented_df) * 100)
        print(f"\n{row['segment_name']}:")
        print(f"  Size: {row['size']:,} customers ({pct:.1f}%)")
        print(f"  Avg Frequency: {row['mean_frequency']:.2f} orders")
        print(f"  Avg Monetary: Rs.{row['mean_monetary']:,.2f}")
        print(f"  Avg Delivery Ratio: {row['mean_delivery_ratio']:.1%}")
        if 'mean_avg_order_value' in row:
            print(f"  Avg Order Value: Rs.{row['mean_avg_order_value']:,.2f}")
    
    print("\nPipeline completed successfully")
    print(f"Saved models to: models/")
    print(f"Saved features to: models/customer_feature_engineered_with_segments.csv")
    
    print("\nVerification:")
    print(f"  Segment label distribution: {segmented_df['segment_label'].value_counts().to_dict()}")
    print(f"  Segment name distribution: {segmented_df['segment_name'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()

# src/train_churn.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.Mymodules.feature_engineering import build_customer_features
from src.data_loader import load_all
from src.Mymodules.modeling import define_churn_label, train_churn_model
from src.utils import save_df

def main():
    print("BLINKIT CHURN PREDICTION MODEL TRAINING")
    
    print("\n[1/5] Loading datasets...")
    data = load_all()
    customers = data['customers']
    orders = data['orders']
    order_items = data['order_items']
    feedback = data['feedback']
    
    print(f"Loaded {len(customers)} customers")
    print(f"Loaded {len(orders)} orders")
    print(f"Loaded {len(feedback)} feedback records")
    
    print("\n[2/5] Engineering features...")
    features, as_of = build_customer_features(
        customers, orders, order_items, feedback, 
        as_of_date=None, save_csv=False
    )
    print(f"Created features for {len(features)} customers")
    
    print("\n[3/5] Defining churn labels (90-day threshold)...")
    features_with_churn = define_churn_label(features, as_of_date=as_of, churn_days=90)
    
    churn_count = features_with_churn['churn'].sum()
    total_count = len(features_with_churn)
    print(f"Churned customers: {churn_count:,} ({churn_count/total_count*100:.1f}%)")
    print(f"Active customers: {total_count - churn_count:,} ({(total_count - churn_count)/total_count*100:.1f}%)")
    
    save_df(features_with_churn, "churn_data.csv")
    print(f"Saved churn dataset to: models/churn_data.csv")
    
    print("\n[4/5] Training RandomForest churn model...")
    model, metrics, splits = train_churn_model(features_with_churn, save=True)
    
    if model is None:
        print("Model training failed due to insufficient class variation")
        return
    
    print("Model training complete")
    print(f"Saved model to: models/churn_prediction_model.pkl")
    
    print("\n[5/5] Model Performance:")
    
    if 'classification_report' in metrics:
        report = metrics['classification_report']
        print("\nClassification Report:")
        print(f"  Class 0 (Active):")
        print(f"    Precision: {report['0']['precision']:.3f}")
        print(f"    Recall: {report['0']['recall']:.3f}")
        print(f"    F1-Score: {report['0']['f1-score']:.3f}")
        print(f"  Class 1 (Churned):")
        print(f"    Precision: {report['1']['precision']:.3f}")
        print(f"    Recall: {report['1']['recall']:.3f}")
        print(f"    F1-Score: {report['1']['f1-score']:.3f}")
        print(f"\n  Overall Accuracy: {report['accuracy']:.3f}")
    
    if 'roc_auc_score' in metrics and metrics['roc_auc_score'] is not None:
        print(f"  ROC-AUC Score: {metrics['roc_auc_score']:.3f}")
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  [[{cm[0][0]:>5}, {cm[0][1]:>5}]")
        print(f"   [{cm[1][0]:>5}, {cm[1][1]:>5}]]")
    
    if 'feature_importance' in metrics:
        print("\nTop 5 Most Important Features:")
        for i, feat in enumerate(metrics['feature_importance'][:5], 1):
            print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
    
    print("\nPipeline completed successfully")

if __name__ == "__main__":
    main()
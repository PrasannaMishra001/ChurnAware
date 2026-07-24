# src/train_churn.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_all
from src.Mymodules.labeling import build_snapshot_dataset, temporal_train_test_split
from src.Mymodules.modeling import train_churn_model
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

    print("\n[2/5] Building temporal snapshot dataset (leak-free labels)...")
    dataset = build_snapshot_dataset(customers, orders, order_items, feedback, horizon_days=90)

    churn_count = dataset['churn'].sum()
    total_count = len(dataset)
    print(f"Snapshot rows: {total_count:,} across {dataset['snapshot_date'].nunique()} cutoffs")
    print(f"Churn rate: {churn_count/total_count*100:.1f}%")

    save_df(dataset, "churn_data.csv")
    print("Saved churn dataset to: models/churn_data.csv")

    print("\n[3/5] Temporal train/test split (last snapshot held out)...")
    train_df, test_df = temporal_train_test_split(dataset, test_snapshots=1)
    print(f"Train: {len(train_df):,} rows from snapshots {sorted(train_df['snapshot_date'].unique())}")
    print(f"Test:  {len(test_df):,} rows from snapshot {sorted(test_df['snapshot_date'].unique())}")

    print("\n[4/5] Training RandomForest churn model...")
    model, metrics, splits = train_churn_model(train_df, test_df, save=True)

    if model is None:
        print("Model training failed due to insufficient class variation")
        return

    print("Model training complete")
    print("Saved model to: models/churn_prediction_model.pkl")

    print("\n[5/5] Model Performance (temporal holdout):")

    if 'classification_report' in metrics:
        report = metrics['classification_report']
        print("\nClassification Report:")
        print(f"  Class 0 (Retained):")
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

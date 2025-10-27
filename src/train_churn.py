# src/train_churn.py
"""
Script to prepare churn dataset and train the RandomForest churn model.
Usage:
    python src/train_churn.py
"""
from src.Mymodules.feature_engineering import build_customer_features
from src.data_loader import load_all
from src.Mymodules.modeling import define_churn_label, train_churn_model
from src.utils import save_df

def main():
    data = load_all()
    customers = data['customers']
    orders = data['orders']
    order_items = data['order_items']
    feedback = data['feedback']

    features, as_of = build_customer_features(customers, orders, order_items, feedback, as_of_date=None, save_csv=False)
    features_with_churn = define_churn_label(features, as_of_date=as_of, churn_days=90)
    save_df(features_with_churn, "churn_data.csv")

    model, metrics, splits = train_churn_model(features_with_churn, save=True)
    print("Training metrics:", metrics)

if __name__ == "__main__":
    main()

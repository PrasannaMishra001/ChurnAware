# src/data_loader.py
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def read_csv(name):
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path)

def load_all():
    customers = read_csv("blinkit_customers.csv")
    feedback = read_csv("blinkit_customer_feedback.csv")
    marketing = read_csv("blinkit_marketing_performance.csv")
    order_items = read_csv("blinkit_order_items.csv")
    orders = read_csv("blinkit_orders.csv")
    products = read_csv("blinkit_products.csv")
    return {
        "customers": customers,
        "feedback": feedback,
        "marketing": marketing,
        "order_items": order_items,
        "orders": orders,
        "products": products
    }

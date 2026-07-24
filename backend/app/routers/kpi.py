# backend/app/routers/kpi.py
import pandas as pd
from fastapi import APIRouter

from backend.app.db import engine
from backend.app.schemas import KPISummary

router = APIRouter(prefix="/api/kpi", tags=["kpi"])


@router.get("/summary", response_model=KPISummary)
def kpi_summary():
    stats = pd.read_sql(
        """SELECT
             COUNT(*) AS total_customers,
             SUM(CASE WHEN frequency > 0 THEN 1 ELSE 0 END) AS customers_with_orders,
             SUM(monetary) AS total_revenue,
             AVG(CASE WHEN frequency > 0 THEN avg_order_value END) AS avg_order_value,
             AVG(CASE WHEN frequency > 0 THEN on_time_ratio END) AS avg_on_time_ratio,
             AVG(churn_proba) AS avg_churn_probability,
             SUM(CASE WHEN churn_proba > 0.7 THEN 1 ELSE 0 END) AS high_risk_customers
           FROM customers""",
        engine,
    ).iloc[0]

    seg = pd.read_sql(
        "SELECT segment_name, COUNT(*) AS n FROM customers GROUP BY segment_name",
        engine,
    )

    return {
        "total_customers": int(stats['total_customers']),
        "customers_with_orders": int(stats['customers_with_orders']),
        "total_revenue": float(stats['total_revenue']),
        "avg_order_value": float(stats['avg_order_value']),
        "avg_on_time_ratio": float(stats['avg_on_time_ratio']),
        "avg_churn_probability": float(stats['avg_churn_probability']),
        "high_risk_customers": int(stats['high_risk_customers']),
        "segment_distribution": dict(zip(seg['segment_name'], seg['n'].astype(int))),
    }

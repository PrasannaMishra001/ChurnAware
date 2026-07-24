# backend/app/schemas.py
from typing import Optional
from pydantic import BaseModel


class CustomerSummary(BaseModel):
    customer_id: int
    customer_name: Optional[str] = None
    segment_name: str
    recency_days: float
    frequency: float
    monetary: float
    avg_order_value: float
    on_time_ratio: float
    avg_rating: float
    churn_proba: float
    recommended_action: str


class CustomerDetail(CustomerSummary):
    area: Optional[str] = None
    pincode: Optional[str] = None
    avg_delay_score: float
    avg_sentiment: float
    avg_text_sentiment: float
    negative_feedback_count: float
    count_feedback: float
    orders_per_month: float
    customer_lifespan_days: float
    expected_value: float


class CustomerPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[CustomerSummary]


class SegmentProfile(BaseModel):
    cluster: int
    segment_name: str
    size: int
    mean_frequency: float
    mean_monetary: float
    mean_on_time_ratio: float
    mean_avg_order_value: float
    mean_negative_feedback: float
    composite_score: float


class KPISummary(BaseModel):
    total_customers: int
    customers_with_orders: int
    total_revenue: float
    avg_order_value: float
    avg_on_time_ratio: float
    avg_churn_probability: float
    high_risk_customers: int
    segment_distribution: dict[str, int]


class ActionRecommendation(BaseModel):
    customer_id: int
    churn_proba: float
    recommended_action_id: int
    recommended_action: str
    q_values: dict[str, float]

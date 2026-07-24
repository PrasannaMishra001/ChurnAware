export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface KPISummary {
  total_customers: number;
  customers_with_orders: number;
  total_revenue: number;
  avg_order_value: number;
  avg_on_time_ratio: number;
  avg_churn_probability: number;
  high_risk_customers: number;
  segment_distribution: Record<string, number>;
}

export interface HistogramBin {
  bin: string;
  count: number;
}

export interface Distributions {
  churn_proba: HistogramBin[];
  monetary: HistogramBin[];
  frequency: HistogramBin[];
  on_time_ratio: HistogramBin[];
}

export interface SegmentProfile {
  cluster: number;
  segment_name: string;
  size: number;
  mean_frequency: number;
  mean_monetary: number;
  mean_on_time_ratio: number;
  mean_avg_order_value: number;
  mean_negative_feedback: number;
  composite_score: number;
}

export interface CustomerSummary {
  customer_id: number;
  customer_name: string | null;
  segment_name: string;
  recency_days: number;
  frequency: number;
  monetary: number;
  avg_order_value: number;
  on_time_ratio: number;
  avg_rating: number;
  churn_proba: number;
  recommended_action: string;
}

export interface CustomerPage {
  total: number;
  limit: number;
  offset: number;
  items: CustomerSummary[];
}

export interface PolicyResult {
  mean_net_profit: number;
  median_net_profit: number;
  retention_rate: number;
  mean_orders: number;
  mean_incentive_spend: number;
  episodes: number;
  uplift_vs_never_act: number;
}

export interface ActionSpec {
  name: string;
  hazard_mult: number;
  flat_cost: number;
  discount_rate: number;
  fulfil_cost: number;
}

export interface Recommendation {
  customer_id: number;
  churn_proba: number;
  recommended_action_id: number;
  recommended_action: string;
  q_values: Record<string, number>;
}

export interface ChurnMetrics {
  roc_auc_score: number;
  test_rows: number;
  test_churn_rate: number;
  classification_report: {
    accuracy: number;
    [key: string]: unknown;
  };
  feature_importance: { feature: string; importance: number }[];
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API ${path} failed with status ${res.status}`);
  }
  return res.json();
}

export const fmtINR = (v: number) =>
  `Rs. ${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;

export const fmtPct = (v: number, digits = 1) =>
  `${(v * 100).toFixed(digits)}%`;

export const actionLabels: Record<string, string> = {
  no_action: "No Action",
  push_notification: "Push Notification",
  free_delivery: "Free Delivery",
  discount_10pct: "10% Discount",
};

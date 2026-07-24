import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { HBarChart } from "@/components/charts/h-bar-chart";
import { TopRiskTable } from "@/components/top-risk-table";
import { getJson, fmtPct, type ChurnMetrics } from "@/lib/api";

export const dynamic = "force-dynamic";

const FEATURE_LABELS: Record<string, string> = {
  avg_order_value: "Avg order value",
  monetary: "Total spend",
  recency_days: "Recency",
  avg_text_sentiment: "Text sentiment (VADER)",
  orders_per_month: "Orders per month",
  customer_lifespan_days: "Customer lifespan",
  avg_rating: "Avg rating",
  avg_sentiment: "Labeled sentiment",
  avg_delay_score: "Delivery delay score",
  on_time_ratio: "On-time ratio",
  text_negative_count: "Negative text reviews",
  frequency: "Order frequency",
  count_feedback: "Feedback count",
  negative_feedback_count: "Negative feedback count",
};

export default async function RiskPage() {
  const metrics = await getJson<ChurnMetrics>("/api/churn/metrics");
  const importance = metrics.feature_importance.slice(0, 8).map((f) => ({
    label: FEATURE_LABELS[f.feature] ?? f.feature,
    value: Number(f.importance.toFixed(4)),
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Churn Risk
        </h1>
        <p className="text-sm text-muted-foreground">
          Random Forest churn model evaluated on an out-of-time holdout snapshot
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Model Quality</CardTitle>
            <CardDescription>
              Temporal holdout: July 2024 snapshot, {metrics.test_rows.toLocaleString()} customers
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-baseline justify-between border-b pb-2">
              <span className="text-sm text-muted-foreground">Accuracy</span>
              <span className="text-xl font-semibold tabular-nums">
                {fmtPct(metrics.classification_report.accuracy)}
              </span>
            </div>
            <div className="flex items-baseline justify-between border-b pb-2">
              <span className="text-sm text-muted-foreground">ROC-AUC</span>
              <span className="text-xl font-semibold tabular-nums">
                {metrics.roc_auc_score.toFixed(3)}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted-foreground">
                Base churn rate
              </span>
              <span className="text-xl font-semibold tabular-nums">
                {fmtPct(metrics.test_churn_rate)}
              </span>
            </div>
            <p className="pt-2 text-xs leading-relaxed text-muted-foreground">
              Labels are built from temporal snapshots: features use only data
              before each cutoff, the label is order activity in the 90 days
              after it. This removes target leakage entirely.
            </p>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>What Drives Churn</CardTitle>
            <CardDescription>
              Feature importance — text sentiment ranks above the dataset&apos;s
              own sentiment labels
            </CardDescription>
          </CardHeader>
          <CardContent>
            <HBarChart
              data={importance}
              valueName="Importance"
              format="decimal3"
              height={300}
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>At-Risk Customers</CardTitle>
          <CardDescription>
            Prioritized retention queue with recommended actions from the
            retention policy
          </CardDescription>
        </CardHeader>
        <CardContent>
          <TopRiskTable />
        </CardContent>
      </Card>
    </div>
  );
}

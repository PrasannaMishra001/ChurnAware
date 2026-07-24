import {
  AlertTriangle,
  IndianRupee,
  ShoppingCart,
  Timer,
  TrendingDown,
  Users,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { KpiCard } from "@/components/kpi-card";
import { CategoryBarChart } from "@/components/charts/category-bar-chart";
import {
  getJson,
  fmtINR,
  fmtPct,
  type KPISummary,
  type Distributions,
} from "@/lib/api";
import { SEGMENT_ORDER, SEGMENT_COLORS } from "@/lib/segments";

export const dynamic = "force-dynamic";

export default async function OverviewPage() {
  const [kpi, dist] = await Promise.all([
    getJson<KPISummary>("/api/kpi/summary"),
    getJson<Distributions>("/api/kpi/distributions"),
  ]);

  const segmentData = SEGMENT_ORDER.filter(
    (s) => s in kpi.segment_distribution
  ).map((s) => ({ label: s, value: kpi.segment_distribution[s] }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
        <p className="text-sm text-muted-foreground">
          Portfolio health for the Blinkit e-grocery customer base
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        <KpiCard
          label="Customers"
          value={kpi.total_customers.toLocaleString()}
          hint={`${kpi.customers_with_orders.toLocaleString()} with orders`}
          icon={Users}
        />
        <KpiCard
          label="Revenue"
          value={fmtINR(kpi.total_revenue)}
          icon={IndianRupee}
        />
        <KpiCard
          label="Avg Order Value"
          value={fmtINR(kpi.avg_order_value)}
          icon={ShoppingCart}
        />
        <KpiCard
          label="On-Time Ratio"
          value={fmtPct(kpi.avg_on_time_ratio)}
          icon={Timer}
        />
        <KpiCard
          label="Avg Churn Risk"
          value={fmtPct(kpi.avg_churn_probability)}
          icon={TrendingDown}
        />
        <KpiCard
          label="High Risk"
          value={kpi.high_risk_customers.toLocaleString()}
          hint="churn probability above 70%"
          icon={AlertTriangle}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Customers by Segment</CardTitle>
            <CardDescription>
              GMM segments over frequency, monetary and on-time ratio
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={segmentData}
              colors={segmentData.map((d) => SEGMENT_COLORS[d.label])}
              valueName="Customers"
              tickAngle={-20}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Churn Probability Distribution</CardTitle>
            <CardDescription>
              Model-scored churn risk across customers with orders
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={dist.churn_proba.map((b) => ({
                label: b.bin,
                value: b.count,
              }))}
              valueName="Customers"
              tickAngle={-35}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Order Frequency</CardTitle>
            <CardDescription>Orders placed per customer</CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={dist.frequency.map((b) => ({
                label: b.bin,
                value: b.count,
              }))}
              valueName="Customers"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Total Spend Distribution</CardTitle>
            <CardDescription>Lifetime monetary value (rupees)</CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={dist.monetary.map((b) => ({
                label: b.bin,
                value: b.count,
              }))}
              valueName="Customers"
              tickAngle={-35}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

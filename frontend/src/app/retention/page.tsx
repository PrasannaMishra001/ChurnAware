import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { HBarChart } from "@/components/charts/h-bar-chart";
import { CategoryBarChart } from "@/components/charts/category-bar-chart";
import { RecommendLookup } from "@/components/recommend-lookup";
import {
  getJson,
  fmtINR,
  fmtPct,
  actionLabels,
  type ActionSpec,
  type PolicyResult,
} from "@/lib/api";

export const dynamic = "force-dynamic";

const POLICY_LABELS: Record<string, string> = {
  never_act: "Never Act",
  always_discount: "Always Discount",
  rule_based: "Rule-Based",
  dqn: "Double DQN (learned)",
};

export default async function RetentionPage() {
  const [evaluation, actions] = await Promise.all([
    getJson<Record<string, PolicyResult>>("/api/retention/evaluation"),
    getJson<Record<string, ActionSpec>>("/api/retention/actions"),
  ]);

  const order = ["never_act", "always_discount", "rule_based", "dqn"];
  const profitData = order.map((p) => ({
    label: POLICY_LABELS[p],
    value: Math.round(evaluation[p].mean_net_profit),
    color: p === "dqn" ? "var(--chart-1)" : "var(--chart-muted)",
  }));
  const retentionData = order.map((p) => ({
    label: POLICY_LABELS[p],
    value: Number((evaluation[p].retention_rate * 100).toFixed(1)),
  }));
  const spendData = order.map((p) => ({
    label: POLICY_LABELS[p],
    value: Math.round(evaluation[p].mean_incentive_spend),
  }));
  const dqnUplift = evaluation.dqn.uplift_vs_never_act;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Retention Engine
        </h1>
        <p className="text-sm text-muted-foreground">
          Double DQN policy trained in a customer simulator calibrated on the
          Blinkit data — {fmtINR(dqnUplift)} profit uplift per customer versus
          never intervening
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Net Profit per Customer by Policy</CardTitle>
            <CardDescription>
              5,000 simulated customers over 39 weeks; profit net of incentive
              costs — the learned policy is highlighted
            </CardDescription>
          </CardHeader>
          <CardContent>
            <HBarChart
              data={profitData}
              valueName="Net profit"
              format="inr"
              height={220}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Retention Rate</CardTitle>
            <CardDescription>
              Customers still active at the end of the horizon (%)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={retentionData}
              valueName="Retention"
              format="pct_whole"
              tickAngle={-20}
              height={240}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Incentive Spend</CardTitle>
            <CardDescription>
              Average incentive cost per customer (rupees)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={spendData}
              valueName="Spend"
              format="inr"
              tickAngle={-20}
              height={240}
            />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Action Catalog</CardTitle>
            <CardDescription>
              Retention actions available to the policy, with simulator
              economics
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {Object.entries(actions).map(([id, spec]) => (
              <div
                key={id}
                className="flex items-center justify-between rounded-md border px-3 py-2"
              >
                <div>
                  <p className="text-sm font-medium">
                    {actionLabels[spec.name] ?? spec.name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Purchase-probability boost ×{spec.hazard_mult.toFixed(2)}
                  </p>
                </div>
                <p className="text-xs text-muted-foreground">
                  {spec.discount_rate > 0
                    ? `${fmtPct(spec.discount_rate, 0)} off order`
                    : spec.fulfil_cost > 0
                      ? `${fmtINR(spec.fulfil_cost)} per redemption`
                      : spec.flat_cost > 0
                        ? `${fmtINR(spec.flat_cost)} per send`
                        : "Free"}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Live Recommendation</CardTitle>
            <CardDescription>
              Score any customer with the trained policy and inspect its
              Q-values
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RecommendLookup />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

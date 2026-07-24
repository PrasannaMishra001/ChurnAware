"use client";

import { useState } from "react";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { HBarChart } from "@/components/charts/h-bar-chart";
import { RiskBadge } from "@/components/risk-badge";
import { API_URL, actionLabels, type Recommendation } from "@/lib/api";

export function RecommendLookup() {
  const [customerId, setCustomerId] = useState("");
  const [result, setResult] = useState<Recommendation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function lookup() {
    if (!customerId.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(
        `${API_URL}/api/retention/recommend/${customerId.trim()}`
      );
      if (res.status === 404) {
        setError("Customer not found. Pick an ID from the Customers page.");
      } else if (!res.ok) {
        setError(`Request failed with status ${res.status}.`);
      } else {
        setResult(await res.json());
      }
    } catch {
      setError("Could not reach the API.");
    } finally {
      setLoading(false);
    }
  }

  const qData = result
    ? Object.entries(result.q_values).map(([name, value]) => ({
        label: actionLabels[name] ?? name,
        value: Number(value.toFixed(3)),
        color:
          name === result.recommended_action
            ? "var(--chart-1)"
            : "var(--chart-muted)",
      }))
    : [];

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Input
          placeholder="Customer ID, e.g. 22210238"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && lookup()}
          className="max-w-xs"
          inputMode="numeric"
        />
        <Button onClick={lookup} disabled={loading}>
          <Search className="mr-1 h-4 w-4" />
          {loading ? "Scoring..." : "Recommend"}
        </Button>
      </div>

      {error ? <p className="text-sm text-destructive">{error}</p> : null}

      {result ? (
        <div className="space-y-3 rounded-lg border p-4">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm text-muted-foreground">
              Customer #{result.customer_id}
            </span>
            <RiskBadge proba={result.churn_proba} />
            <Badge>
              {actionLabels[result.recommended_action] ??
                result.recommended_action}
            </Badge>
          </div>
          <HBarChart
            data={qData}
            valueName="Q-value"
            format="decimal3"
            height={180}
          />
          <p className="text-xs text-muted-foreground">
            Q-values are the policy&apos;s estimate of discounted future profit
            for each action in this customer&apos;s current state. The
            highlighted bar is the recommended action.
          </p>
        </div>
      ) : null}
    </div>
  );
}

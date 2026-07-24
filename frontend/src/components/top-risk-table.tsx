"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { RiskBadge } from "@/components/risk-badge";
import { segmentColor, SEGMENT_ORDER } from "@/lib/segments";
import {
  API_URL,
  fmtINR,
  actionLabels,
  type CustomerSummary,
} from "@/lib/api";

export function TopRiskTable() {
  const [segment, setSegment] = useState<string>("all");
  const [rows, setRows] = useState<CustomerSummary[] | null>(null);

  const load = useCallback(async () => {
    setRows(null);
    const params = new URLSearchParams({ limit: "50" });
    if (segment !== "all") params.set("segment", segment);
    const res = await fetch(`${API_URL}/api/churn/top-risk?${params}`);
    setRows(await res.json());
  }, [segment]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Select value={segment} onValueChange={setSegment}>
          <SelectTrigger className="w-56">
            <SelectValue placeholder="Segment" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All segments</SelectItem>
            {SEGMENT_ORDER.map((s) => (
              <SelectItem key={s} value={s}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-sm text-muted-foreground">
          Top 50 customers by churn probability
        </p>
      </div>

      {rows === null ? (
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Customer</TableHead>
              <TableHead>Segment</TableHead>
              <TableHead>Risk</TableHead>
              <TableHead className="text-right">Recency (days)</TableHead>
              <TableHead className="text-right">Orders</TableHead>
              <TableHead className="text-right">Spend</TableHead>
              <TableHead>Suggested Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((c) => (
              <TableRow key={c.customer_id}>
                <TableCell>
                  <div className="font-medium">{c.customer_name}</div>
                  <div className="text-xs text-muted-foreground">
                    #{c.customer_id}
                  </div>
                </TableCell>
                <TableCell>
                  <span className="flex items-center gap-2 text-sm">
                    <span
                      className="h-2 w-2 rounded-[2px]"
                      style={{ background: segmentColor(c.segment_name) }}
                    />
                    {c.segment_name}
                  </span>
                </TableCell>
                <TableCell>
                  <RiskBadge proba={c.churn_proba} />
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {Math.round(c.recency_days)}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {Math.round(c.frequency)}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {fmtINR(c.monetary)}
                </TableCell>
                <TableCell className="text-sm">
                  {actionLabels[c.recommended_action] ?? c.recommended_action}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  );
}

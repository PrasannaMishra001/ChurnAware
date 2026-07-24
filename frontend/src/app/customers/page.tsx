"use client";

import { useCallback, useEffect, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { RiskBadge } from "@/components/risk-badge";
import { segmentColor, SEGMENT_ORDER } from "@/lib/segments";
import {
  API_URL,
  fmtINR,
  fmtPct,
  actionLabels,
  type CustomerPage,
} from "@/lib/api";

const PAGE_SIZE = 25;

export default function CustomersPage() {
  const [segment, setSegment] = useState("all");
  const [minFreq, setMinFreq] = useState("0");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<CustomerPage | null>(null);

  const load = useCallback(async () => {
    setPage(null);
    const params = new URLSearchParams({
      limit: String(PAGE_SIZE),
      offset: String(offset),
      min_frequency: minFreq || "0",
    });
    if (segment !== "all") params.set("segment", segment);
    const res = await fetch(`${API_URL}/api/customers?${params}`);
    setPage(await res.json());
  }, [segment, minFreq, offset]);

  useEffect(() => {
    load();
  }, [load]);

  const total = page?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE_SIZE, total);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Customers</h1>
        <p className="text-sm text-muted-foreground">
          Full customer base with churn scores and policy recommendations
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Customer Explorer</CardTitle>
          <CardDescription>
            Sorted by lifetime spend; filter by segment and minimum orders
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <Select
              value={segment}
              onValueChange={(v) => {
                setSegment(v);
                setOffset(0);
              }}
            >
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
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Min orders</span>
              <Input
                type="number"
                min={0}
                value={minFreq}
                onChange={(e) => {
                  setMinFreq(e.target.value);
                  setOffset(0);
                }}
                className="w-20"
              />
            </div>
            <div className="ml-auto flex items-center gap-2 text-sm text-muted-foreground">
              <span className="tabular-nums">
                {from}-{to} of {total.toLocaleString()}
              </span>
              <Button
                variant="outline"
                size="icon"
                disabled={offset === 0 || page === null}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                disabled={offset + PAGE_SIZE >= total || page === null}
                onClick={() => setOffset(offset + PAGE_SIZE)}
                aria-label="Next page"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {page === null ? (
            <div className="space-y-2">
              {Array.from({ length: 10 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Customer</TableHead>
                    <TableHead>Segment</TableHead>
                    <TableHead className="text-right">Orders</TableHead>
                    <TableHead className="text-right">Spend</TableHead>
                    <TableHead className="text-right">On-Time</TableHead>
                    <TableHead className="text-right">Rating</TableHead>
                    <TableHead>Risk</TableHead>
                    <TableHead>Suggested Action</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {page.items.map((c) => (
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
                      <TableCell className="text-right tabular-nums">
                        {Math.round(c.frequency)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {fmtINR(c.monetary)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {fmtPct(c.on_time_ratio, 0)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {c.avg_rating > 0 ? c.avg_rating.toFixed(1) : "—"}
                      </TableCell>
                      <TableCell>
                        <RiskBadge proba={c.churn_proba} />
                      </TableCell>
                      <TableCell className="text-sm">
                        {actionLabels[c.recommended_action] ??
                          c.recommended_action}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartTooltipCard } from "./chart-tooltip";
import { formatValue, type ValueFormat } from "@/lib/format";

export interface CategoryDatum {
  label: string;
  value: number;
}

export function CategoryBarChart({
  data,
  colors,
  format = "number",
  valueName = "Value",
  height = 260,
  tickAngle = 0,
}: {
  data: CategoryDatum[];
  colors?: string[];
  format?: ValueFormat;
  valueName?: string;
  height?: number;
  tickAngle?: number;
}) {
  const singleHue = !colors;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: tickAngle ? 28 : 0 }} barCategoryGap="25%">
        <CartesianGrid
          vertical={false}
          stroke="var(--chart-grid)"
          strokeWidth={1}
        />
        <XAxis
          dataKey="label"
          tickLine={false}
          axisLine={{ stroke: "var(--chart-axis)" }}
          tick={{ fill: "var(--chart-muted)", fontSize: 12 }}
          angle={tickAngle}
          textAnchor={tickAngle ? "end" : "middle"}
          interval={0}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={{ fill: "var(--chart-muted)", fontSize: 12 }}
          tickFormatter={(v: number) =>
            v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : String(v)
          }
          width={44}
        />
        <Tooltip
          cursor={{ fill: "var(--chart-grid)", opacity: 0.4 }}
          content={({ active, payload, label }) => {
            if (!active || !payload?.length) return null;
            const idx = data.findIndex((d) => d.label === label);
            return (
              <ChartTooltipCard
                title={String(label)}
                entries={[
                  {
                    name: valueName,
                    value: formatValue(payload[0].value as number, format),
                    color: singleHue
                      ? "var(--chart-1)"
                      : colors![idx % colors!.length],
                  },
                ]}
              />
            );
          }}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={48}>
          {data.map((entry, idx) => (
            <Cell
              key={entry.label}
              fill={
                singleHue ? "var(--chart-1)" : colors![idx % colors!.length]
              }
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

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

export interface HBarDatum {
  label: string;
  value: number;
  color?: string;
}

export function HBarChart({
  data,
  format = "number",
  valueName = "Value",
  height = 260,
  defaultColor = "var(--chart-1)",
}: {
  data: HBarDatum[];
  format?: ValueFormat;
  valueName?: string;
  height?: number;
  defaultColor?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 16, left: 8, bottom: 0 }}
        barCategoryGap="30%"
      >
        <CartesianGrid
          horizontal={false}
          stroke="var(--chart-grid)"
          strokeWidth={1}
        />
        <XAxis
          type="number"
          tickLine={false}
          axisLine={{ stroke: "var(--chart-axis)" }}
          tick={{ fill: "var(--chart-muted)", fontSize: 12 }}
        />
        <YAxis
          type="category"
          dataKey="label"
          tickLine={false}
          axisLine={false}
          tick={{ fill: "var(--chart-muted)", fontSize: 12 }}
          width={150}
        />
        <Tooltip
          cursor={{ fill: "var(--chart-grid)", opacity: 0.4 }}
          content={({ active, payload, label }) => {
            if (!active || !payload?.length) return null;
            const datum = data.find((d) => d.label === label);
            return (
              <ChartTooltipCard
                title={String(label)}
                entries={[
                  {
                    name: valueName,
                    value: formatValue(payload[0].value as number, format),
                    color: datum?.color ?? defaultColor,
                  },
                ]}
              />
            );
          }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={28}>
          {data.map((entry) => (
            <Cell key={entry.label} fill={entry.color ?? defaultColor} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

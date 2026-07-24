export const SEGMENT_ORDER = [
  "High-Value Champions",
  "Promising Customers",
  "Needs Attention",
  "At-Risk",
] as const;

export const SEGMENT_COLORS: Record<string, string> = {
  "High-Value Champions": "var(--chart-1)",
  "Promising Customers": "var(--chart-2)",
  "Needs Attention": "var(--chart-3)",
  "At-Risk": "var(--chart-4)",
};

export function segmentColor(name: string) {
  return SEGMENT_COLORS[name] ?? "var(--chart-5)";
}

const LEVELS = [
  { min: 0.8, label: "Critical", color: "var(--status-critical)" },
  { min: 0.6, label: "High", color: "var(--status-serious)" },
  { min: 0.4, label: "Medium", color: "var(--status-warning)" },
  { min: 0, label: "Low", color: "var(--status-good)" },
];

export function RiskBadge({ proba }: { proba: number }) {
  const level = LEVELS.find((l) => proba >= l.min) ?? LEVELS[3];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium">
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: level.color }}
      />
      {level.label}
      <span className="tabular-nums text-muted-foreground">
        {(proba * 100).toFixed(0)}%
      </span>
    </span>
  );
}

"use client";

export interface TooltipEntry {
  name: string;
  value: string;
  color?: string;
}

export function ChartTooltipCard({
  title,
  entries,
}: {
  title: string;
  entries: TooltipEntry[];
}) {
  return (
    <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
      <p className="mb-1 font-medium text-foreground">{title}</p>
      {entries.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2">
          {entry.color ? (
            <span
              className="h-2 w-2 rounded-[2px]"
              style={{ background: entry.color }}
            />
          ) : null}
          <span className="text-muted-foreground">{entry.name}</span>
          <span className="ml-auto pl-3 font-medium tabular-nums text-foreground">
            {entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}

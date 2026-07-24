export type ValueFormat =
  | "number"
  | "inr"
  | "decimal2"
  | "decimal3"
  | "pct_whole";

export function formatValue(v: number, fmt: ValueFormat = "number"): string {
  switch (fmt) {
    case "inr":
      return `Rs. ${v.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
    case "decimal2":
      return v.toFixed(2);
    case "decimal3":
      return v.toFixed(3);
    case "pct_whole":
      return `${v}%`;
    default:
      return v.toLocaleString();
  }
}

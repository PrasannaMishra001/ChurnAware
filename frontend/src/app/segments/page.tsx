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
import { CategoryBarChart } from "@/components/charts/category-bar-chart";
import { getJson, fmtINR, fmtPct, type SegmentProfile } from "@/lib/api";
import { segmentColor } from "@/lib/segments";

export const dynamic = "force-dynamic";

export default async function SegmentsPage() {
  const profiles = await getJson<SegmentProfile[]>("/api/segments");

  const byMonetary = profiles.map((p) => ({
    label: p.segment_name,
    value: Math.round(p.mean_monetary),
  }));
  const byFrequency = profiles.map((p) => ({
    label: p.segment_name,
    value: Number(p.mean_frequency.toFixed(2)),
  }));
  const colors = profiles.map((p) => segmentColor(p.segment_name));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Segments</h1>
        <p className="text-sm text-muted-foreground">
          Gaussian Mixture Model segmentation over R+FMD behavioral features
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Segment Profiles</CardTitle>
          <CardDescription>
            Mean behavioral metrics per segment, ranked by composite value
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Segment</TableHead>
                <TableHead className="text-right">Customers</TableHead>
                <TableHead className="text-right">Avg Orders</TableHead>
                <TableHead className="text-right">Avg Spend</TableHead>
                <TableHead className="text-right">On-Time Ratio</TableHead>
                <TableHead className="text-right">Avg Order Value</TableHead>
                <TableHead className="text-right">Neg. Feedback</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {profiles.map((p) => (
                <TableRow key={p.cluster}>
                  <TableCell>
                    <span className="flex items-center gap-2 font-medium">
                      <span
                        className="h-2.5 w-2.5 rounded-[3px]"
                        style={{ background: segmentColor(p.segment_name) }}
                      />
                      {p.segment_name}
                    </span>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {p.size.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {p.mean_frequency.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {fmtINR(p.mean_monetary)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {fmtPct(p.mean_on_time_ratio)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {fmtINR(p.mean_avg_order_value)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {p.mean_negative_feedback.toFixed(2)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Average Spend by Segment</CardTitle>
            <CardDescription>Mean lifetime monetary value</CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={byMonetary}
              colors={colors}
              valueName="Avg Spend"
              format="inr"
              tickAngle={-20}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Average Orders by Segment</CardTitle>
            <CardDescription>Mean order frequency</CardDescription>
          </CardHeader>
          <CardContent>
            <CategoryBarChart
              data={byFrequency}
              colors={colors}
              valueName="Avg Orders"
              format="decimal2"
              tickAngle={-20}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

import { useState } from "react";
import {
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Dashboard } from "@/api/types";
import { formatDate } from "@/lib/format";

type Point = Dashboard["progress"][number];

// Validated pair — see the note beside --color-series-* in index.css.
const SCORE = "var(--color-series-1)";
const ROLLING = "var(--color-series-2)";
const SURFACE = "#111018";

const SERIES = [
  { key: "score", label: "Answer score", color: SCORE },
  { key: "rolling", label: "5-answer average", color: ROLLING },
] as const;

interface TooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: unknown }>;
}

function ChartTooltip({ active, payload }: TooltipProps) {
  const point = payload?.[0]?.payload as Point | undefined;
  if (!active || !point) return null;
  return (
    <div className="rounded-lg border border-line-strong bg-ink/95 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1.5 text-muted">
        Answer {point.index} · {point.role} · {formatDate(point.timestamp)}
      </div>
      {SERIES.map((s) => (
        <div key={s.key} className="flex items-center gap-2">
          <span className="size-2 rounded-full" style={{ background: s.color }} />
          <span className="text-muted">{s.label}</span>
          <span className="ml-auto pl-3 font-medium tabular-nums text-fg">
            {Math.round(point[s.key])}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Score per answer over time, with a rolling average to separate trend from noise. */
export function ProgressChart({ data }: { data: Point[] }) {
  const [asTable, setAsTable] = useState(false);
  const lastIndex = data.at(-1)?.index;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-4 text-xs text-muted">
        {SERIES.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className="h-0.5 w-4 rounded-full" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        <button onClick={() => setAsTable((t) => !t)} className="ml-auto text-faint hover:text-fg">
          {asTable ? "Show chart" : "Show as table"}
        </button>
      </div>

      {asTable ? (
        <div className="max-h-64 overflow-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-faint">
              <tr>
                <th className="py-1.5 font-medium">#</th>
                <th className="font-medium">Role</th>
                <th className="text-right font-medium">Score</th>
                <th className="text-right font-medium">5-answer avg</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {data.map((p) => (
                <tr key={p.index} className="border-t border-line">
                  <td className="py-1.5 text-muted">{p.index}</td>
                  <td className="text-muted">{p.role}</td>
                  <td className="text-right">{Math.round(p.score)}</td>
                  <td className="text-right">{Math.round(p.rolling)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-64" role="img" aria-label={`Score trend across ${data.length} answers`}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 40, bottom: 0, left: -18 }}>
              <CartesianGrid stroke="rgb(255 255 255 / 0.06)" vertical={false} />
              <XAxis
                dataKey="index"
                tick={{ fill: "#6d6a82", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
                minTickGap={20}
              />
              <YAxis
                domain={[0, 100]}
                ticks={[0, 25, 50, 75, 100]}
                tick={{ fill: "#6d6a82", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                content={ChartTooltip}
                cursor={{ stroke: "rgb(255 255 255 / 0.25)", strokeWidth: 1 }}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke={SCORE}
                strokeWidth={1.5}
                strokeOpacity={0.55}
                dot={{ r: 4, fill: SCORE, stroke: SURFACE, strokeWidth: 2 }}
                activeDot={{ r: 6, stroke: SURFACE, strokeWidth: 2 }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="rolling"
                stroke={ROLLING}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 5, stroke: SURFACE, strokeWidth: 2 }}
                isAnimationActive={false}
              >
                {/* One direct label, on the latest point only. */}
                <LabelList
                  dataKey="rolling"
                  content={({ x, y, value, index }) =>
                    data[index as number]?.index === lastIndex ? (
                      <text x={Number(x) + 8} y={Number(y) + 4} fill="#eceaf6" fontSize={11}>
                        {Math.round(Number(value))}
                      </text>
                    ) : null
                  }
                />
              </Line>
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

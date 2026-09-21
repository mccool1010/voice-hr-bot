import { ArrowDownRight, ArrowUpRight, Mic, Minus } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";

import { useDashboard } from "@/api/queries";
import { ProgressChart } from "@/components/charts/ProgressChart";
import { Button, Card, EmptyState, ErrorNote, ScoreBar, SectionTitle, Spinner } from "@/components/ui";
import { useAuth } from "@/features/auth/auth-context";
import { COMPETENCY_LABEL, formatDelta, formatScore, RUBRIC_LABEL } from "@/lib/format";

const DELIVERY_ROWS: { key: string; label: string; format: (v: number) => string; hint: string }[] = [
  { key: "words_per_minute", label: "Speaking pace", format: (v) => `${Math.round(v)} wpm`, hint: "110–160 is easiest to follow" },
  { key: "filler_rate", label: "Filler words", format: (v) => `${v.toFixed(1)} / 100 words`, hint: "“um”, “like”, “you know”" },
  { key: "hedge_rate", label: "Hedging", format: (v) => `${v.toFixed(1)} / 100 words`, hint: "“I think”, “sort of”, “maybe”" },
  { key: "long_pause_rate", label: "Long pauses", format: (v) => `${v.toFixed(1)} / min`, hint: "gaps over 0.7 s" },
  { key: "star_coverage", label: "STAR structure", format: (v) => `${Math.round(v * 100)}%`, hint: "situation · action · result" },
];

export default function DashboardPage() {
  const { user } = useAuth();
  const { data, isLoading, error } = useDashboard();

  if (isLoading) return <Spinner label="Crunching your history…" />;
  if (error || !data) return <ErrorNote>Couldn't load your dashboard.</ErrorNote>;

  const { overview } = data;
  const firstName = user?.display_name.split(" ")[0];

  if (overview.total_answers === 0) {
    return (
      <Card>
        <EmptyState
          icon={<Mic className="size-5" />}
          title={`Welcome${firstName ? `, ${firstName}` : ""}`}
          action={
            <Link to="/app/new">
              <Button size="lg">Start your first interview</Button>
            </Link>
          }
        >
          Once you've answered a few questions, this page tracks your scores, your
          strongest and weakest competencies, and how your delivery is changing.
        </EmptyState>
      </Card>
    );
  }

  return (
    <div className="space-y-6 animate-fade-up">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Your progress</h1>
        <p className="mt-1 text-sm text-muted">
          {overview.total_answers} answers across {overview.total_interviews} interview
          {overview.total_interviews === 1 ? "" : "s"}
          {overview.roles_practised.length > 0 && ` · ${overview.roles_practised.join(", ")}`}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile label="Average score" value={formatScore(overview.average_score)} />
        <StatTile label="Best answer" value={formatScore(overview.best_score)} />
        <StatTile label="Most recent" value={formatScore(overview.latest_score)} />
        <StatTile
          label="Trend"
          value={formatDelta(overview.improvement) ?? "—"}
          icon={<TrendIcon delta={overview.improvement} />}
          hint={
            overview.improvement == null
              ? "Needs 6+ answers"
              : data.percentile != null
                ? `Ahead of ${Math.round(data.percentile)}% of candidates`
                : "Recent third vs first third"
          }
        />
      </div>

      <Card>
        <SectionTitle hint="Each point is one answer">Score over time</SectionTitle>
        <ProgressChart data={data.progress} />
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <SectionTitle hint="Average per answer">By competency</SectionTitle>
          <div className="space-y-4">
            {data.competencies.map((c) => (
              <ScoreBar
                key={c.competency}
                label={COMPETENCY_LABEL[c.competency]}
                value={c.score}
                hint={[`${c.answers} ans`, formatDelta(c.delta)]
                  .filter((part) => part && part !== "0")
                  .join(" · ")}
              />
            ))}
          </div>
          {data.competencies.length > 1 && (
            <p className="mt-5 text-xs text-muted">
              Focus next on{" "}
              <span className="text-fg">
                {COMPETENCY_LABEL[data.competencies.at(-1)!.competency]}
              </span>{" "}
              — it's your lowest-scoring area.
            </p>
          )}
        </Card>

        <Card>
          <SectionTitle hint="How your answers are built">Rubric</SectionTitle>
          <div className="space-y-4">
            {(Object.keys(RUBRIC_LABEL) as (keyof typeof RUBRIC_LABEL)[]).map((dim) => (
              <ScoreBar key={dim} label={RUBRIC_LABEL[dim]} value={data.rubric[dim]} />
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <SectionTitle hint="Spoken answers only">Delivery</SectionTitle>
        {Object.keys(data.delivery).length === 0 ? (
          <p className="text-sm text-muted">
            Answer out loud to see pace, pauses and filler-word analysis.
          </p>
        ) : (
          <dl className="grid gap-x-8 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
            {DELIVERY_ROWS.filter((row) => typeof data.delivery[row.key] === "number").map((row) => (
              <div key={row.key}>
                <dt className="text-xs text-muted">{row.label}</dt>
                <dd className="mt-0.5 text-lg font-medium tabular-nums">
                  {row.format(data.delivery[row.key] as number)}
                  {row.key === "words_per_minute" && data.delivery.pace_verdict && (
                    <span className="ml-2 text-xs font-normal text-muted">
                      ({String(data.delivery.pace_verdict)})
                    </span>
                  )}
                </dd>
                <dd className="text-xs text-faint">{row.hint}</dd>
              </div>
            ))}
          </dl>
        )}
      </Card>
    </div>
  );
}

function StatTile({
  label,
  value,
  hint,
  icon,
}: {
  label: string;
  value: string;
  hint?: string;
  icon?: ReactNode;
}) {
  return (
    <Card className="p-4 sm:p-5">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 flex items-center gap-1.5 text-3xl font-semibold tabular-nums">
        {value}
        {icon}
      </div>
      {hint && <div className="mt-1 text-xs text-faint">{hint}</div>}
    </Card>
  );
}

function TrendIcon({ delta }: { delta: number | null }) {
  if (delta == null) return null;
  if (delta > 1) return <ArrowUpRight className="size-5 text-good" aria-label="improving" />;
  if (delta < -1) return <ArrowDownRight className="size-5 text-bad" aria-label="declining" />;
  return <Minus className="size-5 text-muted" aria-label="steady" />;
}

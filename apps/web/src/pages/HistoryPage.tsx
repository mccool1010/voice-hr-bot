import { ChevronRight, History } from "lucide-react";
import { Link } from "react-router";

import { useInterviews } from "@/api/queries";
import type { InterviewStatus } from "@/api/types";
import { Badge, Button, Card, EmptyState, ErrorNote, Spinner } from "@/components/ui";
import { formatDate, formatScore, PERSONA_META, SENIORITY_LABEL } from "@/lib/format";

const STATUS: Record<InterviewStatus, { label: string; tone: "good" | "accent" | "warn" | "neutral" }> = {
  completed: { label: "Completed", tone: "good" },
  in_progress: { label: "In progress", tone: "accent" },
  abandoned: { label: "Ended early", tone: "warn" },
  created: { label: "Not started", tone: "neutral" },
};

export function HistoryPage() {
  const { data, isLoading, error } = useInterviews();

  if (isLoading) return <Spinner />;
  if (error) return <ErrorNote>Couldn't load your interviews.</ErrorNote>;

  return (
    <div className="space-y-6 animate-fade-up">
      <h1 className="text-2xl font-semibold tracking-tight">History</h1>

      {!data?.length ? (
        <Card>
          <EmptyState
            icon={<History className="size-5" />}
            title="No interviews yet"
            action={
              <Link to="/app/new">
                <Button>Start one</Button>
              </Link>
            }
          />
        </Card>
      ) : (
        <Card className="p-0 sm:p-0">
          <ul className="divide-y divide-line">
            {data.map((item) => {
              const href =
                item.status === "in_progress" ? `/app/interview/${item.id}` : `/app/report/${item.id}`;
              return (
                <li key={item.id}>
                  <Link to={href} className="flex items-center gap-4 px-5 py-4 transition hover:bg-panel">
                    <div className="w-12 text-center text-xl font-semibold tabular-nums">
                      {formatScore(item.overall_score)}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="line-clamp-2 font-medium break-words">{item.role}</div>
                      <div className="mt-0.5 text-xs text-muted">
                        {formatDate(item.created_at)} · {SENIORITY_LABEL[item.seniority]} ·{" "}
                        {PERSONA_META[item.persona].label} · {item.answered_questions} answered
                      </div>
                    </div>
                    <Badge tone={STATUS[item.status].tone}>
                      {item.status === "in_progress" ? "Resume" : STATUS[item.status].label}
                    </Badge>
                    <ChevronRight className="size-4 text-faint" aria-hidden />
                  </Link>
                </li>
              );
            })}
          </ul>
        </Card>
      )}
    </div>
  );
}

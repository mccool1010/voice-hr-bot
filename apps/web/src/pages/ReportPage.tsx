import { clsx } from "clsx";
import { ChevronDown, Lightbulb, RotateCcw, Sparkles, TrendingUp } from "lucide-react";
import { useState } from "react";
import { Link, Navigate, useParams } from "react-router";

import { useInterview } from "@/api/queries";
import type { Turn } from "@/api/types";
import { Badge, Button, Card, ErrorNote, ScoreBar, ScoreRing, SectionTitle, Spinner } from "@/components/ui";
import {
  COMPETENCY_LABEL,
  formatDate,
  formatScore,
  PERSONA_META,
  RUBRIC_LABEL,
  SENIORITY_LABEL,
} from "@/lib/format";

export default function ReportPage() {
  const { id } = useParams<{ id: string }>();
  const { data: interview, isLoading, error } = useInterview(id);

  if (isLoading) return <Spinner label="Loading report…" />;
  if (error || !interview) return <ErrorNote>Couldn't load this interview.</ErrorNote>;
  if (interview.status === "in_progress") return <Navigate to={`/app/interview/${id}`} replace />;

  const report = interview.report;
  const answered = interview.turns.filter((t) => t.answer_text);

  return (
    <div className="space-y-6 animate-fade-up">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs text-faint">{formatDate(interview.created_at)}</p>
          <h1 className="text-2xl font-semibold tracking-tight">{interview.role}</h1>
          <div className="mt-2 flex flex-wrap gap-2">
            <Badge>{SENIORITY_LABEL[interview.seniority]}</Badge>
            <Badge>{PERSONA_META[interview.persona].label} interviewer</Badge>
            <Badge>{answered.length} answers</Badge>
            {interview.status === "abandoned" && <Badge tone="warn">Ended early</Badge>}
          </div>
        </div>
        <Link to="/app/new">
          <Button variant="secondary" icon={<RotateCcw className="size-4" />}>
            Practise again
          </Button>
        </Link>
      </div>

      {report ? (
        <>
          <div className="grid gap-6 lg:grid-cols-[auto_1fr]">
            <Card className="flex flex-col items-center justify-center gap-2 lg:px-10">
              <ScoreRing score={report.overall_score} size={168} stroke={12} />
              <span className="text-xs text-muted">Overall</span>
            </Card>
            <Card>
              <SectionTitle>Summary</SectionTitle>
              <p className="leading-relaxed text-fg/90">{report.summary}</p>
              {report.recommended_focus && (
                <div className="mt-5 flex gap-3 rounded-xl border border-accent/30 bg-accent/10 p-4">
                  <Lightbulb className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
                  <div>
                    <div className="text-xs font-medium text-accent">Practise this next</div>
                    <p className="mt-0.5 text-sm">{report.recommended_focus}</p>
                  </div>
                </div>
              )}
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <FeedbackList title="What went well" icon={<Sparkles className="size-4 text-good" />}
              items={report.strengths} />
            <FeedbackList title="What to improve" icon={<TrendingUp className="size-4 text-warn" />}
              items={report.improvements} />
          </div>

          {Object.keys(report.competency_scores).length > 0 && (
            <Card>
              <SectionTitle>By competency</SectionTitle>
              <div className="grid gap-x-10 gap-y-4 sm:grid-cols-2">
                {Object.entries(report.competency_scores)
                  .sort(([, a], [, b]) => b - a)
                  .map(([competency, score]) => (
                    <ScoreBar
                      key={competency}
                      label={COMPETENCY_LABEL[competency as keyof typeof COMPETENCY_LABEL] ?? competency}
                      value={score}
                    />
                  ))}
              </div>
            </Card>
          )}
        </>
      ) : (
        <Card>
          <p className="text-sm text-muted">
            This interview ended before a report was written. Per-answer scores are below.
          </p>
        </Card>
      )}

      <Card>
        <SectionTitle hint="Tap an answer for the breakdown">Transcript</SectionTitle>
        {answered.length === 0 ? (
          <p className="text-sm text-muted">No questions were answered.</p>
        ) : (
          <ol className="divide-y divide-line">
            {answered.map((turn, i) => (
              <TranscriptRow key={turn.id} turn={turn} number={i + 1} />
            ))}
          </ol>
        )}
      </Card>

      <p className="text-center text-xs text-faint">
        Questions and feedback by {interview.llm_provider} · {interview.llm_model}
      </p>
    </div>
  );
}

function FeedbackList({ title, icon, items }: { title: string; icon: React.ReactNode; items: string[] }) {
  return (
    <Card>
      <SectionTitle>
        <span className="flex items-center gap-2">
          {icon} {title}
        </span>
      </SectionTitle>
      {items.length ? (
        <ul className="space-y-3">
          {items.map((item) => (
            <li key={item} className="flex gap-3 text-sm leading-relaxed">
              <span className="mt-2 size-1.5 shrink-0 rounded-full bg-line-strong" />
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted">Nothing noted.</p>
      )}
    </Card>
  );
}

function TranscriptRow({ turn, number }: { turn: Turn; number: number }) {
  const [open, setOpen] = useState(false);
  const score = turn.score;

  return (
    <li className="py-4">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-start gap-4 text-left"
      >
        <span className="w-10 shrink-0 pt-0.5 text-right text-lg font-semibold tabular-nums">
          {formatScore(score?.blended_score)}
        </span>
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-faint">
            <span>Q{number}</span>
            <span>·</span>
            <span>{COMPETENCY_LABEL[turn.competency]}</span>
            {turn.kind === "follow_up" && <Badge tone="warn">Follow-up</Badge>}
            {turn.audio_duration_s != null && <span>· spoken, {Math.round(turn.audio_duration_s)}s</span>}
          </div>
          <p className="font-medium">{turn.question}</p>
          <p className={clsx("mt-1.5 text-sm text-muted", !open && "line-clamp-2")}>
            {turn.answer_text}
          </p>
        </div>
        <ChevronDown
          className={clsx("mt-1 size-4 shrink-0 text-faint transition", open && "rotate-180")}
          aria-hidden
        />
      </button>

      {open && score && (
        <div className="mt-4 grid gap-6 pl-14 sm:grid-cols-2">
          <div className="space-y-3">
            {(Object.keys(RUBRIC_LABEL) as (keyof typeof RUBRIC_LABEL)[]).map((dim) => (
              <ScoreBar key={dim} label={RUBRIC_LABEL[dim]} value={score[dim]} />
            ))}
          </div>
          <dl className="space-y-2 text-sm">
            <SignalRow label="Rubric (LLM)" value={formatScore(score.llm_score)} />
            <SignalRow label="Delivery model (PyTorch)" value={formatScore(score.model_score)} />
            <SignalRow
              label="On-topic similarity"
              value={score.relevance == null ? "—" : `${Math.round(score.relevance * 100)}%`}
            />
            {turn.transcription_confidence != null && (
              <SignalRow
                label="Transcription confidence"
                value={`${Math.round(turn.transcription_confidence * 100)}%`}
              />
            )}
            <p className="pt-1 text-xs text-faint">
              The final score blends the rubric and delivery model, then scales down answers that
              drift off the question.
            </p>
          </dl>
        </div>
      )}
    </li>
  );
}

function SignalRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-line pb-2">
      <dt className="text-muted">{label}</dt>
      <dd className="tabular-nums">{value}</dd>
    </div>
  );
}

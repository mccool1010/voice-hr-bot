import { ArrowRight, AudioLines, BrainCircuit, ChartLine, GitBranch, Mic } from "lucide-react";
import { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router";

import { ApiError } from "@/api/client";
import { Button, ErrorNote } from "@/components/ui";
import { useAuth } from "@/features/auth/auth-context";

const FEATURES = [
  {
    icon: GitBranch,
    title: "An interviewer that adapts",
    body: "A LangGraph state machine plans the agenda, then probes thin answers with follow-ups — like a real interviewer, not a question list.",
  },
  {
    icon: AudioLines,
    title: "Speak naturally",
    body: "Whisper transcribes on the server, so it works in every browser — and measures your pace, pauses and filler words as you talk.",
  },
  {
    icon: BrainCircuit,
    title: "Scored three ways",
    body: "An LLM rubric, a PyTorch model over delivery features, and semantic relevance between question and answer — blended, not trusted blindly.",
  },
  {
    icon: ChartLine,
    title: "See yourself improve",
    body: "Every answer is stored. Track your trend, your weakest competency, and how your delivery changes interview to interview.",
  },
];

export function LandingPage() {
  const { user, loading, demo } = useAuth();
  const navigate = useNavigate();
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Redirect visitors who arrive already signed in — but not mid-demo-start,
  // or this redirect races the navigation to /app/new below and wins.
  if (!loading && user && !starting) return <Navigate to="/app" replace />;

  const tryDemo = async () => {
    setStarting(true);
    setError(null);
    try {
      await demo();
      navigate("/app/new");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't start the demo.");
      setStarting(false);
    }
  };

  return (
    <div className="mx-auto flex min-h-dvh max-w-6xl flex-col px-4">
      <header className="flex h-16 items-center justify-between">
        <div className="flex items-center gap-2 font-semibold">
          <span className="grid size-8 place-items-center rounded-lg bg-gradient-to-br from-accent to-accent-2">
            <Mic className="size-4 text-white" aria-hidden />
          </span>
          Voice HR
        </div>
        <Link to="/login" className="text-sm text-muted hover:text-fg">
          Sign in
        </Link>
      </header>

      <section className="flex flex-1 flex-col items-center justify-center py-16 text-center">
        <span className="mb-6 inline-flex items-center gap-2 rounded-full border border-line bg-panel px-3 py-1 text-xs text-muted animate-fade-up">
          <span className="size-1.5 rounded-full bg-good" /> Version 2 — rebuilt from the ground up
        </span>
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight sm:text-6xl animate-fade-up">
          Practise the interview <span className="text-gradient">before it counts.</span>
        </h1>
        <p className="mt-6 max-w-xl text-base text-muted sm:text-lg animate-fade-up [animation-delay:80ms]">
          Talk to an AI interviewer that listens, asks sharp follow-ups, and tells you
          exactly which answers landed and why.
        </p>

        <div className="mt-10 flex flex-col items-center gap-3 sm:flex-row animate-fade-up [animation-delay:160ms]">
          <Button size="lg" onClick={tryDemo} loading={starting} icon={<ArrowRight className="size-4" />}>
            Try it now — no sign-up
          </Button>
          <Link to="/login?mode=register">
            <Button size="lg" variant="secondary">
              Create an account
            </Button>
          </Link>
        </div>
        {error && (
          <div className="mt-6 w-full max-w-md">
            <ErrorNote>{error}</ErrorNote>
          </div>
        )}
      </section>

      <section className="grid gap-4 pb-16 sm:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map(({ icon: Icon, title, body }) => (
          <div key={title} className="glass p-5">
            <Icon className="mb-3 size-5 text-accent" aria-hidden />
            <h3 className="mb-1.5 font-medium">{title}</h3>
            <p className="text-sm leading-relaxed text-muted">{body}</p>
          </div>
        ))}
      </section>

      <footer className="border-t border-line py-6 text-center text-xs text-faint">
        React · TypeScript · Tailwind — FastAPI · PostgreSQL · LangGraph · PyTorch · Whisper
      </footer>
    </div>
  );
}

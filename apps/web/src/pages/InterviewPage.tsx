import { clsx } from "clsx";
import {
  CheckCircle2,
  Keyboard,
  LoaderCircle,
  Mic,
  Square,
  Volume2,
  VolumeX,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, Navigate, useLocation, useNavigate, useParams } from "react-router";

import { api } from "@/api/client";
import { useCapabilities, useInterview, useInvalidateAfterInterview } from "@/api/queries";
import type { NextQuestion } from "@/api/types";
import {
  Badge,
  Button,
  Card,
  ErrorNote,
  ScoreBar,
  ScoreRing,
  SectionTitle,
  Select,
  Spinner,
} from "@/components/ui";
import { isBusy, type ScoredAnswer } from "@/features/interview/session";
import { useInterviewSocket } from "@/hooks/useInterviewSocket";
import { MAX_RECORDING_S, useRecorder } from "@/hooks/useRecorder";
import { useSpeech } from "@/hooks/useSpeech";
import { COMPETENCY_LABEL, formatDuration, formatScore, RUBRIC_LABEL } from "@/lib/format";

type Mode = "voice" | "text";

interface LocationState {
  opening?: string | null;
  question?: NextQuestion;
}

export function InterviewPage() {
  const { id } = useParams<{ id: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const { data: interview, isLoading } = useInterview(id);
  const { data: caps } = useCapabilities();
  const invalidate = useInvalidateAfterInterview();

  const { state, sendText, beginAudio, sendAudioChunk, endAudio, dismissError } =
    useInterviewSocket(interview?.status === "in_progress" ? id : undefined);
  const speech = useSpeech();

  const voiceAvailable = caps?.speech_to_text ?? false;
  const [mode, setMode] = useState<Mode>("voice");
  const effectiveMode: Mode = voiceAvailable ? mode : "text";
  const [draft, setDraft] = useState("");
  const [deviceId, setDeviceId] = useState<string>("");

  const recorder = useRecorder({
    deviceId: deviceId || undefined,
    onChunk: sendAudioChunk,
    onStop: () => endAudio(),
  });

  // ─── Speak each new question once ──────────────────────────────────────────
  const spokenTurn = useRef<number | null>(null);
  const opening = (location.state as LocationState | null)?.opening;
  useEffect(() => {
    const q = state.question;
    if (!q || spokenTurn.current === q.turnIndex) return;
    const isFirst = spokenTurn.current === null && q.turnIndex === 0;
    spokenTurn.current = q.turnIndex;
    void speech.speak(isFirst && opening ? `${opening} ${q.text}` : q.text);
  }, [state.question, opening, speech]);

  // ─── Finish ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (state.phase === "finished" && id) invalidate(id);
  }, [state.phase, id, invalidate]);

  const busy = isBusy(state.phase);
  const recording = recorder.status === "recording";
  const canAnswer =
    state.phase === "answering" && state.connection === "open" && !busy && !recording;

  const startRecording = useCallback(async () => {
    speech.cancel();
    if (!beginAudio()) return;
    await recorder.start();
  }, [beginAudio, recorder, speech]);

  const submitText = () => {
    if (sendText(draft)) setDraft("");
  };

  // Space bar toggles recording, except while typing in a field.
  useEffect(() => {
    if (effectiveMode !== "voice") return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.code !== "Space" || target.closest("input, textarea, select, button")) return;
      event.preventDefault();
      if (recording) recorder.stop();
      else if (canAnswer) void startRecording();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canAnswer, effectiveMode, recorder, recording, startRecording]);

  const endInterview = async () => {
    if (!id || !window.confirm("End this interview now? Answers so far are kept.")) return;
    speech.cancel();
    recorder.stop();
    await api.interviews.abandon(id).catch(() => undefined);
    invalidate(id);
    navigate("/app/history");
  };

  // ─── Guards ────────────────────────────────────────────────────────────────
  if (isLoading || !interview) return <Spinner label="Loading interview…" />;
  // Arriving at an already-finished interview goes straight to its report. One
  // that finished in this view stays put so the completion screen is seen.
  if (interview.status === "completed" && state.phase !== "finished") {
    return <Navigate to={`/app/report/${id}`} replace />;
  }
  if (interview.status === "abandoned" && state.phase !== "finished") {
    return (
      <Card className="mx-auto max-w-md text-center">
        <p className="mb-4 text-muted">This interview was ended early.</p>
        <Link to="/app/new">
          <Button>Start a new one</Button>
        </Link>
      </Card>
    );
  }

  const answered = state.history.length;

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">{interview.role}</h1>
            <p className="text-sm text-muted">
              {answered} answered · about {interview.target_questions} topics planned
            </p>
          </div>
          <div className="flex items-center gap-2">
            {speech.supported && (
              <Button variant="ghost" size="sm" onClick={speech.toggle}
                icon={speech.enabled ? <Volume2 className="size-4" /> : <VolumeX className="size-4" />}>
                {speech.enabled ? "Voice on" : "Muted"}
              </Button>
            )}
            {state.phase !== "finished" && (
              <Button variant="danger" size="sm" onClick={endInterview}>
                End interview
              </Button>
            )}
          </div>
        </div>

        {state.connection === "reconnecting" && (
          <div className="flex items-center gap-2 rounded-xl border border-warn/30 bg-warn/10 px-4 py-2.5 text-sm text-warn">
            <WifiOff className="size-4" aria-hidden />
            Connection lost — reconnecting. Your progress is saved.
          </div>
        )}

        {state.phase === "finished" && state.report ? (
          <FinishedCard score={state.report.overall_score} id={id!} />
        ) : (
          <>
            <QuestionCard
              text={state.question?.text}
              competency={state.question?.competency}
              followUp={state.question?.kind === "follow_up"}
              speaking={speech.speaking}
              onSkipSpeech={speech.cancel}
            />

            <Card>
              <div className="mb-5 flex items-center justify-between">
                <SectionTitle>Your answer</SectionTitle>
                {voiceAvailable && (
                  <div className="flex rounded-lg bg-black/30 p-0.5 text-xs">
                    {(["voice", "text"] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => setMode(m)}
                        disabled={recording}
                        className={clsx(
                          "flex items-center gap-1.5 rounded-md px-2.5 py-1 transition",
                          effectiveMode === m ? "bg-panel-strong text-fg" : "text-muted",
                        )}
                      >
                        {m === "voice" ? <Mic className="size-3.5" /> : <Keyboard className="size-3.5" />}
                        {m === "voice" ? "Speak" : "Type"}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {effectiveMode === "voice" ? (
                <VoiceAnswer
                  recording={recording}
                  level={recorder.level}
                  elapsed={recorder.elapsed}
                  disabled={!canAnswer && !recording}
                  onStart={() => void startRecording()}
                  onStop={recorder.stop}
                  devices={recorder.devices}
                  deviceId={deviceId}
                  onDevice={setDeviceId}
                />
              ) : (
                <div>
                  {!voiceAvailable && (
                    <p className="mb-3 text-xs text-faint">
                      Voice answers aren't enabled on this server, so type your answer.
                    </p>
                  )}
                  <textarea
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canAnswer) submitText();
                    }}
                    rows={6}
                    placeholder="Answer as you would out loud. Ctrl + Enter to submit."
                    className="w-full resize-y rounded-xl border border-line bg-black/30 p-3.5 text-sm outline-none focus:border-accent/60"
                    disabled={busy}
                  />
                  <div className="mt-3 flex items-center justify-between">
                    <span className="text-xs text-faint">
                      {draft.trim() ? draft.trim().split(/\s+/).length : 0} words
                    </span>
                    <Button onClick={submitText} disabled={!canAnswer || !draft.trim()}>
                      Submit answer
                    </Button>
                  </div>
                </div>
              )}

              <ProgressLine phase={state.phase} transcript={state.pendingAnswer}
                confidence={state.transcriptConfidence} />

              {(state.error || recorder.error) && (
                <div className="mt-4">
                  <ErrorNote onDismiss={state.error ? dismissError : undefined}>
                    {state.error ?? recorder.error}
                  </ErrorNote>
                </div>
              )}
            </Card>
          </>
        )}
      </div>

      <LiveScorecard history={state.history} />
    </div>
  );
}

// ─── Pieces ───────────────────────────────────────────────────────────────────

function QuestionCard({
  text,
  competency,
  followUp,
  speaking,
  onSkipSpeech,
}: {
  text?: string;
  competency?: keyof typeof COMPETENCY_LABEL;
  followUp: boolean;
  speaking: boolean;
  onSkipSpeech: () => void;
}) {
  return (
    <Card className="relative overflow-hidden">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/60 to-transparent" />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {competency && <Badge tone="accent">{COMPETENCY_LABEL[competency]}</Badge>}
        {followUp && <Badge tone="warn">Follow-up</Badge>}
        {speaking && (
          <button onClick={onSkipSpeech} className="ml-auto flex items-center gap-1.5 text-xs text-muted hover:text-fg">
            <Volume2 className="size-3.5 animate-pulse" aria-hidden /> Speaking — skip
          </button>
        )}
      </div>
      {text ? (
        <p key={text} className="text-lg leading-relaxed sm:text-xl animate-fade-up">
          {text}
        </p>
      ) : (
        <div className="flex items-center gap-2 text-muted">
          <LoaderCircle className="size-4 animate-spin" aria-hidden /> Connecting to your interviewer…
        </div>
      )}
    </Card>
  );
}

function VoiceAnswer({
  recording,
  level,
  elapsed,
  disabled,
  onStart,
  onStop,
  devices,
  deviceId,
  onDevice,
}: {
  recording: boolean;
  level: number;
  elapsed: number;
  disabled: boolean;
  onStart: () => void;
  onStop: () => void;
  devices: MediaDeviceInfo[];
  deviceId: string;
  onDevice: (id: string) => void;
}) {
  const remaining = MAX_RECORDING_S - elapsed;
  return (
    <div className="flex flex-col items-center gap-4 py-2">
      <button
        onClick={recording ? onStop : onStart}
        disabled={disabled}
        aria-label={recording ? "Stop recording and submit" : "Start recording"}
        className={clsx(
          "relative grid size-24 place-items-center rounded-full transition disabled:opacity-40",
          recording
            ? "bg-bad text-white animate-pulse-ring"
            : "bg-gradient-to-br from-accent to-accent-2 text-white shadow-xl shadow-accent/30 hover:scale-105",
        )}
        style={recording ? { transform: `scale(${1 + level * 0.12})` } : undefined}
      >
        {recording ? <Square className="size-8 fill-current" /> : <Mic className="size-9" />}
      </button>

      <div className="text-center text-sm" aria-live="polite">
        {recording ? (
          <span className="tabular-nums">
            Recording {formatDuration(elapsed)}
            {remaining < 20 && <span className="ml-2 text-warn">{Math.ceil(remaining)}s left</span>}
            <span className="block text-xs text-faint">Tap again, or press Space, when you're done</span>
          </span>
        ) : (
          <span className="text-muted">
            Tap to answer <span className="text-faint">· or press Space</span>
          </span>
        )}
      </div>

      {devices.length > 1 && !recording && (
        <div className="w-full max-w-xs">
          <Select aria-label="Microphone" value={deviceId} onChange={(e) => onDevice(e.target.value)}>
            <option value="">Default microphone</option>
            {devices.map((d, i) => (
              <option key={d.deviceId || i} value={d.deviceId}>
                {d.label || `Microphone ${i + 1}`}
              </option>
            ))}
          </Select>
        </div>
      )}
    </div>
  );
}

function ProgressLine({
  phase,
  transcript,
  confidence,
}: {
  phase: string;
  transcript: string | null;
  confidence: number | null;
}) {
  if (phase !== "transcribing" && phase !== "scoring") return null;
  return (
    <div className="mt-5 space-y-3 border-t border-line pt-4" aria-live="polite">
      {transcript && (
        <blockquote className="border-l-2 border-accent/50 pl-3 text-sm italic text-muted">
          “{transcript}”
          {confidence != null && confidence < 0.6 && (
            <span className="mt-1 block not-italic text-xs text-warn">
              Low transcription confidence — speak a little closer to the mic next time.
            </span>
          )}
        </blockquote>
      )}
      <div className="flex items-center gap-2 text-sm text-muted">
        <LoaderCircle className="size-4 animate-spin text-accent" aria-hidden />
        {phase === "transcribing" ? "Transcribing your answer…" : "Evaluating your answer…"}
      </div>
    </div>
  );
}

function LiveScorecard({ history }: { history: ScoredAnswer[] }) {
  const last = history.at(-1);
  return (
    <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
      <Card>
        <SectionTitle hint="Last answer">Live score</SectionTitle>
        {last ? (
          <div className="space-y-4">
            <div className="flex justify-center">
              <ScoreRing score={last.score.blended_score} size={128} />
            </div>
            {(Object.keys(RUBRIC_LABEL) as (keyof typeof RUBRIC_LABEL)[]).map((dim) => (
              <ScoreBar key={dim} label={RUBRIC_LABEL[dim]} value={last.score[dim]} />
            ))}
            {last.score.relevance != null && (
              <p className="text-xs text-faint">
                Question–answer similarity: {Math.round(last.score.relevance * 100)}%
              </p>
            )}
          </div>
        ) : (
          <p className="text-sm text-muted">Scores appear here after your first answer.</p>
        )}
      </Card>

      {history.length > 0 && (
        <Card>
          <SectionTitle>So far</SectionTitle>
          <ol className="space-y-2">
            {history.map((entry) => (
              <li key={entry.turnIndex} className="flex items-center gap-3 text-sm">
                <span className="w-8 shrink-0 text-right font-medium tabular-nums">
                  {formatScore(entry.score.blended_score)}
                </span>
                <span className="truncate text-muted" title={entry.question}>
                  {entry.question}
                </span>
              </li>
            ))}
          </ol>
        </Card>
      )}
    </aside>
  );
}

function FinishedCard({ score, id }: { score: number; id: string }) {
  return (
    <Card className="flex flex-col items-center gap-5 py-10 text-center animate-fade-up">
      <CheckCircle2 className="size-8 text-good" aria-hidden />
      <div>
        <h2 className="text-xl font-semibold">Interview complete</h2>
        <p className="mt-1 text-sm text-muted">Your feedback report is ready.</p>
      </div>
      <ScoreRing score={score} size={150} />
      <Link to={`/app/report/${id}`}>
        <Button size="lg">See your full report</Button>
      </Link>
    </Card>
  );
}

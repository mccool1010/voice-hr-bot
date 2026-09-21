import { clsx } from "clsx";
import { FileText, Play, Trash2, Upload } from "lucide-react";
import { useRef, useState, type FormEvent } from "react";
import { useNavigate } from "react-router";

import { ApiError } from "@/api/client";
import {
  useCapabilities,
  useCreateInterview,
  useDeleteResume,
  useResumes,
  useUploadResume,
} from "@/api/queries";
import type { Persona, Seniority } from "@/api/types";
import { Badge, Button, Card, ErrorNote, Input, Label, SectionTitle } from "@/components/ui";
import { PERSONA_META, SENIORITY_LABEL } from "@/lib/format";

const ROLE_SUGGESTIONS = [
  "Software Engineer",
  "Data Scientist",
  "Product Manager",
  "ML Engineer",
  "Frontend Developer",
  "DevOps Engineer",
];

export function SetupPage() {
  const navigate = useNavigate();
  const create = useCreateInterview();

  const [role, setRole] = useState("");
  const [seniority, setSeniority] = useState<Seniority>("mid");
  const [persona, setPersona] = useState<Persona>("neutral");
  const [questions, setQuestions] = useState(5);
  const [resumeId, setResumeId] = useState<string | null>(null);

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    create.mutate(
      { role: role.trim(), seniority, persona, target_questions: questions, resume_id: resumeId },
      {
        onSuccess: (first) =>
          navigate(`/app/interview/${first.interview_id}`, {
            state: { opening: first.opening_remark, question: first },
          }),
      },
    );
  };

  return (
    <form onSubmit={onSubmit} className="mx-auto max-w-3xl space-y-6 animate-fade-up">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New interview</h1>
        <p className="mt-1 text-sm text-muted">
          Set the scene. The interviewer plans its questions around what you choose here.
        </p>
      </div>

      <Card>
        <SectionTitle>Role</SectionTitle>
        <Label htmlFor="role">What role are you practising for?</Label>
        <Input
          id="role"
          required
          minLength={2}
          maxLength={160}
          placeholder="e.g. Senior Backend Engineer at a fintech"
          value={role}
          onChange={(e) => setRole(e.target.value)}
          autoFocus
        />
        <div className="mt-3 flex flex-wrap gap-2">
          {ROLE_SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => setRole(suggestion)}
              className="rounded-full border border-line px-3 py-1 text-xs text-muted transition hover:border-accent/50 hover:text-fg"
            >
              {suggestion}
            </button>
          ))}
        </div>

        <div className="mt-6">
          <Label htmlFor="seniority-mid">Level</Label>
          <div className="grid grid-cols-5 gap-1 rounded-xl bg-black/30 p-1" role="radiogroup">
            {(Object.keys(SENIORITY_LABEL) as Seniority[]).map((level) => (
              <button
                key={level}
                id={`seniority-${level}`}
                type="button"
                role="radio"
                aria-checked={seniority === level}
                onClick={() => setSeniority(level)}
                className={clsx(
                  "rounded-lg py-2 text-xs transition sm:text-sm",
                  seniority === level ? "bg-panel-strong text-fg" : "text-muted hover:text-fg",
                )}
              >
                {SENIORITY_LABEL[level]}
              </button>
            ))}
          </div>
        </div>
      </Card>

      <Card>
        <SectionTitle>Interviewer</SectionTitle>
        <div className="grid gap-3 sm:grid-cols-2" role="radiogroup" aria-label="Interviewer style">
          {(Object.keys(PERSONA_META) as Persona[]).map((key) => (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={persona === key}
              onClick={() => setPersona(key)}
              className={clsx(
                "rounded-xl border p-4 text-left transition",
                persona === key
                  ? "border-accent/60 bg-accent/10"
                  : "border-line hover:border-line-strong",
              )}
            >
              <div className="text-sm font-medium">{PERSONA_META[key].label}</div>
              <div className="mt-1 text-xs text-muted">{PERSONA_META[key].blurb}</div>
            </button>
          ))}
        </div>

        <div className="mt-6">
          <div className="mb-1.5 flex items-baseline justify-between">
            <label htmlFor="questions" className="text-xs font-medium text-muted">
              Topics to cover
            </label>
            <span className="text-sm tabular-nums">
              {questions} <span className="text-faint">· ~{questions * 3} min</span>
            </span>
          </div>
          <input
            id="questions"
            type="range"
            min={3}
            max={10}
            value={questions}
            onChange={(e) => setQuestions(Number(e.target.value))}
            className="w-full accent-[var(--color-accent)]"
          />
          <p className="mt-1 text-xs text-faint">
            Follow-up questions are added on top when an answer deserves a deeper look.
          </p>
        </div>
      </Card>

      <ResumePicker selected={resumeId} onSelect={setResumeId} />

      {create.error && (
        <ErrorNote>
          {create.error instanceof ApiError ? create.error.message : "Couldn't start the interview."}
        </ErrorNote>
      )}

      <div className="flex items-center justify-between gap-4">
        <p className="text-xs text-faint">
          Planning takes a few seconds — the interviewer writes its agenda first.
        </p>
        <Button
          type="submit"
          size="lg"
          loading={create.isPending}
          disabled={role.trim().length < 2}
          icon={<Play className="size-4" />}
        >
          {create.isPending ? "Preparing questions…" : "Start interview"}
        </Button>
      </div>
    </form>
  );
}

function ResumePicker({
  selected,
  onSelect,
}: {
  selected: string | null;
  onSelect: (id: string | null) => void;
}) {
  const { data: resumes = [] } = useResumes();
  const { data: caps } = useCapabilities();
  const upload = useUploadResume();
  const remove = useDeleteResume();
  const inputRef = useRef<HTMLInputElement>(null);
  const maxMb = Math.round((caps?.max_resume_bytes ?? 5 * 1024 * 1024) / 1024 / 1024);

  const onFile = (file: File | undefined) => {
    if (!file) return;
    upload.mutate(file, { onSuccess: (resume) => onSelect(resume.id) });
  };

  return (
    <Card>
      <SectionTitle hint="Optional">Your CV</SectionTitle>
      <p className="mb-4 text-sm text-muted">
        Add a CV and the interviewer will ask about your actual projects instead of generic ones.
      </p>

      {resumes.length > 0 && (
        <ul className="mb-4 space-y-2">
          {resumes.map((resume) => (
            <li key={resume.id}>
              <div
                className={clsx(
                  "flex items-start gap-3 rounded-xl border p-3 transition",
                  selected === resume.id ? "border-accent/60 bg-accent/10" : "border-line",
                )}
              >
                <button
                  type="button"
                  onClick={() => onSelect(selected === resume.id ? null : resume.id)}
                  className="flex flex-1 items-start gap-3 text-left"
                  aria-pressed={selected === resume.id}
                >
                  <FileText className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">{resume.filename}</div>
                    <div className="text-xs text-muted">{resume.extracted.headline}</div>
                    {resume.extracted.skills.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {resume.extracted.skills.slice(0, 8).map((skill) => (
                          <Badge key={skill}>{skill}</Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (selected === resume.id) onSelect(null);
                    remove.mutate(resume.id);
                  }}
                  className="text-faint hover:text-bad"
                  aria-label={`Delete ${resume.filename}`}
                >
                  <Trash2 className="size-4" aria-hidden />
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md,application/pdf,text/plain"
        className="hidden"
        onChange={(e) => {
          onFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <Button
        type="button"
        variant="secondary"
        loading={upload.isPending}
        onClick={() => inputRef.current?.click()}
        icon={<Upload className="size-4" />}
      >
        {upload.isPending ? "Reading your CV…" : "Upload a CV"}
      </Button>
      <span className="ml-3 text-xs text-faint">PDF, DOCX or text · up to {maxMb} MB</span>

      {upload.error && (
        <div className="mt-3">
          <ErrorNote>
            {upload.error instanceof ApiError ? upload.error.message : "Upload failed."}
          </ErrorNote>
        </div>
      )}
    </Card>
  );
}

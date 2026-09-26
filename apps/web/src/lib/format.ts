import type { Competency, Persona, Seniority } from "@/api/types";

export const COMPETENCY_LABEL: Record<Competency, string> = {
  communication: "Communication",
  technical_depth: "Technical depth",
  problem_solving: "Problem solving",
  ownership: "Ownership",
  collaboration: "Collaboration",
  culture_fit: "Culture fit",
};

export const SENIORITY_LABEL: Record<Seniority, string> = {
  intern: "Intern",
  junior: "Junior",
  mid: "Mid-level",
  senior: "Senior",
  staff: "Staff+",
};

export const PERSONA_META: Record<Persona, { label: string; blurb: string }> = {
  friendly: { label: "Friendly", blurb: "Warm and encouraging. Good for a first run." },
  neutral: { label: "Neutral", blurb: "A typical recruiter screen. Even and professional." },
  skeptical: { label: "Skeptical", blurb: "Probes vague claims and asks for evidence." },
  time_pressed: { label: "Time-pressed", blurb: "Brisk, short questions, quick transitions." },
};

export const RUBRIC_LABEL = {
  structure: "Structure",
  specificity: "Specificity",
  clarity: "Clarity",
  depth: "Depth",
} as const;

export type Band = "excellent" | "strong" | "adequate" | "weak" | "poor";

/** Same bands the LLM evaluator is anchored to, so the UI and the grader agree. */
export function band(score: number): Band {
  if (score >= 90) return "excellent";
  if (score >= 75) return "strong";
  if (score >= 60) return "adequate";
  if (score >= 40) return "weak";
  return "poor";
}

export const BAND_LABEL: Record<Band, string> = {
  excellent: "Excellent",
  strong: "Strong",
  adequate: "Adequate",
  weak: "Needs work",
  poor: "Poor",
};

export const BAND_TEXT: Record<Band, string> = {
  excellent: "text-good",
  strong: "text-good",
  adequate: "text-accent-2",
  weak: "text-warn",
  poor: "text-bad",
};

export const BAND_STROKE: Record<Band, string> = {
  excellent: "#34d399",
  strong: "#34d399",
  adequate: "#4f9cf9",
  weak: "#fbbf24",
  poor: "#f87171",
};

export const formatScore = (value: number | null | undefined) =>
  value == null ? "—" : Math.round(value).toString();

export const formatDate = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

export const formatDuration = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

export const formatDelta = (delta: number | null | undefined) => {
  if (delta == null) return null;
  const rounded = Math.round(delta);
  return rounded > 0 ? `+${rounded}` : String(rounded);
};

/** "like ×2, um ×1" — the API already sorts fillers by count; none gives "None". */
export const formatFillers = (counts: Record<string, number> | null | undefined) => {
  const entries = Object.entries(counts ?? {});
  return entries.length ? entries.map(([word, n]) => `${word} ×${n}`).join(", ") : "None";
};

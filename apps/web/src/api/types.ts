// Wire types mirroring the FastAPI Pydantic schemas in apps/api/app/schemas.
// Kept hand-written and small; the OpenAPI schema at /openapi.json is the
// source of truth if these ever drift.

export type Seniority = "intern" | "junior" | "mid" | "senior" | "staff";
export type Persona = "friendly" | "neutral" | "skeptical" | "time_pressed";
export type InterviewStatus = "created" | "in_progress" | "completed" | "abandoned";
export type TurnKind = "opening" | "main" | "follow_up" | "closing";
export type Competency =
  | "communication"
  | "technical_depth"
  | "problem_solving"
  | "ownership"
  | "collaboration"
  | "culture_fit";

export interface User {
  id: string;
  email: string;
  display_name: string;
  is_demo: boolean;
  created_at: string;
}

export interface TokenPair {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: User;
}

export interface Score {
  blended_score: number;
  model_score: number;
  llm_score: number | null;
  relevance: number | null;
  structure: number | null;
  specificity: number | null;
  clarity: number | null;
  depth: number | null;
  scorer_version?: string;
}

export interface Turn {
  id: string;
  index: number;
  kind: TurnKind;
  competency: Competency;
  question: string;
  answer_text: string | null;
  audio_duration_s: number | null;
  transcription_confidence: number | null;
  answered_at: string | null;
  score: Score | null;
  /** Filler words heard in the answer, most frequent first. Display only; not part of any score. */
  filler_words?: Record<string, number>;
}

export interface Report {
  overall_score: number;
  competency_scores: Record<string, number>;
  delivery_metrics: Record<string, number | string | null>;
  summary: string;
  strengths: string[];
  improvements: string[];
  recommended_focus: string | null;
}

export interface Interview {
  id: string;
  role: string;
  seniority: Seniority;
  persona: Persona;
  status: InterviewStatus;
  target_questions: number;
  llm_provider: string;
  llm_model: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  turns: Turn[];
  report: Report | null;
}

export interface InterviewSummary {
  id: string;
  role: string;
  seniority: Seniority;
  persona: Persona;
  status: InterviewStatus;
  created_at: string;
  completed_at: string | null;
  answered_questions: number;
  overall_score: number | null;
}

export interface NextQuestion {
  interview_id: string;
  question: string;
  competency: Competency;
  kind: TurnKind;
  turn_index: number;
  finished: boolean;
  opening_remark?: string | null;
}

export interface AnswerResult {
  score: Score;
  next: NextQuestion | null;
  report: Report | null;
  finished: boolean;
}

export interface InterviewCreate {
  role: string;
  seniority: Seniority;
  persona: Persona;
  target_questions: number;
  resume_id?: string | null;
}

export interface ResumeFacts {
  headline: string;
  years_experience: number;
  skills: string[];
  projects: string[];
  domains: string[];
}

export interface Resume {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  created_at: string;
  extracted: ResumeFacts;
  excerpt: string;
}

export interface Capabilities {
  llm: { provider: string; model: string };
  speech_to_text: boolean;
  semantic_relevance: boolean;
  trained_scorer: boolean;
  scorer_version: string;
  max_audio_bytes: number;
  max_resume_bytes: number;
}

export interface Dashboard {
  overview: {
    total_interviews: number;
    total_answers: number;
    average_score: number | null;
    best_score: number | null;
    latest_score: number | null;
    improvement: number | null;
    roles_practised: string[];
  };
  progress: { index: number; timestamp: string; score: number; rolling: number; role: string }[];
  competencies: {
    competency: Competency;
    score: number;
    answers: number;
    consistency: number | null;
    delta: number | null;
  }[];
  rubric: Record<"structure" | "specificity" | "clarity" | "depth", number | null>;
  delivery: Record<string, number | string>;
  percentile: number | null;
}

// ─── WebSocket protocol (see apps/api/app/routers/ws.py) ───────────────────────

export type ServerMessage =
  | { type: "question"; question: string; competency: Competency; kind: TurnKind; turn_index: number }
  | { type: "status"; stage: "transcribing" | "scoring" }
  | { type: "transcript"; text: string; confidence: number | null; duration_s: number }
  | ({ type: "score" } & Score)
  | ({ type: "report" } & Report)
  | { type: "error"; message: string }
  | { type: "pong" };

export type ClientMessage =
  | { type: "answer"; text: string }
  | { type: "audio_start" }
  | { type: "audio_end" }
  | { type: "ping" };

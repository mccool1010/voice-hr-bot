import type {
  AnswerResult,
  Capabilities,
  Dashboard,
  Interview,
  InterviewCreate,
  InterviewSummary,
  NextQuestion,
  Resume,
  TokenPair,
  User,
} from "./types";

// Empty means same-origin: the Vite dev proxy locally, and the single-container
// deploy where FastAPI serves this bundle. Set VITE_API_BASE_URL only when the
// API lives on a different host.
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const TOKEN_KEY = "voicehr.token";

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(status: number, message: string, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

// ─── Token storage ────────────────────────────────────────────────────────────
// localStorage can throw (private mode, blocked storage); a failure there must
// degrade to "signed out", never crash the app.

export const tokenStore = {
  get(): string | null {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set(token: string): void {
    try {
      localStorage.setItem(TOKEN_KEY, token);
    } catch {
      /* storage unavailable — the session lasts until reload */
    }
  },
  clear(): void {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* nothing to clear */
    }
  },
};

// ─── Transport ────────────────────────────────────────────────────────────────

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = tokenStore.get();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }

  if (response.status === 204) return undefined as T;

  const body: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    if (response.status === 401) tokenStore.clear();
    throw new ApiError(response.status, errorMessage(body, response.status), errorCode(body));
  }
  return body as T;
}

function errorMessage(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    // FastAPI 422s carry a list of field errors.
    if (Array.isArray(detail) && detail[0]?.msg) {
      const first = detail[0] as { msg: string; loc?: (string | number)[] };
      const field = first.loc?.at(-1);
      return field ? `${String(field)}: ${first.msg}` : first.msg;
    }
  }
  if (status >= 500) return "Something went wrong on our side. Please try again.";
  return `Request failed (${status}).`;
}

function errorCode(body: unknown): string | undefined {
  if (body && typeof body === "object" && "code" in body) {
    return String((body as { code: unknown }).code);
  }
  return undefined;
}

const json = (data: unknown) => JSON.stringify(data);

// ─── Endpoints ────────────────────────────────────────────────────────────────

export const api = {
  auth: {
    register: (email: string, password: string, display_name: string) =>
      request<TokenPair>("/auth/register", {
        method: "POST",
        body: json({ email, password, display_name }),
      }),
    login: (email: string, password: string) =>
      request<TokenPair>("/auth/login", { method: "POST", body: json({ email, password }) }),
    demo: () => request<TokenPair>("/auth/demo", { method: "POST" }),
    me: () => request<User>("/auth/me"),
  },

  interviews: {
    create: (payload: InterviewCreate) =>
      request<NextQuestion>("/interviews", { method: "POST", body: json(payload) }),
    list: () => request<InterviewSummary[]>("/interviews"),
    get: (id: string) => request<Interview>(`/interviews/${id}`),
    current: (id: string) => request<NextQuestion>(`/interviews/${id}/current`),
    answer: (id: string, answer: string) =>
      request<AnswerResult>(`/interviews/${id}/answer`, {
        method: "POST",
        body: json({ answer }),
      }),
    abandon: (id: string) => request<Interview>(`/interviews/${id}/abandon`, { method: "POST" }),
  },

  resumes: {
    list: () => request<Resume[]>("/resumes"),
    upload: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return request<Resume>("/resumes", { method: "POST", body: form });
    },
    remove: (id: string) => request<void>(`/resumes/${id}`, { method: "DELETE" }),
  },

  analytics: {
    dashboard: () => request<Dashboard>("/analytics/dashboard"),
  },

  capabilities: () => request<Capabilities>("/capabilities"),
};

/** WebSocket URL for a live interview. The token rides in the query string
 *  because the browser WebSocket API cannot set an Authorization header. */
export function interviewSocketUrl(interviewId: string): string {
  const token = encodeURIComponent(tokenStore.get() ?? "");
  const base = API_BASE || window.location.origin;
  const wsBase = base.replace(/^http/, "ws");
  return `${wsBase}/ws/interviews/${interviewId}?token=${token}`;
}

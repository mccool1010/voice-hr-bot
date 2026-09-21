import type { Competency, Report, Score, ServerMessage, TurnKind } from "@/api/types";

// The live interview as a pure reducer. The WebSocket hook only translates
// socket events into actions; every state transition lives here, which is what
// makes the flow unit-testable without a socket or a browser.

export type Connection = "connecting" | "open" | "reconnecting" | "closed";

export type Phase =
  | "waiting" //      connected, no question yet
  | "answering" //    question on screen, candidate's turn
  | "transcribing" // audio uploaded, Whisper running
  | "scoring" //      answer text known, evaluator running
  | "finished";

export interface CurrentQuestion {
  text: string;
  competency: Competency;
  kind: TurnKind;
  turnIndex: number;
}

export interface ScoredAnswer {
  turnIndex: number;
  question: string;
  answer: string;
  competency: Competency;
  score: Score;
}

export interface SessionState {
  connection: Connection;
  phase: Phase;
  question: CurrentQuestion | null;
  /** Answer text for the turn in flight — typed, or Whisper's transcript. */
  pendingAnswer: string | null;
  transcriptConfidence: number | null;
  history: ScoredAnswer[];
  lastScore: Score | null;
  report: Report | null;
  error: string | null;
}

export type SessionAction =
  | { type: "server"; message: ServerMessage }
  | { type: "connection"; state: Connection }
  | { type: "submitted_text"; text: string }
  | { type: "submitted_audio" }
  | { type: "dismiss_error" };

export const initialSession: SessionState = {
  connection: "connecting",
  phase: "waiting",
  question: null,
  pendingAnswer: null,
  transcriptConfidence: null,
  history: [],
  lastScore: null,
  report: null,
  error: null,
};

export function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case "connection":
      // A closed socket after the report is the normal end, not a failure.
      if (state.phase === "finished") return { ...state, connection: action.state };
      return { ...state, connection: action.state };

    case "submitted_text":
      return { ...state, phase: "scoring", pendingAnswer: action.text, error: null };

    case "submitted_audio":
      return { ...state, phase: "transcribing", pendingAnswer: null, error: null };

    case "dismiss_error":
      return { ...state, error: null };

    case "server":
      return applyServerMessage(state, action.message);
  }
}

function applyServerMessage(state: SessionState, message: ServerMessage): SessionState {
  switch (message.type) {
    case "question": {
      // A reconnect replays the pending question; ignore it if already showing.
      if (state.question?.turnIndex === message.turn_index && state.phase === "answering") {
        return state;
      }
      return {
        ...state,
        phase: "answering",
        question: {
          text: message.question,
          competency: message.competency,
          kind: message.kind,
          turnIndex: message.turn_index,
        },
        pendingAnswer: null,
        transcriptConfidence: null,
        error: null,
      };
    }

    case "status":
      return { ...state, phase: message.stage };

    case "transcript":
      return {
        ...state,
        phase: "scoring",
        pendingAnswer: message.text,
        transcriptConfidence: message.confidence,
      };

    case "score": {
      const { type: _type, ...score } = message;
      const entry: ScoredAnswer | null = state.question
        ? {
            turnIndex: state.question.turnIndex,
            question: state.question.text,
            answer: state.pendingAnswer ?? "",
            competency: state.question.competency,
            score,
          }
        : null;
      return {
        ...state,
        lastScore: score,
        history: entry ? [...state.history, entry] : state.history,
      };
    }

    case "report": {
      const { type: _type, ...report } = message;
      return { ...state, phase: "finished", report, error: null };
    }

    case "error":
      return {
        ...state,
        error: message.message,
        // Hand the turn back so the candidate can retry; never un-finish.
        phase: state.phase === "finished" ? "finished" : state.question ? "answering" : "waiting",
      };

    case "pong":
      return state;
  }
}

export const isBusy = (phase: Phase) => phase === "transcribing" || phase === "scoring";

import { describe, expect, it } from "vitest";

import type { Score, ServerMessage } from "@/api/types";

import { initialSession, isBusy, sessionReducer, type SessionState } from "./session";

const question = (turn: number, text = `Question ${turn}?`): ServerMessage => ({
  type: "question",
  question: text,
  competency: "problem_solving",
  kind: turn === 0 ? "opening" : "main",
  turn_index: turn,
});

const score: Score = {
  blended_score: 72,
  model_score: 65,
  llm_score: 76,
  relevance: 0.61,
  structure: 70,
  specificity: 68,
  clarity: 80,
  depth: 66,
};

const run = (...actions: Parameters<typeof sessionReducer>[1][]) =>
  actions.reduce<SessionState>(sessionReducer, initialSession);

const server = (message: ServerMessage) => ({ type: "server" as const, message });

describe("sessionReducer", () => {
  it("shows the first question and hands the turn to the candidate", () => {
    const state = run({ type: "connection", state: "open" }, server(question(0)));
    expect(state.phase).toBe("answering");
    expect(state.question?.text).toBe("Question 0?");
  });

  it("walks a typed answer through scoring to the next question", () => {
    let state = run(server(question(0)), { type: "submitted_text", text: "I built it." });
    expect(state.phase).toBe("scoring");
    expect(state.pendingAnswer).toBe("I built it.");

    state = sessionReducer(state, server({ type: "score", ...score }));
    expect(state.history).toHaveLength(1);
    expect(state.history[0]).toMatchObject({ answer: "I built it.", question: "Question 0?" });

    state = sessionReducer(state, server(question(1)));
    expect(state.phase).toBe("answering");
    expect(state.pendingAnswer).toBeNull();
  });

  it("walks a spoken answer through transcription", () => {
    let state = run(server(question(0)), { type: "submitted_audio" });
    expect(state.phase).toBe("transcribing");

    state = sessionReducer(state, server({ type: "transcript", text: "Spoken words", confidence: 0.9, duration_s: 4 }));
    expect(state.phase).toBe("scoring");
    expect(state.pendingAnswer).toBe("Spoken words");

    state = sessionReducer(state, server({ type: "score", ...score }));
    expect(state.history[0]?.answer).toBe("Spoken words");
  });

  it("ignores a question replayed on reconnect", () => {
    const before = run(server(question(2)));
    const after = sessionReducer(before, server(question(2)));
    expect(after).toBe(before);
  });

  it("hands the turn back after an error so the candidate can retry", () => {
    const state = run(
      server(question(0)),
      { type: "submitted_audio" },
      server({ type: "error", message: "Nothing could be heard." }),
    );
    expect(state.phase).toBe("answering");
    expect(state.error).toBe("Nothing could be heard.");
  });

  it("finishes on the report and stays finished through later errors", () => {
    let state = run(
      server(question(0)),
      server({
        type: "report",
        overall_score: 74,
        competency_scores: {},
        delivery_metrics: {},
        summary: "Good.",
        strengths: [],
        improvements: [],
        recommended_focus: null,
      }),
    );
    expect(state.phase).toBe("finished");
    state = sessionReducer(state, server({ type: "error", message: "late" }));
    expect(state.phase).toBe("finished");
  });

  it("treats transcribing and scoring as busy", () => {
    expect(isBusy("transcribing")).toBe(true);
    expect(isBusy("scoring")).toBe(true);
    expect(isBusy("answering")).toBe(false);
  });
});

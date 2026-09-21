import { useCallback, useEffect, useReducer, useRef } from "react";

import { interviewSocketUrl } from "@/api/client";
import type { ClientMessage, ServerMessage } from "@/api/types";
import { initialSession, sessionReducer } from "@/features/interview/session";

const PING_INTERVAL_MS = 25_000;
const MAX_BACKOFF_MS = 8_000;

/**
 * Owns the interview WebSocket.
 *
 * Reconnects with exponential backoff; the server replays the outstanding
 * question on connect, and the LangGraph checkpoint means no progress is lost.
 * Stops reconnecting once the report has arrived.
 */
export function useInterviewSocket(interviewId: string | undefined) {
  const [state, dispatch] = useReducer(sessionReducer, initialSession);
  const socketRef = useRef<WebSocket | null>(null);
  const finishedRef = useRef(false);

  useEffect(() => {
    finishedRef.current = state.phase === "finished";
  }, [state.phase]);

  useEffect(() => {
    if (!interviewId) return;

    let attempt = 0;
    let disposed = false;
    let reconnectTimer = 0;
    let pingTimer = 0;

    const connect = () => {
      dispatch({ type: "connection", state: attempt === 0 ? "connecting" : "reconnecting" });
      const socket = new WebSocket(interviewSocketUrl(interviewId));
      socket.binaryType = "arraybuffer";
      socketRef.current = socket;

      socket.onopen = () => {
        attempt = 0;
        dispatch({ type: "connection", state: "open" });
        pingTimer = window.setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "ping" }));
        }, PING_INTERVAL_MS);
      };

      socket.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        try {
          dispatch({ type: "server", message: JSON.parse(event.data) as ServerMessage });
        } catch {
          /* ignore malformed frames */
        }
      };

      socket.onclose = (event) => {
        window.clearInterval(pingTimer);
        if (disposed) return;
        // 1008 = policy violation: bad token or not our interview. Don't retry.
        if (finishedRef.current || event.code === 1008) {
          dispatch({ type: "connection", state: "closed" });
          return;
        }
        attempt += 1;
        dispatch({ type: "connection", state: "reconnecting" });
        const delay = Math.min(MAX_BACKOFF_MS, 500 * 2 ** attempt);
        reconnectTimer = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      disposed = true;
      window.clearTimeout(reconnectTimer);
      window.clearInterval(pingTimer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [interviewId]);

  const send = useCallback((message: ClientMessage | Blob) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(message instanceof Blob ? message : JSON.stringify(message));
    return true;
  }, []);

  const sendText = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return false;
      const ok = send({ type: "answer", text: trimmed });
      if (ok) dispatch({ type: "submitted_text", text: trimmed });
      return ok;
    },
    [send],
  );

  const beginAudio = useCallback(() => send({ type: "audio_start" }), [send]);
  const sendAudioChunk = useCallback((chunk: Blob) => send(chunk), [send]);
  const endAudio = useCallback(() => {
    const ok = send({ type: "audio_end" });
    if (ok) dispatch({ type: "submitted_audio" });
    return ok;
  }, [send]);

  const dismissError = useCallback(() => dispatch({ type: "dismiss_error" }), []);

  return { state, sendText, beginAudio, sendAudioChunk, endAudio, dismissError };
}

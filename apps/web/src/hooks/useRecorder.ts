import { useCallback, useEffect, useRef, useState } from "react";

/** Hard stop so a forgotten recording cannot grow without bound. */
export const MAX_RECORDING_S = 120;
const CHUNK_MS = 250;

// Opus in WebM everywhere it exists; Safari only records MP4/AAC. Whisper's
// decoder sniffs the container, so either works server-side.
const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type));
}

export type RecorderStatus = "idle" | "requesting" | "recording" | "stopping" | "error";

interface Options {
  deviceId?: string;
  /** Called with each audio chunk as it is produced — used to stream to the server. */
  onChunk?: (chunk: Blob) => void;
  /** Called once when recording has fully stopped and every chunk was emitted. */
  onStop?: () => void;
}

export function useRecorder({ deviceId, onChunk, onStop }: Options = {}) {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const frameRef = useRef<number>(0);
  const timerRef = useRef<number>(0);

  // Callbacks change identity every render; keep the latest without re-subscribing.
  const onChunkRef = useRef(onChunk);
  const onStopRef = useRef(onStop);
  useEffect(() => {
    onChunkRef.current = onChunk;
    onStopRef.current = onStop;
  });

  const supported = typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined";

  const refreshDevices = useCallback(() => {
    void navigator.mediaDevices
      ?.enumerateDevices?.()
      .then((all) => setDevices(all.filter((d) => d.kind === "audioinput")));
  }, []);

  // Subscribe to device changes; state is only ever set from the async callback.
  useEffect(() => {
    const media = navigator.mediaDevices;
    if (!media?.enumerateDevices) return;
    let active = true;
    const update = () => {
      void media.enumerateDevices().then((all) => {
        if (active) setDevices(all.filter((d) => d.kind === "audioinput"));
      });
    };
    update();
    media.addEventListener?.("devicechange", update);
    return () => {
      active = false;
      media.removeEventListener?.("devicechange", update);
    };
  }, []);

  const teardown = useCallback(() => {
    cancelAnimationFrame(frameRef.current);
    window.clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void audioCtxRef.current?.close().catch(() => undefined);
    audioCtxRef.current = null;
    recorderRef.current = null;
    setLevel(0);
  }, []);

  useEffect(() => teardown, [teardown]);

  const stop = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      setStatus("stopping");
      recorder.stop();
    }
  }, []);

  const start = useCallback(async () => {
    if (!supported) {
      setError("This browser can't record audio. Type your answer instead.");
      setStatus("error");
      return false;
    }
    setError(null);
    setStatus("requesting");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
    } catch (err) {
      const denied = err instanceof DOMException && err.name === "NotAllowedError";
      setError(
        denied
          ? "Microphone access was blocked. Allow it in your browser's site settings."
          : "Couldn't open the microphone. Check it's connected and not in use.",
      );
      setStatus("error");
      return false;
    }
    streamRef.current = stream;
    // Device labels are only exposed after permission is granted.
    refreshDevices();

    // Live input level for the visualiser — RMS of the time-domain signal.
    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buffer = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(buffer);
      let sum = 0;
      for (const sample of buffer) {
        const v = (sample - 128) / 128;
        sum += v * v;
      }
      setLevel(Math.min(1, Math.sqrt(sum / buffer.length) * 3.2));
      frameRef.current = requestAnimationFrame(tick);
    };
    tick();

    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) onChunkRef.current?.(event.data);
    };
    recorder.onstop = () => {
      teardown();
      setStatus("idle");
      onStopRef.current?.();
    };

    const startedAt = performance.now();
    setElapsed(0);
    timerRef.current = window.setInterval(() => {
      const seconds = (performance.now() - startedAt) / 1000;
      setElapsed(seconds);
      if (seconds >= MAX_RECORDING_S) stop();
    }, 200);

    recorder.start(CHUNK_MS);
    setStatus("recording");
    return true;
  }, [deviceId, refreshDevices, stop, supported, teardown]);

  return { status, error, level, elapsed, devices, supported, start, stop };
}

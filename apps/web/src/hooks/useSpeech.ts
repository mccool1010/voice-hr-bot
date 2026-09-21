import { useCallback, useEffect, useRef, useState } from "react";

// Speech *synthesis* is well supported across browsers (unlike recognition,
// which is why transcription moved to the server). Voices load asynchronously,
// so the preferred voice is re-resolved whenever the list changes.

const PREFERRED = [/natural/i, /neural/i, /google us english/i, /samantha/i, /aria/i, /jenny/i];

function pickVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  const english = voices.filter((v) => v.lang.toLowerCase().startsWith("en"));
  for (const pattern of PREFERRED) {
    const match = english.find((v) => pattern.test(v.name));
    if (match) return match;
  }
  return english.find((v) => v.lang === "en-US") ?? english[0] ?? null;
}

export function useSpeech() {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [speaking, setSpeaking] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);

  useEffect(() => {
    if (!supported) return;
    const load = () => {
      voiceRef.current = pickVoice(window.speechSynthesis.getVoices());
    };
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", load);
      window.speechSynthesis.cancel();
    };
  }, [supported]);

  const cancel = useCallback(() => {
    if (supported) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [supported]);

  /** Speak text; resolves when finished (or immediately if muted/unsupported). */
  const speak = useCallback(
    (text: string) =>
      new Promise<void>((resolve) => {
        if (!supported || !enabled || !text.trim()) return resolve();

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = voiceRef.current;
        utterance.lang = voiceRef.current?.lang ?? "en-US";
        utterance.rate = 1.02;
        utterance.pitch = 1;

        const done = () => {
          setSpeaking(false);
          resolve();
        };
        utterance.onend = done;
        utterance.onerror = done;

        setSpeaking(true);
        window.speechSynthesis.speak(utterance);
      }),
    [enabled, supported],
  );

  const toggle = useCallback(() => {
    setEnabled((on) => {
      if (on) window.speechSynthesis?.cancel();
      return !on;
    });
  }, []);

  return { supported, speaking, enabled, speak, cancel, toggle };
}

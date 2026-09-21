import { clsx } from "clsx";
import { LoaderCircle } from "lucide-react";
import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
} from "react";

import { band, BAND_LABEL, BAND_STROKE, BAND_TEXT, formatScore } from "@/lib/format";

// ─── Button ───────────────────────────────────────────────────────────────────

type Variant = "primary" | "secondary" | "ghost" | "danger";

const VARIANT: Record<Variant, string> = {
  primary:
    "bg-gradient-to-r from-accent to-accent-2 text-white shadow-lg shadow-accent/20 hover:brightness-110",
  secondary: "bg-panel-strong text-fg border border-line hover:border-line-strong",
  ghost: "text-muted hover:text-fg hover:bg-panel",
  danger: "bg-bad/15 text-bad border border-bad/30 hover:bg-bad/25",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  icon,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-xl font-medium transition",
        "disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" && "h-8 px-3 text-sm",
        size === "md" && "h-10 px-4 text-sm",
        size === "lg" && "h-12 px-6 text-base",
        VARIANT[variant],
        className,
      )}
    >
      {loading ? <LoaderCircle className="size-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  );
}

// ─── Surfaces ─────────────────────────────────────────────────────────────────

export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div {...rest} className={clsx("glass p-5 sm:p-6", className)} />;
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-4 flex items-baseline justify-between gap-3">
      <h2 className="text-sm font-semibold tracking-wide text-fg">{children}</h2>
      {hint && <span className="text-xs text-faint">{hint}</span>}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "good" | "warn" | "bad";
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        tone === "neutral" && "border-line bg-panel text-muted",
        tone === "accent" && "border-accent/30 bg-accent/10 text-accent",
        tone === "good" && "border-good/30 bg-good/10 text-good",
        tone === "warn" && "border-warn/30 bg-warn/10 text-warn",
        tone === "bad" && "border-bad/30 bg-bad/10 text-bad",
      )}
    >
      {children}
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div role="status" className="flex items-center justify-center gap-2 p-8 text-muted">
      <LoaderCircle className="size-5 animate-spin" aria-hidden />
      <span className="text-sm">{label ?? "Loading…"}</span>
    </div>
  );
}

export function ErrorNote({ children, onDismiss }: { children: ReactNode; onDismiss?: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-start justify-between gap-3 rounded-xl border border-bad/30 bg-bad/10 px-4 py-3 text-sm text-bad"
    >
      <span>{children}</span>
      {onDismiss && (
        <button onClick={onDismiss} className="shrink-0 text-bad/70 hover:text-bad">
          Dismiss
        </button>
      )}
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon: ReactNode;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 px-6 py-14 text-center">
      <div className="grid size-12 place-items-center rounded-2xl bg-panel-strong text-accent">
        {icon}
      </div>
      <h3 className="font-semibold">{title}</h3>
      {children && <p className="max-w-sm text-sm text-muted">{children}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

// ─── Form fields ──────────────────────────────────────────────────────────────

const FIELD =
  "w-full rounded-xl border border-line bg-black/30 px-3.5 py-2.5 text-sm text-fg placeholder:text-faint " +
  "outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20";

export function Label({ htmlFor, children }: { htmlFor: string; children: ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="mb-1.5 block text-xs font-medium text-muted">
      {children}
    </label>
  );
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...rest} className={clsx(FIELD, className)} />;
}

export function Select({ className, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...rest} className={clsx(FIELD, "appearance-none", className)} />;
}

// ─── Scores ───────────────────────────────────────────────────────────────────

/** Circular score gauge. Colour follows the same bands as the LLM rubric. */
export function ScoreRing({
  score,
  size = 120,
  stroke = 10,
  label,
}: {
  score: number | null;
  size?: number;
  stroke?: number;
  label?: string;
}) {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const value = Math.max(0, Math.min(100, score ?? 0));
  const b = band(value);

  return (
    <div className="relative inline-grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={radius} stroke="rgb(255 255 255 / 0.08)"
          strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={score == null ? "transparent" : BAND_STROKE[b]}
          strokeWidth={stroke}
          strokeLinecap="round"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - value / 100)}
          style={{ transition: "stroke-dashoffset 0.9s cubic-bezier(0.2, 0.8, 0.2, 1)" }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="text-3xl font-semibold tabular-nums" style={{ fontSize: size / 4 }}>
            {formatScore(score)}
          </div>
          <div className={clsx("text-xs", score == null ? "text-faint" : BAND_TEXT[b])}>
            {label ?? (score == null ? "No score" : BAND_LABEL[b])}
          </div>
        </div>
      </div>
      <span className="sr-only">
        Score {formatScore(score)} out of 100{score != null && `, ${BAND_LABEL[b]}`}
      </span>
    </div>
  );
}

/** Horizontal 0-100 bar. One hue for magnitude — status colours are reserved for state. */
export function ScoreBar({ label, value, hint }: { label: string; value: number | null; hint?: string }) {
  const v = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-sm">
        <span className="text-muted">{label}</span>
        <span className="tabular-nums">
          {formatScore(value)}
          {hint && <span className="ml-1.5 text-xs text-faint">{hint}</span>}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/8">
        <div
          className="h-full rounded-full transition-[width] duration-700"
          style={{ width: `${v}%`, background: value == null ? "transparent" : "var(--color-series-1)" }}
        />
      </div>
    </div>
  );
}

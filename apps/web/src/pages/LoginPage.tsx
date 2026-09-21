import { clsx } from "clsx";
import { Mic } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate, useSearchParams } from "react-router";

import { ApiError } from "@/api/client";
import { Button, Card, ErrorNote, Input, Label } from "@/components/ui";
import { useAuth } from "@/features/auth/auth-context";

type Mode = "login" | "register";

export function LoginPage() {
  const { user, login, register, demo } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();

  const [mode, setMode] = useState<Mode>(params.get("mode") === "register" ? "register" : "login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState<"form" | "demo" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const redirectTo = (location.state as { from?: string } | null)?.from ?? "/app";
  // Only redirect visitors who were already signed in; after an in-page sign-in,
  // run() navigates itself (the demo goes to /app/new, not redirectTo).
  if (user && busy === null) return <Navigate to={redirectTo} replace />;

  const run = async (kind: "form" | "demo", action: () => Promise<void>) => {
    setBusy(kind);
    setError(null);
    try {
      await action();
      navigate(kind === "demo" ? "/app/new" : redirectTo, { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
      setBusy(null);
    }
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void run("form", () =>
      mode === "login" ? login(email, password) : register(email, password, name),
    );
  };

  return (
    <div className="grid min-h-dvh place-items-center px-4 py-10">
      <div className="w-full max-w-sm animate-fade-up">
        <Link to="/" className="mb-8 flex items-center justify-center gap-2 font-semibold">
          <span className="grid size-8 place-items-center rounded-lg bg-gradient-to-br from-accent to-accent-2">
            <Mic className="size-4 text-white" aria-hidden />
          </span>
          Voice HR
        </Link>

        <Card>
          <div className="mb-6 grid grid-cols-2 rounded-xl bg-black/30 p-1" role="tablist">
            {(["login", "register"] as const).map((m) => (
              <button
                key={m}
                role="tab"
                aria-selected={mode === m}
                onClick={() => {
                  setMode(m);
                  setError(null);
                }}
                className={clsx(
                  "rounded-lg py-1.5 text-sm transition",
                  mode === m ? "bg-panel-strong text-fg" : "text-muted hover:text-fg",
                )}
              >
                {m === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>

          <form onSubmit={onSubmit} className="space-y-4">
            {mode === "register" && (
              <div>
                <Label htmlFor="name">Name</Label>
                <Input id="name" autoComplete="name" required value={name}
                  onChange={(e) => setName(e.target.value)} />
              </div>
            )}
            <div>
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" autoComplete="email" required value={email}
                onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                minLength={mode === "register" ? 8 : undefined}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              {mode === "register" && (
                <p className="mt-1.5 text-xs text-faint">At least 8 characters.</p>
              )}
            </div>

            {error && <ErrorNote>{error}</ErrorNote>}

            <Button type="submit" className="w-full" loading={busy === "form"} disabled={busy !== null}>
              {mode === "login" ? "Sign in" : "Create account"}
            </Button>
          </form>

          <div className="my-5 flex items-center gap-3 text-xs text-faint">
            <span className="h-px flex-1 bg-line" /> or <span className="h-px flex-1 bg-line" />
          </div>

          <Button
            variant="secondary"
            className="w-full"
            loading={busy === "demo"}
            disabled={busy !== null}
            onClick={() => void run("demo", demo)}
          >
            Continue with the demo account
          </Button>
        </Card>
      </div>
    </div>
  );
}

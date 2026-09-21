import { clsx } from "clsx";
import { History, LayoutDashboard, LogOut, Mic, Plus } from "lucide-react";
import type { ReactNode } from "react";
import { Link, NavLink } from "react-router";

import { useCapabilities } from "@/api/queries";
import { Badge } from "@/components/ui";
import { useAuth } from "@/features/auth/auth-context";

const NAV = [
  { to: "/app", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/app/history", label: "History", icon: History, end: false },
];

export function AppShell({ children }: { children: ReactNode }) {
  const { user, logout } = useAuth();
  const { data: caps } = useCapabilities();

  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-20 border-b border-line bg-ink/60 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-2 px-4">
          <Link to="/app" className="mr-3 flex items-center gap-2 font-semibold">
            <span className="grid size-7 place-items-center rounded-lg bg-gradient-to-br from-accent to-accent-2">
              <Mic className="size-4 text-white" aria-hidden />
            </span>
            <span className="hidden sm:inline">Voice HR</span>
          </Link>

          <nav className="flex items-center gap-1" aria-label="Main">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  clsx(
                    "flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition",
                    isActive ? "bg-panel-strong text-fg" : "text-muted hover:text-fg",
                  )
                }
              >
                <Icon className="size-4" aria-hidden />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            {caps && (
              <span className="hidden md:inline" title={`LLM: ${caps.llm.model}`}>
                <Badge tone="accent">{caps.llm.provider}</Badge>
              </span>
            )}
            <Link
              to="/app/new"
              className="flex h-8 items-center gap-1.5 rounded-lg bg-gradient-to-r from-accent to-accent-2 px-3 text-sm font-medium text-white"
            >
              <Plus className="size-4" aria-hidden />
              New interview
            </Link>
            <span className="hidden max-w-32 truncate text-sm text-muted lg:inline">
              {user?.display_name}
            </span>
            <button
              onClick={logout}
              className="grid size-8 place-items-center rounded-lg text-muted hover:bg-panel hover:text-fg"
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut className="size-4" aria-hidden />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
    </div>
  );
}

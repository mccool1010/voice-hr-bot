import { lazy, Suspense, type ReactNode } from "react";
import { createBrowserRouter, Navigate, Outlet, useLocation } from "react-router";

import { Spinner } from "@/components/ui";
import { AppShell } from "@/components/AppShell";
import { useAuth } from "@/features/auth/auth-context";
import { HistoryPage } from "@/pages/HistoryPage";
import { InterviewPage } from "@/pages/InterviewPage";
import { LandingPage } from "@/pages/LandingPage";
import { LoginPage } from "@/pages/LoginPage";
import { NotFoundPage } from "@/pages/NotFoundPage";
import { SetupPage } from "@/pages/SetupPage";

// Recharts is ~40% of the bundle and only these two pages chart anything.
const DashboardPage = lazy(() => import("@/pages/DashboardPage"));
const ReportPage = lazy(() => import("@/pages/ReportPage"));

function RequireAuth() {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <Spinner label="Signing you in…" />;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return (
    <AppShell>
      <Outlet />
    </AppShell>
  );
}

const lazyPage = (node: ReactNode) => <Suspense fallback={<Spinner />}>{node}</Suspense>;

export const router = createBrowserRouter([
  { path: "/", element: <LandingPage /> },
  { path: "/login", element: <LoginPage /> },
  {
    path: "/app",
    element: <RequireAuth />,
    children: [
      { index: true, element: lazyPage(<DashboardPage />) },
      { path: "new", element: <SetupPage /> },
      { path: "history", element: <HistoryPage /> },
      { path: "interview/:id", element: <InterviewPage /> },
      { path: "report/:id", element: lazyPage(<ReportPage />) },
    ],
  },
  { path: "*", element: <NotFoundPage /> },
]);

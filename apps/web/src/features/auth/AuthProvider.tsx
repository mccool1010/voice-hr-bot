import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { api, tokenStore } from "@/api/client";
import type { TokenPair, User } from "@/api/types";

import { AuthContext, type AuthState } from "./auth-context";

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(() => tokenStore.get() !== null);

  // Validate a stored token once on load; a stale token simply signs you out.
  useEffect(() => {
    if (!tokenStore.get()) return;
    let cancelled = false;
    api.auth
      .me()
      .then((me) => !cancelled && setUser(me))
      .catch(() => tokenStore.clear())
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const accept = useCallback(
    (pair: TokenPair) => {
      tokenStore.set(pair.access_token);
      queryClient.clear(); // never show one account's cached data to another
      setUser(pair.user);
    },
    [queryClient],
  );

  const value = useMemo<AuthState>(
    () => ({
      user,
      loading,
      login: async (email, password) => accept(await api.auth.login(email, password)),
      register: async (email, password, name) =>
        accept(await api.auth.register(email, password, name)),
      demo: async () => accept(await api.auth.demo()),
      logout: () => {
        tokenStore.clear();
        queryClient.clear();
        setUser(null);
      },
    }),
    [user, loading, accept, queryClient],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

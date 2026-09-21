import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";

import { api } from "./client";
import type { InterviewCreate } from "./types";

// Centralised keys so invalidation after a mutation can never miss a cache entry.
export const keys = {
  me: ["me"] as const,
  capabilities: ["capabilities"] as const,
  interviews: ["interviews"] as const,
  interview: (id: string) => ["interviews", id] as const,
  resumes: ["resumes"] as const,
  dashboard: ["dashboard"] as const,
};

export const useCapabilities = () =>
  useQuery({ queryKey: keys.capabilities, queryFn: api.capabilities, staleTime: 5 * 60_000 });

export const useInterviews = () =>
  useQuery({ queryKey: keys.interviews, queryFn: api.interviews.list });

export const useInterview = (id: string | undefined) =>
  useQuery({
    queryKey: keys.interview(id ?? ""),
    queryFn: () => api.interviews.get(id!),
    enabled: Boolean(id),
  });

export const useResumes = () => useQuery({ queryKey: keys.resumes, queryFn: api.resumes.list });

export const useDashboard = () =>
  useQuery({ queryKey: keys.dashboard, queryFn: api.analytics.dashboard });

export function useCreateInterview() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: InterviewCreate) => api.interviews.create(payload),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.interviews }),
  });
}

export function useUploadResume() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => api.resumes.upload(file),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.resumes }),
  });
}

export function useDeleteResume() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.resumes.remove(id),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.resumes }),
  });
}

/** Call after an interview finishes: history, detail and dashboard all change.
 *  Stable identity matters — callers put this in effect dependency lists, and a
 *  fresh function per render would re-invalidate (and refetch) on every render. */
export function useInvalidateAfterInterview() {
  const client = useQueryClient();
  return useCallback(
    (id: string) => {
      void client.invalidateQueries({ queryKey: keys.interviews });
      void client.invalidateQueries({ queryKey: keys.interview(id) });
      void client.invalidateQueries({ queryKey: keys.dashboard });
    },
    [client],
  );
}

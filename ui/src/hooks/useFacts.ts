"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";

export interface ContradictionItem {
  id: string;
  category: string;
  key: string;
  existing_value: string;
  existing_confidence: number;
  candidate_value: string;
  candidate_source: string;
  candidate_confidence: number;
  created_at: string;
}

export interface ContradictionListResponse {
  user_id: string;
  app_id: string;
  contradictions: ContradictionItem[];
  total: number;
}

export interface ContradictionResolved {
  id: string;
  resolution: string;
  resolved_at: string;
}

export function useContradictions(appId = "default") {
  const { data: session } = useSession();
  const userId = session?.user?.id;
  const token = session?.accessToken;

  return useQuery<ContradictionListResponse>({
    queryKey: ["contradictions", userId, appId],
    queryFn: () =>
      createApiClient(token).listContradictions(userId!, appId) as Promise<ContradictionListResponse>,
    enabled: !!userId && !!token,
  });
}

export function useResolveContradiction() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const userId = session?.user?.id;
  const qc = useQueryClient();

  return useMutation<
    ContradictionResolved,
    Error,
    { id: string; keep: "existing" | "candidate" | "merge"; merged_value?: string }
  >({
    mutationFn: ({ id, keep, merged_value }) =>
      createApiClient(token).resolveContradiction(id, { keep, merged_value }) as Promise<ContradictionResolved>,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["contradictions", userId] }),
  });
}

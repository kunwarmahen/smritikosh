"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { RecentEventsResponse, SearchResponse, FeedbackResponse } from "@/types";

export function useRecentEvents(params?: { limit?: number; app_id?: string }) {
  const { data: session } = useSession();
  const userId = session?.user?.id;
  const token = session?.accessToken;

  return useQuery<RecentEventsResponse>({
    queryKey: ["memory", "recent", userId, params],
    queryFn: () =>
      createApiClient(token).getRecentEvents(userId!, params) as Promise<RecentEventsResponse>,
    enabled: !!userId && !!token,
  });
}

export function useSearchMemory() {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useMutation<
    SearchResponse,
    Error,
    { query: string; app_id?: string; limit?: number; from_date?: string; to_date?: string }
  >({
    mutationFn: (vars) =>
      createApiClient(token).searchMemory({
        user_id: session!.user.id,
        ...vars,
      }) as Promise<SearchResponse>,
  });
}

export function useSubmitFeedback() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const qc = useQueryClient();

  return useMutation<
    FeedbackResponse,
    Error,
    { event_id: string; feedback_type: "positive" | "negative" | "neutral"; comment?: string }
  >({
    mutationFn: (vars) =>
      createApiClient(token).submitFeedback({
        ...vars,
        user_id: session!.user.id,
      }) as Promise<FeedbackResponse>,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["memory", "recent"] });
    },
  });
}

export function useDeleteEvent() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const qc = useQueryClient();

  return useMutation<unknown, Error, string>({
    mutationFn: (eventId) => createApiClient(token).deleteEvent(eventId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["memory"] });
    },
  });
}

"use client";

import { useQuery } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { MemoryEvent, MemoryLinksResponse } from "@/types";

export function useMemoryEvent(eventId?: string) {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useQuery<MemoryEvent>({
    queryKey: ["memory-event", eventId],
    queryFn: () =>
      createApiClient(token).getEvent(eventId!) as Promise<MemoryEvent>,
    enabled: !!eventId && !!token,
    staleTime: 30_000,
  });
}

export function useMemoryLinks(eventId?: string) {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useQuery<MemoryLinksResponse>({
    queryKey: ["memory-links", eventId],
    queryFn: () =>
      createApiClient(token).getEventLinks(eventId!) as Promise<MemoryLinksResponse>,
    enabled: !!eventId && !!token,
    staleTime: 30_000,
  });
}

"use client";

import { useQuery } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { AuditEvent, AuditStats } from "@/types";

export function useAuditTimeline(params?: {
  event_type?: string;
  limit?: number;
  offset?: number;
  from_ts?: string;
  to_ts?: string;
  userId?: string; // admin: override user; regular: uses session user
}) {
  const { data: session } = useSession();
  const userId = params?.userId ?? session?.user?.id;
  const token = session?.accessToken;

  return useQuery<AuditEvent[]>({
    queryKey: ["audit", "timeline", userId, params],
    queryFn: () =>
      createApiClient(token).getAuditTimeline(userId!, params).then(
        (res: any) => res?.records ?? res ?? []
      ) as Promise<AuditEvent[]>,
    enabled: !!userId && !!token,
  });
}

export function useAuditStats(userId?: string) {
  const { data: session } = useSession();
  const targetUser = userId ?? session?.user?.id;
  const token = session?.accessToken;

  return useQuery<AuditStats>({
    queryKey: ["audit", "stats", targetUser],
    queryFn: () =>
      createApiClient(token).getAuditStats(targetUser!).then(
        (res: any) => res?.counts ?? res ?? {}
      ) as Promise<AuditStats>,
    enabled: !!targetUser && !!token,
  });
}

export function useEventLineage(eventId?: string) {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useQuery<AuditEvent[]>({
    queryKey: ["audit", "lineage", eventId],
    queryFn: () =>
      createApiClient(token).getEventLineage(eventId!) as Promise<AuditEvent[]>,
    enabled: !!eventId && !!token,
  });
}

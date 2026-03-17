"use client";

import { useQuery } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { FactGraph } from "@/types";

export function useFactGraph(minConfidence = 0) {
  const { data: session } = useSession();
  const userId = session?.user?.id;
  const token = session?.accessToken;

  return useQuery<FactGraph>({
    queryKey: ["fact-graph", userId, minConfidence],
    queryFn: () =>
      createApiClient(token).getFactGraph(userId!, "default", minConfidence) as Promise<FactGraph>,
    enabled: !!userId && !!token,
    staleTime: 60_000,
  });
}

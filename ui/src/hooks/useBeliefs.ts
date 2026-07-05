"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { BeliefEvidence } from "@/types";

export function useBeliefEvidence(beliefId: string | null) {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const userId = session?.user?.id;

  return useQuery<BeliefEvidence>({
    queryKey: ["belief-evidence", userId, beliefId],
    queryFn: () =>
      createApiClient(token).getBeliefEvidence(userId!, beliefId!) as Promise<BeliefEvidence>,
    enabled: !!token && !!userId && !!beliefId,
  });
}

export function useRetractBelief() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const userId = session?.user?.id;
  const qc = useQueryClient();

  return useMutation<unknown, Error, string>({
    mutationFn: (beliefId) => createApiClient(token).retractBelief(userId!, beliefId),
    onSuccess: () => {
      // The identity profile embeds the belief list — refresh both.
      qc.invalidateQueries({ queryKey: ["identity"] });
      qc.invalidateQueries({ queryKey: ["belief-evidence"] });
    },
  });
}

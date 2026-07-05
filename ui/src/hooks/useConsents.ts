"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";

export interface ConsentItem {
  consent_id: string;
  user_id: string;
  source_app_id: string;
  target_app_id: string;
  categories: string[];
  active: boolean;
  granted_at: string;
  revoked_at: string | null;
  created_by: string;
}

interface ConsentListResponse {
  user_id: string;
  consents: ConsentItem[];
}

export function useConsents(includeRevoked = false) {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const userId = session?.user?.id;

  return useQuery<ConsentItem[]>({
    queryKey: ["consents", userId, includeRevoked],
    queryFn: () =>
      (createApiClient(token).listConsents(userId!, includeRevoked) as Promise<ConsentListResponse>)
        .then((res) => res?.consents ?? []),
    enabled: !!token && !!userId,
  });
}

export function useGrantConsent() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const userId = session?.user?.id;
  const qc = useQueryClient();

  return useMutation<ConsentItem, Error, { source_app_id: string; target_app_id: string; categories: string[] }>({
    mutationFn: (body) =>
      createApiClient(token).grantConsent({ user_id: userId!, ...body }) as Promise<ConsentItem>,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["consents"] }),
  });
}

export function useRevokeConsent() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const userId = session?.user?.id;
  const qc = useQueryClient();

  return useMutation<unknown, Error, { source_app_id: string; target_app_id: string }>({
    mutationFn: (body) =>
      createApiClient(token).revokeConsent({ user_id: userId!, ...body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["consents"] }),
  });
}

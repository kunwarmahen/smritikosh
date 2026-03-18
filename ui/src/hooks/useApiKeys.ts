"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";

export interface ApiKeyItem {
  id: string;
  name: string;
  key_prefix: string;
  app_ids: string[];
  last_used_at: string | null;
  created_at: string;
}

export interface CreatedApiKey extends ApiKeyItem {
  key: string; // full key — only present immediately after creation
}

export function useApiKeys() {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useQuery<ApiKeyItem[]>({
    queryKey: ["api-keys"],
    queryFn: () =>
      createApiClient(token).listApiKeys().then((res: any) => res?.keys ?? []),
    enabled: !!token,
  });
}

export function useCreateApiKey() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const qc = useQueryClient();

  return useMutation<CreatedApiKey, Error, { name: string; app_ids?: string[] }>({
    mutationFn: (body) => createApiClient(token).createApiKey(body) as Promise<CreatedApiKey>,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["api-keys"] }),
  });
}

export function useRevokeApiKey() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const qc = useQueryClient();

  return useMutation<unknown, Error, string>({
    mutationFn: (keyId) => createApiClient(token).revokeApiKey(keyId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["api-keys"] }),
  });
}

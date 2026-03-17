"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { AdminUser, AdminUsersResponse, HealthStatus } from "@/types";

export function useAdminUsers(params?: { limit?: number; offset?: number; role?: string }) {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useQuery<AdminUsersResponse>({
    queryKey: ["admin-users", params],
    queryFn: () => createApiClient(token).adminListUsers(params) as Promise<AdminUsersResponse>,
    enabled: !!token,
  });
}

export function useAdminUser(username?: string) {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useQuery<AdminUser>({
    queryKey: ["admin-user", username],
    queryFn: () => createApiClient(token).adminGetUser(username!) as Promise<AdminUser>,
    enabled: !!username && !!token,
  });
}

export function useAdminPatchUser() {
  const { data: session } = useSession();
  const qc = useQueryClient();
  const token = session?.accessToken;

  return useMutation({
    mutationFn: ({ username, body }: { username: string; body: { is_active?: boolean; role?: string } }) =>
      createApiClient(token).adminPatchUser(username, body),
    onSuccess: (_, { username }) => {
      qc.invalidateQueries({ queryKey: ["admin-users"] });
      qc.invalidateQueries({ queryKey: ["admin-user", username] });
    },
  });
}

export function useHealth() {
  const { data: session } = useSession();
  const token = session?.accessToken;

  return useQuery<HealthStatus>({
    queryKey: ["health"],
    queryFn: () => createApiClient(token).health() as Promise<HealthStatus>,
    refetchInterval: 30_000,
  });
}

export function useAdminConsolidate() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  return useMutation({
    mutationFn: ({ userId, appId }: { userId: string; appId?: string }) =>
      createApiClient(token).adminConsolidate(userId, appId),
  });
}

export function useAdminPrune() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  return useMutation({
    mutationFn: ({ userId, appId }: { userId: string; appId?: string }) =>
      createApiClient(token).adminPrune(userId, appId),
  });
}

export function useAdminCluster() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  return useMutation({
    mutationFn: ({ userId, appId }: { userId: string; appId?: string }) =>
      createApiClient(token).adminCluster(userId, appId),
  });
}

export function useAdminMineBeliefs() {
  const { data: session } = useSession();
  const token = session?.accessToken;
  return useMutation({
    mutationFn: ({ userId, appId }: { userId: string; appId?: string }) =>
      createApiClient(token).adminMineBeliefs(userId, appId),
  });
}

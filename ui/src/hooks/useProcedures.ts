"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { Procedure } from "@/types";

export function useProcedures(activeOnly = false) {
  const { data: session } = useSession();
  const userId = session?.user?.id;
  const token = session?.accessToken;

  return useQuery<Procedure[]>({
    queryKey: ["procedures", userId, activeOnly],
    queryFn: () =>
      createApiClient(token).getProcedures(userId!, undefined, activeOnly).then(
        (res: any) => res?.procedures ?? res ?? []
      ) as Promise<Procedure[]>,
    enabled: !!userId && !!token,
  });
}

export function useCreateProcedure() {
  const { data: session } = useSession();
  const qc = useQueryClient();
  const token = session?.accessToken;
  const userId = session?.user?.id;

  return useMutation({
    mutationFn: (body: { trigger: string; instruction: string; category?: string; priority?: number }) =>
      createApiClient(token).createProcedure({
        user_id: userId!,
        app_id: "default",
        ...body,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["procedures"] }),
  });
}

export function useUpdateProcedure() {
  const { data: session } = useSession();
  const qc = useQueryClient();
  const token = session?.accessToken;

  return useMutation({
    mutationFn: ({
      procedureId,
      body,
    }: {
      procedureId: string;
      body: Partial<{ priority: number; is_active: boolean; instruction: string }>;
    }) => createApiClient(token).updateProcedure(procedureId, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["procedures"] }),
  });
}

export function useDeleteProcedure() {
  const { data: session } = useSession();
  const qc = useQueryClient();
  const token = session?.accessToken;

  return useMutation({
    mutationFn: (procedureId: string) => createApiClient(token).deleteProcedure(procedureId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["procedures"] }),
  });
}

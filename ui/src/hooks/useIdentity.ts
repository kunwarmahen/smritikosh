"use client";

import { useQuery } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { IdentityProfile } from "@/types";

export function useIdentity(userId?: string) {
  const { data: session } = useSession();
  const targetUser = userId ?? session?.user?.id;
  const token = session?.accessToken;

  return useQuery<IdentityProfile>({
    queryKey: ["identity", targetUser],
    queryFn: () =>
      createApiClient(token).getIdentity(targetUser!) as Promise<IdentityProfile>,
    enabled: !!targetUser && !!token,
    staleTime: 60_000, // identity changes slowly
  });
}

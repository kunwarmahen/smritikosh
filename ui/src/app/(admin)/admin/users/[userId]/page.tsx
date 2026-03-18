"use client";

import { use, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft, ShieldCheck, Shield, ToggleLeft, ToggleRight,
  Loader2, Trash2, ExternalLink,
} from "lucide-react";
import { clsx } from "clsx";
import { formatDistanceToNow } from "date-fns";
import { useMutation } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import { useAdminUser, useAdminPatchUser } from "@/hooks/useAdmin";

export default function AdminUserDetailPage({
  params,
}: {
  params: Promise<{ userId: string }>;
}) {
  const { userId: username } = use(params);
  const router = useRouter();
  const { data: session } = useSession();
  const token = session?.accessToken;

  const { data: user, isLoading, isError, refetch } = useAdminUser(username);
  const patch = useAdminPatchUser();

  const deleteMemory = useMutation({
    mutationFn: () => createApiClient(token).deleteUserMemory(username),
  });
  const [deleteConfirm, setDeleteConfirm] = useState(false);

  function toggleActive() {
    if (!user) return;
    patch.mutate({ username, body: { is_active: !user.is_active } }, { onSuccess: () => refetch() });
  }

  function toggleRole() {
    if (!user) return;
    patch.mutate(
      { username, body: { role: user.role === "admin" ? "user" : "admin" } },
      { onSuccess: () => refetch() },
    );
  }

  return (
    <div className="max-w-xl">
      {/* Back + title */}
      <div className="flex items-center gap-3 mb-6">
        <button onClick={() => router.back()} className="btn-ghost px-2 py-1.5">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div>
          <h1 className="text-base font-semibold text-zinc-100 tracking-tight">{username}</h1>
          <p className="text-xs text-zinc-600 mt-0.5">User detail</p>
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-zinc-600 py-12 justify-center">
          <Loader2 className="w-4 h-4 animate-spin" />
        </div>
      )}

      {isError && (
        <div className="card border-rose-500/20 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load user.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {user && (
        <div className="space-y-3">
          {/* Account info */}
          <div className="card">
            <p className="section-heading mb-3">Account</p>
            <div className="space-y-0">
              <InfoRow label="Username"><span className="mono text-zinc-300">{user.username}</span></InfoRow>
              <InfoRow label="Email"><span className="text-zinc-400">{user.email ?? "—"}</span></InfoRow>
              <InfoRow label="App IDs"><span className="mono text-zinc-400">{(user.app_ids ?? []).join(", ")}</span></InfoRow>
              <InfoRow label="Created">
                <span className="text-zinc-500 text-xs">
                  {formatDistanceToNow(new Date(user.created_at), { addSuffix: true })}
                </span>
              </InfoRow>
              <InfoRow label="Updated" last>
                <span className="text-zinc-500 text-xs">
                  {formatDistanceToNow(new Date(user.updated_at), { addSuffix: true })}
                </span>
              </InfoRow>
            </div>
          </div>

          {/* Access control */}
          <div className="card">
            <p className="section-heading mb-3">Access control</p>
            <div className="space-y-0">
              {/* Active */}
              <div className="flex items-center justify-between py-3 border-b border-zinc-800/60">
                <div>
                  <p className="text-sm text-zinc-300">Account active</p>
                  <p className="text-xs text-zinc-600 mt-0.5">Inactive users cannot sign in.</p>
                </div>
                <button
                  onClick={toggleActive}
                  disabled={patch.isPending}
                  className="flex items-center gap-2 transition-colors"
                >
                  {user.is_active
                    ? <ToggleRight className="w-7 h-7 text-emerald-500" />
                    : <ToggleLeft  className="w-7 h-7 text-zinc-700" />}
                  <span className={clsx(
                    "text-xs font-medium w-14",
                    user.is_active ? "text-emerald-500" : "text-zinc-600",
                  )}>
                    {user.is_active ? "Active" : "Inactive"}
                  </span>
                </button>
              </div>

              {/* Role */}
              <div className="flex items-center justify-between py-3">
                <div>
                  <p className="text-sm text-zinc-300">Role</p>
                  <p className="text-xs text-zinc-600 mt-0.5">Admin users access the admin panel.</p>
                </div>
                <button
                  onClick={toggleRole}
                  disabled={patch.isPending}
                  className={clsx(
                    "flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border transition-colors",
                    user.role === "admin"
                      ? "bg-amber-500/10 border-amber-500/20 text-amber-400"
                      : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200",
                  )}
                >
                  {user.role === "admin"
                    ? <ShieldCheck className="w-3.5 h-3.5" />
                    : <Shield      className="w-3.5 h-3.5" />}
                  {user.role}
                </button>
              </div>
            </div>
          </div>

          {/* Audit link */}
          <div className="card">
            <p className="section-heading mb-3">Data</p>
            <button
              onClick={() => router.push(`/admin/audit?user=${username}`)}
              className="flex items-center gap-1.5 text-sm text-violet-400 hover:text-violet-300 transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              View audit log for {username}
            </button>
          </div>

          {/* Danger zone */}
          <div className="card border-rose-500/20 bg-rose-500/5">
            <p className="section-heading text-rose-500 mb-3">Danger zone</p>
            {!deleteConfirm ? (
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm text-zinc-300">Delete all memories</p>
                  <p className="text-xs text-zinc-600 mt-0.5">
                    Permanently removes every episodic event. Neo4j facts are not affected.
                  </p>
                </div>
                <button
                  onClick={() => setDeleteConfirm(true)}
                  className="btn-danger flex-shrink-0 text-xs"
                >
                  <Trash2 className="w-3.5 h-3.5" /> Delete
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-rose-300">
                  Delete all memories for <strong>{username}</strong>? This cannot be undone.
                </p>
                <div className="flex gap-2">
                  <button onClick={() => setDeleteConfirm(false)} className="btn-secondary flex-1">
                    Cancel
                  </button>
                  <button
                    onClick={() => deleteMemory.mutate(undefined, { onSuccess: () => setDeleteConfirm(false) })}
                    disabled={deleteMemory.isPending}
                    className="btn-danger flex-1 justify-center"
                  >
                    {deleteMemory.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                    Confirm delete
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function InfoRow({
  label,
  children,
  last = false,
}: {
  label: string;
  children: React.ReactNode;
  last?: boolean;
}) {
  return (
    <div className={clsx(
      "flex items-center justify-between py-2.5",
      !last && "border-b border-zinc-800/60",
    )}>
      <span className="text-xs text-zinc-600 w-20 flex-shrink-0">{label}</span>
      <span className="text-sm flex-1 text-right">{children}</span>
    </div>
  );
}

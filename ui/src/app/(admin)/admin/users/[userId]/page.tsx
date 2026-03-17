"use client";

import { use, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowLeft, User, ShieldCheck, Shield, ToggleLeft, ToggleRight,
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

  function handleDeleteMemory() {
    deleteMemory.mutate(undefined, {
      onSuccess: () => setDeleteConfirm(false),
    });
  }

  function toggleActive() {
    if (!user) return;
    patch.mutate({ username, body: { is_active: !user.is_active } }, {
      onSuccess: () => refetch(),
    });
  }

  function toggleRole() {
    if (!user) return;
    const newRole = user.role === "admin" ? "user" : "admin";
    patch.mutate({ username, body: { role: newRole } }, {
      onSuccess: () => refetch(),
    });
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={() => router.back()}
          className="btn-ghost flex items-center gap-1.5 text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <div>
          <h1 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <User className="w-5 h-5 text-amber-400" />
            {username}
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">User detail</p>
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-slate-500 py-12 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
        </div>
      )}

      {isError && (
        <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load user.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {user && (
        <div className="space-y-4">
          {/* Info card */}
          <div className="card">
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-4">
              Account info
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
              <InfoRow label="Username"   value={<span className="font-mono">{user.username}</span>} />
              <InfoRow label="Email"      value={user.email ?? "—"} />
              <InfoRow label="App ID"     value={<span className="font-mono">{user.app_id}</span>} />
              <InfoRow label="Created"
                value={formatDistanceToNow(new Date(user.created_at), { addSuffix: true })} />
              <InfoRow label="Updated"
                value={formatDistanceToNow(new Date(user.updated_at), { addSuffix: true })} />
            </div>
          </div>

          {/* Controls */}
          <div className="card">
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-4">
              Access control
            </p>
            <div className="space-y-3">
              {/* Active toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-300">Account active</p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Inactive users cannot log in.
                  </p>
                </div>
                <button
                  onClick={toggleActive}
                  disabled={patch.isPending}
                  className="flex items-center gap-2 transition-colors"
                >
                  {user.is_active
                    ? <ToggleRight className="w-7 h-7 text-emerald-400" />
                    : <ToggleLeft  className="w-7 h-7 text-slate-600"  />}
                  <span className={clsx(
                    "text-sm font-medium",
                    user.is_active ? "text-emerald-400" : "text-slate-500",
                  )}>
                    {user.is_active ? "Active" : "Inactive"}
                  </span>
                </button>
              </div>

              {/* Role toggle */}
              <div className="flex items-center justify-between pt-3 border-t border-slate-800">
                <div>
                  <p className="text-sm text-slate-300">Role</p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Admin users can access the admin panel.
                  </p>
                </div>
                <button
                  onClick={toggleRole}
                  disabled={patch.isPending}
                  className={clsx(
                    "flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg border transition-colors",
                    user.role === "admin"
                      ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                      : "bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200",
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

          {/* Memories link */}
          <div className="card">
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-3">
              Memories
            </p>
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
            <p className="text-xs font-medium text-rose-400 uppercase tracking-wide mb-4">
              Danger zone
            </p>

            {!deleteConfirm ? (
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-sm text-slate-300">Delete all memories</p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Permanently removes every episodic event for this user.
                    Semantic facts in Neo4j are not affected.
                  </p>
                </div>
                <button
                  onClick={() => setDeleteConfirm(true)}
                  className="btn-danger flex items-center gap-1.5 text-sm flex-shrink-0"
                >
                  <Trash2 className="w-3.5 h-3.5" /> Delete memories
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-rose-300">
                  Delete all memories for <strong>{username}</strong>? This cannot be undone.
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={() => setDeleteConfirm(false)}
                    className="btn-secondary flex-1"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleDeleteMemory}
                    disabled={deleteMemory.isPending}
                    className="btn-danger flex-1 flex items-center justify-center gap-2"
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

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="text-slate-500 w-24 flex-shrink-0">{label}</span>
      <span className="text-slate-300">{value}</span>
    </div>
  );
}

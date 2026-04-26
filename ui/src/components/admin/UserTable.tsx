"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Loader2, Users, Plus, X, ChevronLeft, ChevronRight,
  ToggleLeft, ToggleRight, ShieldCheck, Shield,
} from "lucide-react";
import { clsx } from "clsx";
import { formatDistanceToNow } from "date-fns";
import { useMutation } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import { useAdminUsers, useAdminPatchUser } from "@/hooks/useAdmin";
import type { AdminUser } from "@/types";

const PAGE_SIZE = 20;

// ── New user modal ─────────────────────────────────────────────────────────────
interface NewUserForm {
  username: string; password: string;
  role: "user" | "admin"; email: string; app_id: string;
}
const EMPTY: NewUserForm = { username: "", password: "", role: "user", email: "", app_id: "default" };

function NewUserModal({ onClose }: { onClose: () => void }) {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const [form, setForm] = useState<NewUserForm>(EMPTY);
  const [error, setError] = useState("");

  const register = useMutation({
    mutationFn: (body: NewUserForm) => createApiClient(token).register(body),
    onSuccess: () => onClose(),
    onError: (err: Error) => setError(err.message ?? "Registration failed."),
  });

  function f(patch: Partial<NewUserForm>) { setForm((p) => ({ ...p, ...patch })); }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!form.username.trim() || !form.password.trim()) {
      setError("Username and password are required.");
      return;
    }
    register.mutate(form);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-zinc-950/80 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Register new user</h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-400 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={submit} className="p-5 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label">Username</label>
              <input type="text" className="input" value={form.username}
                onChange={(e) => f({ username: e.target.value })} placeholder="alice" autoFocus />
            </div>
            <div>
              <label className="label">Password</label>
              <input type="password" className="input" value={form.password}
                onChange={(e) => f({ password: e.target.value })} placeholder="••••••••" />
            </div>
            <div>
              <label className="label">Role</label>
              <select className="input" value={form.role}
                onChange={(e) => f({ role: e.target.value as "user" | "admin" })}>
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select>
            </div>
            <div>
              <label className="label">App ID</label>
              <input type="text" className="input" value={form.app_id}
                onChange={(e) => f({ app_id: e.target.value })} placeholder="default" />
            </div>
            <div className="col-span-2">
              <label className="label">Email <span className="normal-case text-zinc-700">(optional)</span></label>
              <input type="email" className="input" value={form.email}
                onChange={(e) => f({ email: e.target.value })} placeholder="alice@example.com" />
            </div>
          </div>

          {error && (
            <p className="text-xs text-rose-400 bg-rose-500/8 border border-rose-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <div className="flex gap-2 pt-1">
            <button type="button" onClick={onClose} className="btn-secondary flex-1">Cancel</button>
            <button type="submit" disabled={register.isPending} className="btn-primary flex-1 justify-center">
              {register.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              Create user
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ── User row ──────────────────────────────────────────────────────────────────
function UserRow({ user }: { user: AdminUser }) {
  const router = useRouter();
  const patch = useAdminPatchUser();

  return (
    <tr
      className="border-b border-zinc-200 dark:border-zinc-800/60 hover:bg-zinc-50 dark:hover:bg-zinc-800/30 cursor-pointer transition-colors"
      onClick={() => router.push(`/admin/users/${user.username}`)}
    >
      <td className="px-4 py-3">
        <span className="mono text-zinc-800 dark:text-zinc-200 text-xs">{user.username}</span>
      </td>
      <td className="px-4 py-3">
        <span className={clsx(
          "badge text-xs",
          user.role === "admin"
            ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
            : "bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border border-zinc-200 dark:border-zinc-700/50",
        )}>
          {user.role === "admin" ? <ShieldCheck className="w-3 h-3" /> : <Shield className="w-3 h-3" />}
          {user.role}
        </span>
      </td>
      <td className="px-4 py-3">
        <span className="mono text-zinc-600">{user.app_id}</span>
      </td>
      <td className="px-4 py-3">
        <span className="text-xs text-zinc-600">{user.email ?? "—"}</span>
      </td>
      <td className="px-4 py-3">
        <span className="text-xs text-zinc-600">
          {formatDistanceToNow(new Date(user.created_at), { addSuffix: true })}
        </span>
      </td>
      <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={() => patch.mutate({ username: user.username, body: { is_active: !user.is_active } })}
          disabled={patch.isPending}
          className="flex items-center gap-1.5 transition-colors"
        >
          {user.is_active
            ? <ToggleRight className="w-5 h-5 text-emerald-500" />
            : <ToggleLeft  className="w-5 h-5 text-zinc-700" />}
          <span className={clsx("text-xs", user.is_active ? "text-emerald-500" : "text-zinc-600")}>
            {user.is_active ? "Active" : "Inactive"}
          </span>
        </button>
      </td>
    </tr>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export function UserTable() {
  const [offset, setOffset] = useState(0);
  const [showForm, setShowForm] = useState(false);
  const { data, isLoading, isError, refetch } = useAdminUsers({ limit: PAGE_SIZE, offset });
  const users = data?.users ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-zinc-600">
          {total > 0 ? `${total} user${total !== 1 ? "s" : ""}` : ""}
        </p>
        <button onClick={() => setShowForm(true)} className="btn-primary text-xs px-3 py-1.5">
          <Plus className="w-3.5 h-3.5" /> New user
        </button>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-zinc-600 py-12 justify-center">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm">Loading users…</span>
        </div>
      )}

      {isError && (
        <div className="card border-rose-500/20 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load users.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {!isLoading && !isError && users.length === 0 && (
        <div className="card text-center py-16">
          <Users className="w-10 h-10 text-zinc-800 mx-auto mb-3" />
          <p className="text-zinc-500 text-sm font-medium">No users yet.</p>
          <p className="text-zinc-700 text-xs mt-1">Register the first user above.</p>
        </div>
      )}

      {users.length > 0 && (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-zinc-200 dark:border-zinc-800">
                {["Username", "Role", "App", "Email", "Created", "Status"].map((h) => (
                  <th key={h} className="px-4 py-2.5 section-heading whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {users.map((u) => <UserRow key={u.username} user={u} />)}
            </tbody>
          </table>

          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-200 dark:border-zinc-800">
              <span className="text-xs text-zinc-600">Page {currentPage} of {totalPages}</span>
              <div className="flex gap-2">
                <button onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                  disabled={offset === 0} className="btn-secondary px-2 py-1">
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <button onClick={() => setOffset((o) => o + PAGE_SIZE)}
                  disabled={offset + PAGE_SIZE >= total} className="btn-secondary px-2 py-1">
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {showForm && <NewUserModal onClose={() => { setShowForm(false); refetch(); }} />}
    </div>
  );
}

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

// ── New user form ─────────────────────────────────────────────────────────────
interface NewUserForm {
  username: string; password: string;
  role: "user" | "admin"; email: string; app_id: string;
}
const EMPTY: NewUserForm = { username: "", password: "", role: "user", email: "", app_id: "default" };

function NewUserDrawer({ onClose }: { onClose: () => void }) {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const [form, setForm] = useState<NewUserForm>(EMPTY);
  const [error, setError] = useState("");

  const register = useMutation({
    mutationFn: (body: NewUserForm) => createApiClient(token).register(body),
    onSuccess: () => { onClose(); },
    onError: (err: Error) => setError(err.message ?? "Registration failed."),
  });

  function f(patch: Partial<NewUserForm>) { setForm((p) => ({ ...p, ...patch })); }

  async function submit(e: React.FormEvent) {
    e.preventDefault(); setError("");
    if (!form.username.trim() || !form.password.trim()) {
      setError("Username and password are required."); return;
    }
    register.mutate(form);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 w-full sm:max-w-lg bg-slate-900 border border-slate-700
                      rounded-t-2xl sm:rounded-2xl shadow-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-semibold text-slate-100">Register user</h2>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300"><X className="w-4 h-4" /></button>
        </div>
        <form onSubmit={submit} className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div><label className="label">Username</label>
              <input type="text" className="input mt-1" value={form.username}
                onChange={(e) => f({ username: e.target.value })} placeholder="alice" /></div>
            <div><label className="label">Password</label>
              <input type="password" className="input mt-1" value={form.password}
                onChange={(e) => f({ password: e.target.value })} placeholder="••••••••" /></div>
            <div><label className="label">Role</label>
              <select className="input mt-1" value={form.role}
                onChange={(e) => f({ role: e.target.value as "user" | "admin" })}>
                <option value="user">user</option>
                <option value="admin">admin</option>
              </select></div>
            <div><label className="label">App ID</label>
              <input type="text" className="input mt-1" value={form.app_id}
                onChange={(e) => f({ app_id: e.target.value })} placeholder="default" /></div>
            <div className="col-span-2"><label className="label">Email (optional)</label>
              <input type="email" className="input mt-1" value={form.email}
                onChange={(e) => f({ email: e.target.value })} placeholder="alice@example.com" /></div>
          </div>
          {error && <p className="text-xs text-rose-400">{error}</p>}
          <div className="flex gap-3 pt-1">
            <button type="button" onClick={onClose} className="btn-secondary flex-1">Cancel</button>
            <button type="submit" disabled={register.isPending}
              className="btn-primary flex-1 flex items-center justify-center gap-2">
              {register.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}Create
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

  function toggleActive() {
    patch.mutate({ username: user.username, body: { is_active: !user.is_active } });
  }

  return (
    <tr
      className="border-b border-slate-800 hover:bg-slate-800/30 cursor-pointer transition-colors"
      onClick={() => router.push(`/admin/users/${user.username}`)}
    >
      <td className="px-4 py-3">
        <span className="text-sm font-medium text-slate-200 font-mono">{user.username}</span>
      </td>
      <td className="px-4 py-3">
        <span className={clsx(
          "badge border text-xs",
          user.role === "admin"
            ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
            : "bg-slate-800 text-slate-400 border-slate-700/50",
        )}>
          {user.role === "admin"
            ? <ShieldCheck className="w-3 h-3" />
            : <Shield className="w-3 h-3" />}
          {user.role}
        </span>
      </td>
      <td className="px-4 py-3 text-xs text-slate-500">{user.app_id}</td>
      <td className="px-4 py-3 text-xs text-slate-500">{user.email ?? "—"}</td>
      <td className="px-4 py-3 text-xs text-slate-500">
        {formatDistanceToNow(new Date(user.created_at), { addSuffix: true })}
      </td>
      <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={toggleActive}
          disabled={patch.isPending}
          className="flex items-center gap-1 text-xs transition-colors"
          title={user.is_active ? "Deactivate" : "Activate"}
        >
          {user.is_active
            ? <ToggleRight className="w-5 h-5 text-emerald-400" />
            : <ToggleLeft  className="w-5 h-5 text-slate-600"  />}
          <span className={user.is_active ? "text-emerald-400" : "text-slate-600"}>
            {user.is_active ? "Active" : "Inactive"}
          </span>
        </button>
      </td>
    </tr>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
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
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-500">
          {total > 0 ? `${total} user${total !== 1 ? "s" : ""}` : ""}
        </p>
        <button
          onClick={() => setShowForm(true)}
          className="btn-primary flex items-center gap-1.5 text-xs px-3 py-1.5"
        >
          <Plus className="w-3.5 h-3.5" /> New user
        </button>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-slate-500 py-12 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading users…</span>
        </div>
      )}

      {isError && (
        <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load users.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {!isLoading && !isError && users.length === 0 && (
        <div className="card text-center py-16">
          <Users className="w-10 h-10 text-slate-700 mx-auto mb-3" />
          <p className="text-slate-400 text-sm font-medium">No users yet.</p>
          <p className="text-slate-600 text-xs mt-1">Register the first user above.</p>
        </div>
      )}

      {users.length > 0 && (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/50">
                {["Username", "Role", "App", "Email", "Created", "Status"].map((h) => (
                  <th key={h} className="px-4 py-2.5 text-xs font-medium text-slate-500 uppercase tracking-wide">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {users.map((u) => <UserRow key={u.username} user={u} />)}
            </tbody>
          </table>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-slate-800">
              <span className="text-xs text-slate-500">
                Page {currentPage} of {totalPages}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                  disabled={offset === 0}
                  className="btn-secondary px-2 py-1 disabled:opacity-40"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setOffset((o) => o + PAGE_SIZE)}
                  disabled={offset + PAGE_SIZE >= total}
                  className="btn-secondary px-2 py-1 disabled:opacity-40"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {showForm && <NewUserDrawer onClose={() => { setShowForm(false); refetch(); }} />}
    </div>
  );
}

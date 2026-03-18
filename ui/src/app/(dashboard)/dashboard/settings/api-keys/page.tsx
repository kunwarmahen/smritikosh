"use client";

import { useState } from "react";
import { Key, Plus, Trash2, Copy, Check, Loader2, Eye, EyeOff } from "lucide-react";
import { clsx } from "clsx";
import { formatDistanceToNow } from "date-fns";
import { useApiKeys, useCreateApiKey, useRevokeApiKey, type ApiKeyItem } from "@/hooks/useApiKeys";

// ── New key modal ──────────────────────────────────────────────────────────────

function NewKeyModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [appId, setAppId] = useState("default");
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");

  const create = useCreateApiKey();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!name.trim()) { setError("Name is required."); return; }
    create.mutate(
      { name: name.trim(), app_id: appId },
      {
        onSuccess: (data) => setCreatedKey(data.key),
        onError: (err) => setError(err.message ?? "Failed to create key."),
      },
    );
  }

  async function copyKey() {
    if (!createdKey) return;
    await navigator.clipboard.writeText(createdKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-zinc-950/80 backdrop-blur-sm" onClick={!createdKey ? onClose : undefined} />
      <div className="relative z-10 w-full max-w-md bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-100">
            {createdKey ? "Key created" : "New API key"}
          </h2>
        </div>

        {/* Success state — show key once */}
        {createdKey ? (
          <div className="p-5 space-y-4">
            <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2.5">
              <p className="text-xs text-amber-400 font-medium mb-1">Store this key now</p>
              <p className="text-xs text-amber-500/80">
                It won't be shown again. If you lose it, revoke and generate a new one.
              </p>
            </div>

            <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 flex items-center gap-2">
              <code className="flex-1 text-xs text-violet-300 font-mono break-all">{createdKey}</code>
              <button onClick={copyKey} className="flex-shrink-0 text-zinc-500 hover:text-zinc-300 transition-colors">
                {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>

            <button onClick={onClose} className="btn-primary w-full justify-center">
              Done
            </button>
          </div>
        ) : (
          <form onSubmit={submit} className="p-5 space-y-3">
            <div>
              <label className="label">Key name</label>
              <input
                type="text"
                className="input"
                placeholder="My n8n integration"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
              />
            </div>
            <div>
              <label className="label">App ID</label>
              <input
                type="text"
                className="input"
                placeholder="default"
                value={appId}
                onChange={(e) => setAppId(e.target.value)}
              />
            </div>

            {error && (
              <p className="text-xs text-rose-400 bg-rose-500/8 border border-rose-500/20 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <div className="flex gap-2 pt-1">
              <button type="button" onClick={onClose} className="btn-secondary flex-1">Cancel</button>
              <button type="submit" disabled={create.isPending} className="btn-primary flex-1 justify-center">
                {create.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                Generate key
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}

// ── Key row ───────────────────────────────────────────────────────────────────

function KeyRow({ apiKey }: { apiKey: ApiKeyItem }) {
  const revoke = useRevokeApiKey();
  const [confirm, setConfirm] = useState(false);

  return (
    <div className="flex items-center justify-between py-3 border-b border-zinc-800/60 last:border-0">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-7 h-7 rounded-lg bg-zinc-800 flex items-center justify-center flex-shrink-0">
          <Key className="w-3.5 h-3.5 text-zinc-500" />
        </div>
        <div className="min-w-0">
          <p className="text-sm text-zinc-200 truncate">{apiKey.name}</p>
          <div className="flex items-center gap-2 mt-0.5">
            <code className="text-xs text-zinc-600 font-mono">
              sk-smriti-{apiKey.key_prefix}…
            </code>
            <span className="text-zinc-700">·</span>
            <span className="text-xs text-zinc-600">{apiKey.app_id}</span>
            <span className="text-zinc-700">·</span>
            <span className="text-xs text-zinc-600">
              {apiKey.last_used_at
                ? `Used ${formatDistanceToNow(new Date(apiKey.last_used_at), { addSuffix: true })}`
                : "Never used"}
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-shrink-0 ml-4">
        {!confirm ? (
          <button
            onClick={() => setConfirm(true)}
            className="text-zinc-600 hover:text-rose-400 transition-colors p-1.5 rounded-md hover:bg-rose-500/10"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        ) : (
          <div className="flex items-center gap-1.5">
            <button onClick={() => setConfirm(false)} className="btn-secondary text-xs px-2 py-1">
              Cancel
            </button>
            <button
              onClick={() => revoke.mutate(apiKey.id)}
              disabled={revoke.isPending}
              className="btn-danger text-xs px-2 py-1"
            >
              {revoke.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : "Revoke"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function ApiKeysPage() {
  const [showModal, setShowModal] = useState(false);
  const { data: keys = [], isLoading, isError, refetch } = useApiKeys();

  return (
    <div className="max-w-2xl">
      <div className="mb-6">
        <h1 className="page-title">API Keys</h1>
        <p className="text-xs text-zinc-500 mt-1">
          Use API keys to authenticate SDK and programmatic access. Keys never expire unless revoked.
        </p>
      </div>

      <div className="space-y-3">
        {/* Keys list */}
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <p className="section-heading">Active keys</p>
            <button onClick={() => setShowModal(true)} className="btn-primary text-xs px-3 py-1.5">
              <Plus className="w-3.5 h-3.5" /> New key
            </button>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-4 h-4 animate-spin text-zinc-600" />
            </div>
          )}

          {isError && (
            <div className="text-center py-8">
              <p className="text-sm text-rose-400">Failed to load keys.</p>
              <button onClick={() => refetch()} className="btn-secondary mt-3 text-xs">Retry</button>
            </div>
          )}

          {!isLoading && !isError && keys.length === 0 && (
            <div className="text-center py-10">
              <Key className="w-8 h-8 text-zinc-800 mx-auto mb-2" />
              <p className="text-sm text-zinc-500">No API keys yet.</p>
              <p className="text-xs text-zinc-700 mt-1">Generate a key to use the SDK or external integrations.</p>
            </div>
          )}

          {keys.length > 0 && (
            <div>
              {keys.map((k) => <KeyRow key={k.id} apiKey={k} />)}
            </div>
          )}
        </div>

        {/* Usage guide */}
        <div className="card">
          <p className="section-heading mb-3">Using your key</p>
          <p className="text-xs text-zinc-500 mb-2">Pass the key as a Bearer token in the Authorization header:</p>
          <pre className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2.5 text-xs text-violet-300 font-mono overflow-x-auto">
{`Authorization: Bearer sk-smriti-your-key-here`}
          </pre>
          <p className="text-xs text-zinc-600 mt-3">
            The key is scoped to your user account and app ID. It can be used anywhere a session token is accepted.
          </p>
        </div>
      </div>

      {showModal && <NewKeyModal onClose={() => setShowModal(false)} />}
    </div>
  );
}

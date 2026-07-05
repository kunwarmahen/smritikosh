"use client";

import { useState } from "react";
import { ArrowRight, Loader2, Plus, Share2, Trash2 } from "lucide-react";
import { clsx } from "clsx";
import { formatDistanceToNow } from "date-fns";
import { useSession } from "next-auth/react";
import { useConsents, useGrantConsent, useRevokeConsent, type ConsentItem } from "@/hooks/useConsents";
import { FACT_CATEGORIES } from "@/lib/fact-categories";

// ── New grant modal ───────────────────────────────────────────────────────────

function NewGrantModal({ onClose }: { onClose: () => void }) {
  const { data: session } = useSession();
  const availableAppIds: string[] = session?.user?.appIds ?? ["default"];

  const [sourceApp, setSourceApp] = useState<string>(availableAppIds[0] ?? "");
  const [targetApp, setTargetApp] = useState<string>("");
  const [customTarget, setCustomTarget] = useState("");
  const [allCategories, setAllCategories] = useState(true);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [error, setError] = useState("");

  const grant = useGrantConsent();

  const effectiveTarget = customTarget.trim() || targetApp;

  function toggleCategory(value: string) {
    setSelectedCategories((prev) =>
      prev.includes(value) ? prev.filter((c) => c !== value) : [...prev, value]
    );
  }

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!sourceApp) { setError("Select a source app."); return; }
    if (!effectiveTarget) { setError("Select or enter a target app."); return; }
    if (effectiveTarget === sourceApp) { setError("Source and target apps must differ."); return; }
    if (!allCategories && selectedCategories.length === 0) {
      setError("Select at least one category, or share all categories.");
      return;
    }
    grant.mutate(
      {
        source_app_id: sourceApp,
        target_app_id: effectiveTarget,
        categories: allCategories ? [] : selectedCategories,
      },
      {
        onSuccess: onClose,
        onError: (err) => setError(err.message ?? "Failed to create grant."),
      },
    );
  }

  const appChip = (id: string, selected: boolean, onClick: () => void) => (
    <button
      key={id}
      type="button"
      onClick={onClick}
      className={clsx(
        "px-2.5 py-1 rounded-md text-xs font-mono border transition-colors",
        selected
          ? "bg-violet-500/15 border-violet-500/40 text-violet-700 dark:text-violet-300"
          : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300",
      )}
    >
      {id}
    </button>
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-zinc-950/80 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 w-full max-w-md bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">New sharing grant</h2>
        </div>

        <form onSubmit={submit} className="p-5 space-y-4">
          <div>
            <label className="label">Share facts learned in</label>
            <div className="flex flex-wrap gap-2 mt-1">
              {availableAppIds.map((id) =>
                appChip(id, sourceApp === id, () => setSourceApp(id))
              )}
            </div>
          </div>

          <div>
            <label className="label">With app</label>
            <div className="flex flex-wrap gap-2 mt-1">
              {availableAppIds
                .filter((id) => id !== sourceApp)
                .map((id) =>
                  appChip(id, !customTarget.trim() && targetApp === id, () => {
                    setTargetApp(id);
                    setCustomTarget("");
                  })
                )}
            </div>
            <input
              type="text"
              className="input mt-2"
              placeholder="or enter another app id"
              value={customTarget}
              onChange={(e) => setCustomTarget(e.target.value)}
            />
            <p className="text-[11px] text-zinc-500 mt-1.5">
              The target app will see the shared facts in its context reads, tagged with their source app.
            </p>
          </div>

          <div>
            <label className="label">Categories</label>
            <div className="flex flex-wrap gap-2 mt-1 mb-2">
              <button
                type="button"
                onClick={() => setAllCategories(true)}
                className={clsx(
                  "px-2.5 py-1 rounded-md text-xs font-medium border transition-colors",
                  allCategories
                    ? "bg-violet-500/15 border-violet-500/40 text-violet-700 dark:text-violet-300"
                    : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300",
                )}
              >
                All categories
              </button>
              <button
                type="button"
                onClick={() => setAllCategories(false)}
                className={clsx(
                  "px-2.5 py-1 rounded-md text-xs font-medium border transition-colors",
                  !allCategories
                    ? "bg-violet-500/15 border-violet-500/40 text-violet-700 dark:text-violet-300"
                    : "bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300",
                )}
              >
                Only selected
              </button>
            </div>
            {!allCategories && (
              <div className="flex flex-wrap gap-1.5 p-2.5 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 max-h-44 overflow-y-auto">
                {FACT_CATEGORIES.map((c) => (
                  <button
                    key={c.value}
                    type="button"
                    onClick={() => toggleCategory(c.value)}
                    className={clsx(
                      "px-2 py-0.5 rounded-full text-[11px] border transition-colors",
                      selectedCategories.includes(c.value)
                        ? "bg-violet-500/15 border-violet-500/40 text-violet-700 dark:text-violet-300"
                        : "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-700 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300",
                    )}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            )}
            <p className="text-[11px] text-zinc-500 mt-1.5">
              Sensitive categories like health, finance and religion are only shared if you pick them explicitly here.
            </p>
          </div>

          {error && (
            <p className="text-xs text-rose-400 bg-rose-500/8 border border-rose-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <div className="flex gap-2 pt-1">
            <button type="button" onClick={onClose} className="btn-secondary flex-1">Cancel</button>
            <button type="submit" disabled={grant.isPending} className="btn-primary flex-1 justify-center">
              {grant.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              Grant access
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ── Grant row ─────────────────────────────────────────────────────────────────

function GrantRow({ consent }: { consent: ConsentItem }) {
  const revoke = useRevokeConsent();
  const [confirm, setConfirm] = useState(false);

  const categoryLabel = (v: string) =>
    FACT_CATEGORIES.find((c) => c.value === v)?.label ?? v;

  return (
    <div
      className={clsx(
        "py-3 border-b border-zinc-200 dark:border-zinc-800/60 last:border-0",
        !consent.active && "opacity-50",
      )}
    >
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <code className="text-sm font-mono text-zinc-800 dark:text-zinc-200">{consent.source_app_id}</code>
            <ArrowRight className="w-3.5 h-3.5 text-zinc-400 dark:text-zinc-600 flex-shrink-0" />
            <code className="text-sm font-mono text-zinc-800 dark:text-zinc-200">{consent.target_app_id}</code>
            {!consent.active && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full border font-medium bg-zinc-100 dark:bg-zinc-800 border-zinc-300 dark:border-zinc-700 text-zinc-500">
                revoked
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 mt-1.5 flex-wrap">
            {consent.categories.length === 0 ? (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full border font-medium bg-amber-500/10 border-amber-500/30 text-amber-600 dark:text-amber-400">
                all categories
              </span>
            ) : (
              consent.categories.map((c) => (
                <span
                  key={c}
                  className="text-[10px] px-1.5 py-0.5 rounded-full border font-medium bg-violet-500/10 border-violet-500/30 text-violet-600 dark:text-violet-400"
                >
                  {categoryLabel(c)}
                </span>
              ))
            )}
          </div>

          <p className="text-xs text-zinc-500 mt-1.5">
            {consent.active
              ? `Granted ${formatDistanceToNow(new Date(consent.granted_at), { addSuffix: true })}`
              : consent.revoked_at
              ? `Revoked ${formatDistanceToNow(new Date(consent.revoked_at), { addSuffix: true })}`
              : "Revoked"}
          </p>
        </div>

        {consent.active && (
          <div className="flex items-center gap-2 flex-shrink-0">
            {!confirm ? (
              <button
                onClick={() => setConfirm(true)}
                title="Revoke grant"
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
                  onClick={() =>
                    revoke.mutate({
                      source_app_id: consent.source_app_id,
                      target_app_id: consent.target_app_id,
                    })
                  }
                  disabled={revoke.isPending}
                  className="btn-danger text-xs px-2 py-1"
                >
                  {revoke.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : "Revoke"}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function SharingPage() {
  const [showModal, setShowModal] = useState(false);
  const [showRevoked, setShowRevoked] = useState(false);
  const { data: consents = [], isLoading, isError, refetch } = useConsents(showRevoked);

  return (
    <div className="max-w-2xl">
      <div className="mb-6">
        <h1 className="page-title">Cross-app sharing</h1>
        <p className="text-xs text-zinc-500 mt-1">
          Grant one app read access to facts learned in another. Grants are per category, revocable
          any time, and every cross-app read is recorded in your audit trail.
        </p>
      </div>

      <div className="space-y-3">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <p className="section-heading">Sharing grants</p>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 text-xs text-zinc-500 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showRevoked}
                  onChange={(e) => setShowRevoked(e.target.checked)}
                  className="accent-violet-600"
                />
                Show revoked
              </label>
              <button onClick={() => setShowModal(true)} className="btn-primary text-xs px-3 py-1.5">
                <Plus className="w-3.5 h-3.5" /> New grant
              </button>
            </div>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-4 h-4 animate-spin text-zinc-600" />
            </div>
          )}

          {isError && (
            <div className="text-center py-8">
              <p className="text-sm text-rose-400">Failed to load grants.</p>
              <button onClick={() => refetch()} className="btn-secondary mt-3 text-xs">Retry</button>
            </div>
          )}

          {!isLoading && !isError && consents.length === 0 && (
            <div className="text-center py-10">
              <Share2 className="w-8 h-8 text-zinc-300 dark:text-zinc-800 mx-auto mb-2" />
              <p className="text-sm text-zinc-500">No sharing grants yet.</p>
              <p className="text-xs text-zinc-400 dark:text-zinc-700 mt-1">
                Your apps are fully isolated until you grant access.
              </p>
            </div>
          )}

          {consents.length > 0 && (
            <div>
              {consents.map((c) => <GrantRow key={c.consent_id} consent={c} />)}
            </div>
          )}
        </div>

        <div className="card">
          <p className="section-heading mb-3">How sharing works</p>
          <ul className="text-xs text-zinc-500 space-y-1.5 list-disc pl-4">
            <li>Shared facts appear in the target app&apos;s context reads, tagged with the app they came from.</li>
            <li>If both apps know the same fact, the target app&apos;s own version wins.</li>
            <li>Revoking takes effect on the next read — nothing is copied between apps.</li>
            <li>Every grant, revocation and cross-app read is logged in the audit trail.</li>
          </ul>
        </div>
      </div>

      {showModal && <NewGrantModal onClose={() => setShowModal(false)} />}
    </div>
  );
}

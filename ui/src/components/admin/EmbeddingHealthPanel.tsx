"use client";

import { clsx } from "clsx";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { useState } from "react";
import { useEmbeddingHealth, useAdminReEmbed } from "@/hooks/useAdmin";

function Stat({ label, value, highlight }: { label: string; value: React.ReactNode; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-zinc-200 dark:border-zinc-800/60 last:border-0">
      <span className="text-sm text-zinc-500">{label}</span>
      <span className={clsx("mono text-sm", highlight ? "text-rose-400 font-semibold" : "text-zinc-400")}>
        {value}
      </span>
    </div>
  );
}

export function EmbeddingHealthPanel() {
  const { data, isLoading, refetch, isFetching } = useEmbeddingHealth();
  const reEmbed = useAdminReEmbed();
  const [reEmbedMsg, setReEmbedMsg] = useState<string | null>(null);

  async function handleReEmbed() {
    setReEmbedMsg(null);
    try {
      const result = await reEmbed.mutateAsync();
      setReEmbedMsg(
        result.queued === 0
          ? "No stale embeddings — nothing to re-embed."
          : `Re-embed started: ${result.queued} event${result.queued !== 1 ? "s" : ""} queued.`,
      );
    } catch {
      setReEmbedMsg("Re-embed request failed.");
    }
  }

  const isStale = !!data && (data.stale_events > 0 || data.null_embeddings > 0);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <p className="section-heading">Embedding health</p>
          {data && (
            <span
              className={clsx(
                "inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-md border",
                data.healthy
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : "bg-rose-500/10 text-rose-400 border-rose-500/20",
              )}
            >
              {data.healthy ? (
                <><CheckCircle2 className="w-3 h-3" /> healthy</>
              ) : (
                <><XCircle className="w-3 h-3" /> stale</>
              )}
            </span>
          )}
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="text-zinc-600 hover:text-zinc-400 transition-colors"
          title="Refresh"
        >
          <RefreshCw className={clsx("w-3.5 h-3.5", isFetching && "animate-spin")} />
        </button>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 text-zinc-600 py-6">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm">Checking…</span>
        </div>
      ) : data ? (
        <>
          <Stat label="Configured dimension" value={data.configured_dim} />
          <Stat label="Embedded events" value={data.total_embedded} />
          <Stat
            label="Stale (wrong dimension)"
            value={data.stale_events}
            highlight={data.stale_events > 0}
          />
          <Stat
            label="Missing embedding"
            value={data.null_embeddings}
            highlight={data.null_embeddings > 0}
          />

          {isStale && (
            <div className="mt-4 flex items-center gap-3">
              <button
                onClick={handleReEmbed}
                disabled={reEmbed.isPending}
                className={clsx(
                  "flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg font-medium transition-colors",
                  reEmbed.isPending
                    ? "bg-zinc-200 dark:bg-zinc-800 text-zinc-500 cursor-not-allowed"
                    : "bg-violet-600 hover:bg-violet-500 text-white",
                )}
              >
                {reEmbed.isPending ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RotateCcw className="w-3.5 h-3.5" />
                )}
                Re-embed stale events
              </button>
              {reEmbedMsg && (
                <p className="text-xs text-zinc-500">{reEmbedMsg}</p>
              )}
            </div>
          )}

          {!isStale && reEmbedMsg && (
            <p className="mt-3 text-xs text-zinc-500">{reEmbedMsg}</p>
          )}

          {!isStale && (
            <div className="mt-4 flex items-center gap-1.5 text-xs text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" />
              All embeddings match the configured dimension.
            </div>
          )}

          {isStale && !reEmbedMsg && (
            <div className="mt-2 flex items-center gap-1.5 text-xs text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5" />
              Stale embeddings degrade similarity search. Run re-embed to fix.
            </div>
          )}
        </>
      ) : (
        <p className="text-sm text-zinc-600 py-4">Could not reach the API.</p>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import { Loader2, Tag, Inbox } from "lucide-react";
import { clsx } from "clsx";
import { useRecentEvents } from "@/hooks/useMemory";
import { MemoryCard } from "@/components/memory/MemoryCard";
import { useRouter } from "next/navigation";
import type { MemoryEvent } from "@/types";

function groupByClusters(events: MemoryEvent[]) {
  const map = new Map<string, MemoryEvent[]>();
  for (const e of events) {
    const key = e.cluster_label ?? "Unclustered";
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(e);
  }
  return [...map.entries()].sort(([a], [b]) => {
    if (a === "Unclustered") return 1;
    if (b === "Unclustered") return -1;
    return a.localeCompare(b);
  });
}

export default function ClustersPage() {
  const router = useRouter();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const { data, isLoading, isError, refetch } = useRecentEvents({ limit: 200 });

  const groups = groupByClusters(data?.events ?? []);

  function toggle(label: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(label) ? next.delete(label) : next.add(label);
      return next;
    });
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100 tracking-tight">Clusters</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Memories grouped by topic via semantic clustering.
        </p>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-zinc-500 py-12 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading clusters…</span>
        </div>
      )}

      {isError && (
        <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load memories.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {!isLoading && !isError && groups.length === 0 && (
        <div className="card text-center py-16">
          <Tag className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-zinc-400 text-sm font-medium">No clusters yet.</p>
          <p className="text-zinc-600 text-xs mt-1">
            Clusters are computed automatically as memories accumulate.
          </p>
        </div>
      )}

      <div className="space-y-3">
        {groups.map(([label, events]) => {
          const isOpen = expanded.has(label);
          const isUnclustered = label === "Unclustered";
          return (
            <div key={label} className="card">
              <button
                onClick={() => toggle(label)}
                className="w-full flex items-center justify-between gap-3 text-left"
              >
                <div className="flex items-center gap-2">
                  <Tag className={clsx("w-3.5 h-3.5", isUnclustered ? "text-zinc-600" : "text-violet-400")} />
                  <span className={clsx(
                    "text-sm font-medium",
                    isUnclustered ? "text-zinc-500" : "text-zinc-800 dark:text-zinc-200",
                  )}>
                    {label}
                  </span>
                  <span className="badge bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border border-zinc-200 dark:border-zinc-700/50 text-xs">
                    {events.length}
                  </span>
                </div>
                <span className="text-xs text-zinc-600">{isOpen ? "Hide" : "Show"}</span>
              </button>

              {isOpen && (
                <div className="mt-4 space-y-3">
                  {events.map((event) => (
                    <MemoryCard
                      key={event.event_id}
                      event={event}
                      onViewGraph={(id) => router.push(`/dashboard/memories/${id}`)}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

"use client";

import { use } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Network } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { clsx } from "clsx";
import { useMemoryEvent } from "@/hooks/useMemoryGraph";
import { useEventLineage } from "@/hooks/useAudit";
import { MemoryGraphView } from "@/components/memory/MemoryGraphView";
import { importanceLevel, EVENT_TYPE_LABELS, EVENT_TYPE_COLORS } from "@/types";

export default function MemoryGraphPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const { data: event } = useMemoryEvent(id);
  const lineage = useEventLineage(id);
  const auditEvents = lineage.data ?? [];

  const level = event ? importanceLevel(event.importance_score) : null;
  const IMPORTANCE_COLORS = {
    high:   "text-emerald-400 border-emerald-500/30 bg-emerald-500/10",
    medium: "text-amber-400 border-amber-500/30 bg-amber-500/10",
    low:    "text-rose-400 border-rose-500/30 bg-rose-500/10",
  };

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
        <div className="min-w-0">
          <h1 className="text-base font-semibold text-zinc-100 tracking-tight flex items-center gap-2">
            <Network className="w-5 h-5 text-violet-400 flex-shrink-0" />
            Memory Graph
          </h1>
          <p className="text-xs text-zinc-500 mt-0.5 font-mono truncate">{id}</p>
        </div>
      </div>

      {/* Event summary card */}
      {event && (
        <div className="card mb-6">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            {level && (
              <span className={clsx("badge border text-xs", IMPORTANCE_COLORS[level])}>
                {(event.importance_score * 100).toFixed(0)}% importance
              </span>
            )}
            {event.consolidated && (
              <span className="badge bg-slate-700/50 text-zinc-400 border border-slate-600/50 text-xs">
                Consolidated
              </span>
            )}
            {event.cluster_label && (
              <span className="badge bg-violet-500/10 text-violet-400 border border-violet-500/20 text-xs">
                {event.cluster_label}
              </span>
            )}
            <span className="text-xs text-zinc-500 ml-auto">
              {formatDistanceToNow(new Date(event.created_at), { addSuffix: true })}
            </span>
          </div>
          <p className="text-sm text-zinc-300 leading-relaxed">{event.raw_text}</p>
          {event.summary && (
            <div className="mt-3 pt-3 border-t border-zinc-800">
              <p className="text-xs text-zinc-500 mb-1 font-medium">Summary</p>
              <p className="text-xs text-zinc-400 leading-relaxed">{event.summary}</p>
            </div>
          )}
        </div>
      )}

      {/* Narrative graph */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-zinc-400 mb-3">Narrative Links</h2>
        <MemoryGraphView eventId={id} />
      </div>

      {/* Audit lineage */}
      {auditEvents.length > 0 && (
        <div>
          <h2 className="text-sm font-medium text-zinc-400 mb-3">Audit Lineage</h2>
          <div className="space-y-2">
            {auditEvents.map((evt) => (
              <div key={evt.id} className="card py-2.5 px-4 flex items-center gap-3">
                <span className={clsx(
                  "badge border text-xs bg-zinc-800/50 border-zinc-700/50",
                  EVENT_TYPE_COLORS[evt.event_type] ?? "text-zinc-400",
                )}>
                  {EVENT_TYPE_LABELS[evt.event_type] ?? evt.event_type}
                </span>
                <span className="text-xs text-zinc-500 ml-auto">
                  {formatDistanceToNow(new Date(evt.timestamp), { addSuffix: true })}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

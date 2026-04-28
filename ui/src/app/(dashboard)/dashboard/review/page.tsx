"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { formatDistanceToNow } from "date-fns";
import {
  Loader2,
  CheckCircle2,
  Trash2,
  Network,
  SlidersHorizontal,
  ThumbsUp,
  GitMerge,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { clsx } from "clsx";
import { useRecentEvents, useSubmitFeedback, useDeleteEvent } from "@/hooks/useMemory";
import { useContradictions, useResolveContradiction } from "@/hooks/useFacts";
import { SourceBadge, isAutoExtracted } from "@/components/memory/SourceBadge";
import type { MemoryEvent } from "@/types";

const SOURCE_FILTER_OPTIONS = [
  { value: "all",                  label: "All sources" },
  { value: "passive_distillation", label: "Distilled" },
  { value: "passive_streaming",    label: "Streaming" },
  { value: "trigger_word",         label: "Triggered" },
  { value: "sdk_middleware",       label: "SDK" },
  { value: "webhook_ingest",       label: "Webhook" },
  { value: "tool_use",             label: "Tool" },
  { value: "media_voice",          label: "Voice Note" },
  { value: "media_document",       label: "Document" },
  { value: "media_image",          label: "Image" },
];

function ReviewCard({ event, onApproved }: { event: MemoryEvent; onApproved: () => void }) {
  const router = useRouter();
  const [approved, setApproved] = useState(false);
  const submitFeedback = useSubmitFeedback();
  const deleteEvent = useDeleteEvent();

  const timeAgo = (() => {
    try {
      const ts = event.created_at.endsWith("Z") ? event.created_at : event.created_at + "Z";
      return formatDistanceToNow(new Date(ts), { addSuffix: true });
    } catch { return ""; }
  })();

  async function handleApprove() {
    setApproved(true);
    await submitFeedback.mutateAsync({ event_id: event.event_id, feedback_type: "positive" });
    onApproved();
  }

  async function handleDelete() {
    if (!confirm("Remove this memory?")) return;
    await deleteEvent.mutateAsync(event.event_id);
  }

  return (
    <div
      className={clsx(
        "group bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl px-4 py-3.5",
        "border-l-2 border-l-amber-500/50 transition-all duration-200",
        approved && "opacity-50 scale-[0.99]",
      )}
    >
      {/* Source + time */}
      <div className="flex items-center justify-between gap-3 mb-2.5">
        <div className="flex items-center gap-2 flex-wrap">
          {event.source_type && <SourceBadge sourceType={event.source_type} />}
          {event.consolidated && (
            <span className="badge bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border border-zinc-200 dark:border-zinc-700/60 text-xs">
              consolidated
            </span>
          )}
        </div>
        <span className="text-xs text-zinc-600 flex-shrink-0">{timeAgo}</span>
      </div>

      {/* Text */}
      <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{event.raw_text}</p>

      {/* Summary */}
      {event.summary && event.consolidated && (
        <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-zinc-800">
          <p className="text-xs text-zinc-600 font-medium uppercase tracking-wide mb-1">Summary</p>
          <p className="text-xs text-zinc-500 leading-relaxed">{event.summary}</p>
        </div>
      )}

      {/* Actions */}
      <div
        className="flex items-center justify-between mt-3 pt-2.5 border-t border-zinc-200/80 dark:border-zinc-800/60
                   opacity-0 group-hover:opacity-100 transition-opacity duration-150"
      >
        <button
          onClick={handleApprove}
          disabled={approved || submitFeedback.isPending}
          className={clsx(
            "flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg transition-colors",
            approved
              ? "text-emerald-400 bg-emerald-500/10"
              : "text-zinc-500 hover:text-emerald-400 hover:bg-emerald-500/10",
          )}
        >
          {approved
            ? <><CheckCircle2 className="w-3.5 h-3.5" /> Approved</>
            : <><ThumbsUp className="w-3.5 h-3.5" /> Approve</>
          }
        </button>

        <div className="flex items-center gap-1">
          <button
            onClick={() => router.push(`/dashboard/memories/${event.event_id}`)}
            className="flex items-center gap-1.5 text-xs text-zinc-600 hover:text-violet-400
                       px-2 py-1 rounded hover:bg-violet-500/10 transition-colors"
          >
            <Network className="w-3.5 h-3.5" />
            Graph
          </button>
          <button
            onClick={handleDelete}
            disabled={deleteEvent.isPending}
            className="w-6 h-6 flex items-center justify-center rounded text-zinc-700
                       hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            title="Remove"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

interface ContradictionCardProps {
  contradiction: {
    id: string;
    category: string;
    key: string;
    existing_value: string;
    existing_confidence: number;
    candidate_value: string;
    candidate_source: string;
    candidate_confidence: number;
    created_at: string;
  };
  onResolved: () => void;
}

function ContradictionCard({ contradiction, onResolved }: ContradictionCardProps) {
  const [mergeOpen, setMergeOpen] = useState(false);
  const [mergedValue, setMergedValue] = useState("");
  const resolve = useResolveContradiction();

  const timeAgo = (() => {
    try {
      const ts = contradiction.created_at.endsWith("Z")
        ? contradiction.created_at
        : contradiction.created_at + "Z";
      return formatDistanceToNow(new Date(ts), { addSuffix: true });
    } catch { return ""; }
  })();

  async function handleResolve(keep: "existing" | "candidate" | "merge") {
    if (keep === "merge" && !mergedValue.trim()) return;
    await resolve.mutateAsync({
      id: contradiction.id,
      keep,
      merged_value: keep === "merge" ? mergedValue.trim() : undefined,
    });
    onResolved();
  }

  const isPending = resolve.isPending;

  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl px-4 py-3.5 border-l-2 border-l-rose-500/50">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-400 flex-shrink-0" />
          <span className="text-xs font-medium text-zinc-500 uppercase tracking-wide">
            {contradiction.category} · {contradiction.key}
          </span>
        </div>
        <span className="text-xs text-zinc-600 flex-shrink-0">{timeAgo}</span>
      </div>

      {/* Fact comparison */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="rounded-lg bg-zinc-50 dark:bg-zinc-800/60 border border-zinc-200 dark:border-zinc-700/50 p-3">
          <p className="text-xs font-medium text-zinc-500 mb-1.5">Current</p>
          <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{contradiction.existing_value}</p>
          <p className="text-xs text-zinc-500 mt-1.5">
            conf {(contradiction.existing_confidence * 100).toFixed(0)}%
          </p>
        </div>
        <div className="rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700/40 p-3">
          <p className="text-xs font-medium text-amber-600 dark:text-amber-400 mb-1.5">Candidate</p>
          <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{contradiction.candidate_value}</p>
          <p className="text-xs text-zinc-500 mt-1.5">
            conf {(contradiction.candidate_confidence * 100).toFixed(0)}% · {contradiction.candidate_source}
          </p>
        </div>
      </div>

      {/* Merge text area */}
      {mergeOpen && (
        <div className="mb-3">
          <label className="block text-xs text-zinc-500 mb-1.5">Write the canonical merged value</label>
          <textarea
            value={mergedValue}
            onChange={(e) => setMergedValue(e.target.value)}
            placeholder={`e.g. "${contradiction.existing_value} (updated: ${contradiction.candidate_value})"`}
            rows={2}
            className="w-full text-sm bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700
                       rounded-lg px-3 py-2 text-zinc-700 dark:text-zinc-300 resize-none
                       focus:outline-none focus:ring-1 focus:ring-violet-500/50"
          />
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2 flex-wrap pt-1">
        <button
          onClick={() => handleResolve("existing")}
          disabled={isPending}
          className="text-xs px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-zinc-700
                     text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:hover:text-zinc-200
                     hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors disabled:opacity-50"
        >
          Keep current
        </button>
        <button
          onClick={() => handleResolve("candidate")}
          disabled={isPending}
          className="text-xs px-2.5 py-1.5 rounded-lg border border-amber-300 dark:border-amber-700/50
                     text-amber-700 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20
                     transition-colors disabled:opacity-50"
        >
          Use candidate
        </button>
        <button
          onClick={() => {
            if (mergeOpen && mergedValue.trim()) {
              handleResolve("merge");
            } else {
              setMergeOpen((v) => !v);
            }
          }}
          disabled={isPending}
          className={clsx(
            "flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg border transition-colors disabled:opacity-50",
            mergeOpen && mergedValue.trim()
              ? "border-violet-400 dark:border-violet-600 bg-violet-500/10 text-violet-700 dark:text-violet-300"
              : "border-violet-200 dark:border-violet-800/60 text-violet-600 dark:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-900/20",
          )}
        >
          <GitMerge className="w-3.5 h-3.5" />
          {mergeOpen && mergedValue.trim() ? "Save merged" : mergeOpen ? "Write merge…" : "Merge"}
          {mergeOpen ? <ChevronUp className="w-3 h-3 ml-0.5" /> : <ChevronDown className="w-3 h-3 ml-0.5" />}
        </button>
        {isPending && <Loader2 className="w-3.5 h-3.5 animate-spin text-zinc-400" />}
      </div>
    </div>
  );
}

export default function ReviewPage() {
  const [sourceFilter, setSourceFilter] = useState("all");
  const [approvedIds, setApprovedIds] = useState<Set<string>>(new Set());
  const [resolvedIds, setResolvedIds] = useState<Set<string>>(new Set());
  const [showContradictions, setShowContradictions] = useState(true);

  const { data, isLoading, isError, refetch } = useRecentEvents({ limit: 100 });
  const { data: contradictionsData, isLoading: contradictionsLoading } = useContradictions();

  const autoExtracted = useMemo(() => {
    if (!data?.events) return [];
    return data.events.filter(
      (e: MemoryEvent) =>
        isAutoExtracted(e.source_type) && !approvedIds.has(e.event_id),
    );
  }, [data, approvedIds]);

  const filtered = useMemo(() => {
    if (sourceFilter === "all") return autoExtracted;
    return autoExtracted.filter((e: MemoryEvent) => e.source_type === sourceFilter);
  }, [autoExtracted, sourceFilter]);

  const sourceCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const e of autoExtracted) {
      counts[e.source_type ?? "unknown"] = (counts[e.source_type ?? "unknown"] ?? 0) + 1;
    }
    return counts;
  }, [autoExtracted]);

  const unresolved = useMemo(() => {
    if (!contradictionsData?.contradictions) return [];
    return contradictionsData.contradictions.filter((c) => !resolvedIds.has(c.id));
  }, [contradictionsData, resolvedIds]);

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100 tracking-tight">Review</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Auto-extracted memories waiting for your review. Approve or remove them.
        </p>
      </div>

      {/* Contradictions section */}
      {(unresolved.length > 0 || contradictionsLoading) && (
        <div className="mb-8">
          <button
            onClick={() => setShowContradictions((v) => !v)}
            className="flex items-center gap-2 mb-4 text-sm font-medium text-zinc-700 dark:text-zinc-300"
          >
            <AlertTriangle className="w-4 h-4 text-rose-400" />
            Fact conflicts
            {unresolved.length > 0 && (
              <span className="ml-1 bg-rose-500/15 text-rose-600 dark:text-rose-400 text-xs px-1.5 py-0.5 rounded-full">
                {unresolved.length}
              </span>
            )}
            {showContradictions ? <ChevronUp className="w-4 h-4 ml-auto" /> : <ChevronDown className="w-4 h-4 ml-auto" />}
          </button>

          {showContradictions && (
            <>
              {contradictionsLoading ? (
                <div className="flex items-center gap-2 text-zinc-500 py-4 justify-center">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Loading conflicts…</span>
                </div>
              ) : (
                <div className="space-y-3 mb-2">
                  {unresolved.map((c) => (
                    <ContradictionCard
                      key={c.id}
                      contradiction={c}
                      onResolved={() => setResolvedIds((prev) => new Set([...prev, c.id]))}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Filter bar */}
      <div className="flex items-center gap-3 mb-5">
        <SlidersHorizontal className="w-4 h-4 text-zinc-600 flex-shrink-0" />
        <div className="flex items-center gap-2 flex-wrap">
          {SOURCE_FILTER_OPTIONS.map((opt) => {
            const count = opt.value === "all"
              ? autoExtracted.length
              : (sourceCounts[opt.value] ?? 0);
            if (opt.value !== "all" && count === 0) return null;
            return (
              <button
                key={opt.value}
                onClick={() => setSourceFilter(opt.value)}
                className={clsx(
                  "text-xs px-2.5 py-1 rounded-md border transition-colors",
                  sourceFilter === opt.value
                    ? "bg-violet-500/15 text-violet-700 dark:text-violet-300 border-violet-500/30"
                    : "bg-zinc-100 dark:bg-zinc-800/60 text-zinc-500 border-zinc-200 dark:border-zinc-700/50 hover:text-zinc-700 dark:hover:text-zinc-300",
                )}
              >
                {opt.label}
                {count > 0 && (
                  <span className="ml-1.5 text-zinc-600">{count}</span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center gap-2 text-zinc-500 py-12 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading memories…</span>
        </div>
      )}

      {/* Error */}
      {isError && (
        <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load memories.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !isError && autoExtracted.length === 0 && unresolved.length === 0 && (
        <div className="card text-center py-16">
          <CheckCircle2 className="w-10 h-10 text-emerald-600/60 mx-auto mb-3" />
          <p className="text-zinc-400 text-sm font-medium">Nothing to review</p>
          <p className="text-zinc-600 text-xs mt-1">
            Auto-extracted memories will appear here once passive extraction runs.
          </p>
        </div>
      )}

      {/* Filtered empty */}
      {!isLoading && !isError && autoExtracted.length > 0 && filtered.length === 0 && (
        <div className="card text-center py-10">
          <p className="text-zinc-500 text-sm">No memories for this source filter.</p>
        </div>
      )}

      {/* Cards */}
      {!isLoading && filtered.length > 0 && (
        <div className="space-y-3">
          {filtered.map((event: MemoryEvent) => (
            <ReviewCard
              key={event.event_id}
              event={event}
              onApproved={() =>
                setApprovedIds((prev) => new Set([...prev, event.event_id]))
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

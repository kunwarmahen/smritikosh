"use client";

import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Network, ThumbsUp, ThumbsDown, Trash2, Tag, GitMerge, Sparkles } from "lucide-react";
import { clsx } from "clsx";
import { useDeleteEvent, useSubmitFeedback } from "@/hooks/useMemory";
import type { MemoryEvent } from "@/types";
import { importanceLevel } from "@/types";
import { SourceBadge } from "./SourceBadge";

interface Props {
  event: MemoryEvent;
  onViewGraph?: (eventId: string) => void;
}

const IMPORTANCE_BORDER: Record<string, string> = {
  high:   "border-l-emerald-500/70",
  medium: "border-l-amber-500/70",
  low:    "border-l-zinc-700",
};

const IMPORTANCE_DOT: Record<string, string> = {
  high:   "bg-emerald-500",
  medium: "bg-amber-500",
  low:    "bg-zinc-600",
};

export function MemoryCard({ event, onViewGraph }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<string | null>(null);
  const submitFeedback = useSubmitFeedback();
  const deleteEvent = useDeleteEvent();

  const level = importanceLevel(event.importance_score);
  const isLong = event.raw_text.length > 220;
  const displayText = expanded || !isLong
    ? event.raw_text
    : event.raw_text.slice(0, 220) + "…";

  const timeAgo = (() => {
    try {
      const ts = event.created_at.endsWith("Z") ? event.created_at : event.created_at + "Z";
      return formatDistanceToNow(new Date(ts), { addSuffix: true });
    } catch { return ""; }
  })();

  async function handleFeedback(type: "positive" | "negative" | "neutral") {
    if (feedbackGiven) return;
    setFeedbackGiven(type);
    await submitFeedback.mutateAsync({ event_id: event.event_id, feedback_type: type });
  }

  async function handleDelete() {
    if (!confirm("Delete this memory? This cannot be undone.")) return;
    await deleteEvent.mutateAsync(event.event_id);
  }

  return (
    <div
      className={clsx(
        "group relative bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl",
        "border-l-2 pl-4 pr-4 py-3.5",
        "hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors duration-100",
        IMPORTANCE_BORDER[level],
      )}
    >
      {/* Top row: meta + timestamp */}
      <div className="flex items-center justify-between gap-3 mb-2.5">
        <div className="flex items-center gap-2.5 min-w-0 flex-wrap">
          <span
            className={clsx("w-1.5 h-1.5 rounded-full flex-shrink-0", IMPORTANCE_DOT[level])}
            title={`Importance: ${(event.importance_score * 100).toFixed(0)}%`}
          />
          <span className="mono text-zinc-600">
            {(event.importance_score * 100).toFixed(0)}%
          </span>
          {event.consolidated && (
            <span className="badge bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border border-zinc-200 dark:border-zinc-700/60">
              <GitMerge className="w-3 h-3" />
              consolidated
            </span>
          )}
          {event.cluster_label && (
            <span className="badge bg-violet-500/10 text-violet-400 border border-violet-500/20">
              <Tag className="w-3 h-3" />
              {event.cluster_label}
            </span>
          )}
          {event.hybrid_score !== undefined && (
            <span
              className="badge bg-sky-500/10 text-sky-400 border border-sky-500/20"
              title={`Similarity: ${(event.similarity_score! * 100).toFixed(0)}%`}
            >
              <Sparkles className="w-3 h-3" />
              {(event.hybrid_score * 100).toFixed(0)}% match
            </span>
          )}
          {event.source_type && event.source_type !== "api_explicit" && (
            <SourceBadge sourceType={event.source_type} />
          )}
        </div>
        <span className="text-xs text-zinc-600 flex-shrink-0">{timeAgo}</span>
      </div>

      {/* Text body */}
      <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{displayText}</p>
      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1.5 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      )}

      {/* Summary */}
      {event.summary && event.consolidated && (
        <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-zinc-800">
          <p className="section-heading mb-1">Summary</p>
          <p className="text-xs text-zinc-500 leading-relaxed">{event.summary}</p>
        </div>
      )}

      {/* Actions — visible on hover */}
      <div
        className="flex items-center justify-between mt-3 pt-2.5 border-t border-zinc-200/80 dark:border-zinc-800/60
                   opacity-0 group-hover:opacity-100 transition-opacity duration-150"
      >
        <div className="flex items-center gap-0.5">
          {(["positive", "negative"] as const).map((type) => (
            <button
              key={type}
              onClick={() => handleFeedback(type)}
              disabled={!!feedbackGiven || submitFeedback.isPending}
              className={clsx(
                "w-6 h-6 flex items-center justify-center rounded transition-colors",
                feedbackGiven === type
                  ? type === "positive" ? "text-emerald-400" : "text-rose-400"
                  : "text-zinc-400 dark:text-zinc-700 hover:text-zinc-600 dark:hover:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800",
              )}
              title={type}
            >
              {type === "positive"
                ? <ThumbsUp className="w-3.5 h-3.5" />
                : <ThumbsDown className="w-3.5 h-3.5" />}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          {onViewGraph && (
            <button
              onClick={() => onViewGraph(event.event_id)}
              className="flex items-center gap-1.5 text-xs text-zinc-600 hover:text-violet-400
                         px-2 py-1 rounded hover:bg-violet-500/10 transition-colors"
            >
              <Network className="w-3.5 h-3.5" />
              Graph
            </button>
          )}
          <button
            onClick={handleDelete}
            disabled={deleteEvent.isPending}
            className="w-6 h-6 flex items-center justify-center rounded text-zinc-700
                       hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            title="Delete"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  GitMerge, Network, ThumbsUp, ThumbsDown, Minus,
  Trash2, ChevronDown, ChevronUp, Tag,
} from "lucide-react";
import { clsx } from "clsx";
import { useDeleteEvent, useSubmitFeedback } from "@/hooks/useMemory";
import type { MemoryEvent } from "@/types";
import { importanceLevel } from "@/types";

interface Props {
  event: MemoryEvent;
  onViewGraph?: (eventId: string) => void;
}

const IMPORTANCE_COLORS = {
  high:   "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  medium: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  low:    "bg-rose-500/20 text-rose-300 border-rose-500/30",
};

export function MemoryCard({ event, onViewGraph }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<string | null>(null);
  const submitFeedback = useSubmitFeedback();
  const deleteEvent = useDeleteEvent();

  const level = importanceLevel(event.importance_score);
  const isLong = event.raw_text.length > 200;
  const displayText = expanded || !isLong
    ? event.raw_text
    : event.raw_text.slice(0, 200) + "…";

  const createdAt = new Date(event.created_at);
  const timeAgo = formatDistanceToNow(createdAt, { addSuffix: true });

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
    <div className={clsx(
      "card group relative transition-all duration-150",
      "hover:border-slate-700",
    )}>
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Importance badge */}
          <span className={clsx("badge border", IMPORTANCE_COLORS[level])}>
            {(event.importance_score * 100).toFixed(0)}%
          </span>

          {/* Consolidated badge */}
          {event.consolidated && (
            <span className="badge bg-slate-700/50 text-slate-400 border border-slate-600/50">
              <GitMerge className="w-3 h-3" />
              Consolidated
            </span>
          )}

          {/* Cluster label */}
          {event.cluster_label && (
            <span className="badge bg-violet-500/10 text-violet-400 border border-violet-500/20">
              <Tag className="w-3 h-3" />
              {event.cluster_label}
            </span>
          )}
        </div>

        {/* Time */}
        <span className="text-xs text-slate-500 flex-shrink-0" title={createdAt.toISOString()}>
          {timeAgo}
        </span>
      </div>

      {/* Text */}
      <p className="text-sm text-slate-300 leading-relaxed mb-3">{displayText}</p>

      {/* Summary (if consolidated and has summary) */}
      {event.summary && event.consolidated && (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 mb-3">
          <p className="text-xs text-slate-500 mb-0.5 font-medium">Summary</p>
          <p className="text-xs text-slate-400 leading-relaxed">{event.summary}</p>
        </div>
      )}

      {/* Expand / collapse */}
      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 mb-3"
        >
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {expanded ? "Show less" : "Show more"}
        </button>
      )}

      {/* Footer: actions */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-800">
        {/* Feedback */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-slate-600 mr-1">Was this relevant?</span>
          {(["positive", "neutral", "negative"] as const).map((type) => (
            <button
              key={type}
              onClick={() => handleFeedback(type)}
              disabled={!!feedbackGiven || submitFeedback.isPending}
              className={clsx(
                "w-6 h-6 flex items-center justify-center rounded transition-colors",
                feedbackGiven === type
                  ? type === "positive" ? "text-emerald-400" : type === "negative" ? "text-rose-400" : "text-slate-400"
                  : "text-slate-600 hover:text-slate-300 hover:bg-slate-800",
              )}
              title={type}
            >
              {type === "positive" ? <ThumbsUp className="w-3.5 h-3.5" /> :
               type === "negative" ? <ThumbsDown className="w-3.5 h-3.5" /> :
               <Minus className="w-3.5 h-3.5" />}
            </button>
          ))}
        </div>

        {/* Right actions */}
        <div className="flex items-center gap-1">
          {onViewGraph && (
            <button
              onClick={() => onViewGraph(event.event_id)}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-violet-400
                         px-2 py-1 rounded hover:bg-violet-500/10 transition-colors"
            >
              <Network className="w-3.5 h-3.5" />
              Graph
            </button>
          )}
          <button
            onClick={handleDelete}
            disabled={deleteEvent.isPending}
            className="w-7 h-7 flex items-center justify-center rounded text-slate-600
                       hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            title="Delete memory"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

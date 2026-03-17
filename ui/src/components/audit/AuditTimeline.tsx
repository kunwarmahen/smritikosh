"use client";

import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Loader2, Inbox, ChevronDown, ChevronUp } from "lucide-react";
import { clsx } from "clsx";
import { useAuditTimeline } from "@/hooks/useAudit";
import { EVENT_TYPE_LABELS, EVENT_TYPE_COLORS } from "@/types";
import type { AuditEvent } from "@/types";

const EVENT_TYPE_OPTIONS = [
  { value: "", label: "All events" },
  { value: "memory.encoded",         label: "Encoded" },
  { value: "memory.facts_extracted", label: "Facts extracted" },
  { value: "memory.consolidated",    label: "Consolidated" },
  { value: "memory.reconsolidated",  label: "Reconsolidated" },
  { value: "memory.pruned",          label: "Pruned" },
  { value: "memory.clustered",       label: "Clustered" },
  { value: "belief.mined",           label: "Belief mined" },
  { value: "feedback.submitted",     label: "Feedback" },
  { value: "context.built",          label: "Context built" },
  { value: "search.performed",       label: "Search" },
];

function PayloadRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="flex gap-2 text-xs">
      <span className="text-slate-600 min-w-[120px] flex-shrink-0">{label}</span>
      <span className="text-slate-400 break-all">
        {typeof value === "object" ? JSON.stringify(value) : String(value ?? "—")}
      </span>
    </div>
  );
}

function AuditRow({ event }: { event: AuditEvent }) {
  const [open, setOpen] = useState(false);
  const timeAgo = (() => {
    try {
      if (!event.timestamp) return "unknown time";
      const ts = event.timestamp.endsWith("Z") ? event.timestamp : event.timestamp + "Z";
      return formatDistanceToNow(new Date(ts), { addSuffix: true });
    } catch {
      return "unknown time";
    }
  })();
  const payloadEntries = Object.entries(event.payload ?? {});

  return (
    <div className="card py-3 px-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap min-w-0">
          <span
            className={clsx(
              "badge border text-xs",
              EVENT_TYPE_COLORS[event.event_type] ?? "text-slate-400",
              "bg-slate-800/50 border-slate-700/50",
            )}
          >
            {EVENT_TYPE_LABELS[event.event_type] ?? event.event_type}
          </span>
          {event.event_id && (
            <span className="text-xs text-slate-600 font-mono truncate max-w-[180px]">
              {event.event_id}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs text-slate-500">{timeAgo}</span>
          {payloadEntries.length > 0 && (
            <button
              onClick={() => setOpen((v) => !v)}
              className="text-slate-600 hover:text-slate-300 transition-colors"
            >
              {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          )}
        </div>
      </div>

      {open && payloadEntries.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-800 space-y-1">
          {payloadEntries.map(([k, v]) => (
            <PayloadRow key={k} label={k} value={v} />
          ))}
        </div>
      )}
    </div>
  );
}

export function AuditTimeline() {
  const [eventType, setEventType] = useState("");
  const [limit, setLimit] = useState(30);

  const { data, isLoading, isError, refetch, isFetching } = useAuditTimeline({
    event_type: eventType || undefined,
    limit,
  });

  const events = data ?? [];

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex items-center gap-3">
        <select
          value={eventType}
          onChange={(e) => { setEventType(e.target.value); setLimit(30); }}
          className="input text-sm py-1.5 w-48"
        >
          {EVENT_TYPE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="btn-secondary text-xs px-3 py-1.5"
        >
          {isFetching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : "Refresh"}
        </button>
        {events.length > 0 && (
          <span className="text-xs text-slate-500 ml-auto">
            {events.length} event{events.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-slate-500 py-8 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading audit log…</span>
        </div>
      )}

      {isError && (
        <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load audit events.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {!isLoading && !isError && events.length === 0 && (
        <div className="card text-center py-12">
          <Inbox className="w-10 h-10 text-slate-700 mx-auto mb-3" />
          <p className="text-slate-400 text-sm font-medium">No audit events found.</p>
          <p className="text-slate-600 text-xs mt-1">
            {eventType ? "Try a different filter." : "Audit events appear as memory operations run."}
          </p>
        </div>
      )}

      <div className="space-y-2">
        {events.map((event) => (
          <AuditRow key={event.event_id ? `${event.event_id}-${event.event_type}-${event.timestamp}` : `${event.event_type}-${event.timestamp}`} event={event} />
        ))}
      </div>

      {events.length > 0 && events.length >= limit && (
        <div className="text-center pt-2">
          <button
            onClick={() => setLimit((l) => l + 30)}
            disabled={isFetching}
            className="btn-secondary"
          >
            {isFetching ? <Loader2 className="w-4 h-4 animate-spin" /> : "Load more"}
          </button>
        </div>
      )}
    </div>
  );
}

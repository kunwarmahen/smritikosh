"use client";

import { useAuditStats } from "@/hooks/useAudit";
import { Loader2 } from "lucide-react";
import { EVENT_TYPE_LABELS, EVENT_TYPE_COLORS } from "@/types";

export function AuditStatsBar() {
  const { data, isLoading } = useAuditStats();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-slate-500 py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">Loading stats…</span>
      </div>
    );
  }

  if (!data || Object.keys(data).length === 0) return null;

  const total = Object.values(data).reduce((a, b) => a + b, 0);

  return (
    <div className="card mb-6">
      <p className="text-xs font-medium text-slate-500 mb-3 uppercase tracking-wide">
        Event breakdown · {total.toLocaleString()} total
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {Object.entries(data)
          .sort(([, a], [, b]) => b - a)
          .map(([type, count]) => (
            <div
              key={type}
              className="bg-slate-800/50 rounded-lg px-3 py-2 border border-slate-700/50"
            >
              <p className={`text-xs font-medium truncate ${EVENT_TYPE_COLORS[type] ?? "text-slate-400"}`}>
                {EVENT_TYPE_LABELS[type] ?? type}
              </p>
              <p className="text-lg font-semibold text-slate-200 mt-0.5">
                {count.toLocaleString()}
              </p>
            </div>
          ))}
      </div>
    </div>
  );
}

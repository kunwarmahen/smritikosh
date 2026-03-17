"use client";

import { clsx } from "clsx";
import { CheckCircle2, XCircle, AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import { useHealth } from "@/hooks/useAdmin";

function StatusDot({ status }: { status: "ok" | "error" | "degraded" }) {
  return (
    <span className={clsx(
      "inline-flex items-center gap-1.5 text-sm font-medium",
      status === "ok"      && "text-emerald-400",
      status === "degraded" && "text-amber-400",
      status === "error"   && "text-rose-400",
    )}>
      {status === "ok"       && <CheckCircle2 className="w-4 h-4" />}
      {status === "degraded" && <AlertTriangle className="w-4 h-4" />}
      {status === "error"    && <XCircle className="w-4 h-4" />}
      {status}
    </span>
  );
}

export function HealthPanel() {
  const { data, isLoading, refetch, isFetching } = useHealth();

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-medium text-slate-400">System Health</h2>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="text-slate-500 hover:text-slate-300 transition-colors"
          title="Refresh"
        >
          <RefreshCw className={clsx("w-4 h-4", isFetching && "animate-spin")} />
        </button>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-2 text-slate-500 py-4">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-sm">Checking health…</span>
        </div>
      ) : data ? (
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2 border-b border-slate-800">
            <span className="text-sm text-slate-400">Overall</span>
            <StatusDot status={data.status} />
          </div>
          <div className="flex items-center justify-between py-2 border-b border-slate-800">
            <span className="text-sm text-slate-400">PostgreSQL</span>
            <StatusDot status={data.postgres} />
          </div>
          <div className="flex items-center justify-between py-2 border-b border-slate-800">
            <span className="text-sm text-slate-400">Neo4j</span>
            <StatusDot status={data.neo4j} />
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-sm text-slate-400">Version</span>
            <span className="text-sm text-slate-300 font-mono">{data.version}</span>
          </div>
        </div>
      ) : (
        <p className="text-sm text-slate-500 py-4">Could not reach the API.</p>
      )}
    </div>
  );
}

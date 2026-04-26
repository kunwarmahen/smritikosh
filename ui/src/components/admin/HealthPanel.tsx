"use client";

import { clsx } from "clsx";
import { CheckCircle2, XCircle, AlertTriangle, MinusCircle, Loader2, RefreshCw } from "lucide-react";
import { useHealth } from "@/hooks/useAdmin";

type Status = "ok" | "error" | "degraded" | "not_configured";

function StatusChip({ status }: { status: Status }) {
  const label = status === "not_configured" ? "not configured" : status;
  return (
    <span className={clsx(
      "inline-flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 rounded-md border",
      status === "ok"             && "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
      status === "degraded"       && "bg-amber-500/10   text-amber-400   border-amber-500/20",
      status === "error"          && "bg-rose-500/10    text-rose-400    border-rose-500/20",
      status === "not_configured" && "bg-zinc-500/10    text-zinc-500    border-zinc-300 dark:border-zinc-700/40",
    )}>
      {status === "ok"             && <CheckCircle2  className="w-3 h-3" />}
      {status === "degraded"       && <AlertTriangle className="w-3 h-3" />}
      {status === "error"          && <XCircle       className="w-3 h-3" />}
      {status === "not_configured" && <MinusCircle   className="w-3 h-3" />}
      {label}
    </span>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-zinc-200 dark:border-zinc-800/60 last:border-0">
      <span className="text-sm text-zinc-500">{label}</span>
      {children}
    </div>
  );
}

export function HealthPanel() {
  const { data, isLoading, refetch, isFetching } = useHealth();

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <p className="section-heading">System status</p>
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
        <div>
          <Row label="Overall">    <StatusChip status={data.status}      /> </Row>
          <Row label="PostgreSQL"> <StatusChip status={data.postgres}    /> </Row>
          <Row label="Neo4j">      <StatusChip status={data.neo4j}       /> </Row>
          <Row label="MongoDB">    <StatusChip status={data.mongodb}     /> </Row>
          <Row label="LLM">
            <div className="flex items-center gap-2">
              <span className="mono text-zinc-400 text-xs">{data.llm_model}</span>
              <StatusChip status={data.llm_status} />
            </div>
          </Row>
          <Row label="Version">
            <span className="mono text-zinc-400">{data.version}</span>
          </Row>
        </div>
      ) : (
        <p className="text-sm text-zinc-600 py-4">Could not reach the API.</p>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import { Loader2, Inbox, Zap, ToggleLeft, ToggleRight, Trash2, Plus } from "lucide-react";
import { clsx } from "clsx";
import { useProcedures, useUpdateProcedure, useDeleteProcedure } from "@/hooks/useProcedures";
import type { Procedure } from "@/types";

interface Props {
  onNew: () => void;
}

function ProcedureRow({ proc }: { proc: Procedure }) {
  const update = useUpdateProcedure();
  const del = useDeleteProcedure();

  async function toggle() {
    await update.mutateAsync({ procedureId: proc.procedure_id, body: { is_active: !proc.is_active } });
  }

  async function handleDelete() {
    if (!confirm("Delete this procedure? This cannot be undone.")) return;
    await del.mutateAsync(proc.procedure_id);
  }

  return (
    <div className={clsx("card py-3 px-4 group", !proc.is_active && "opacity-50")}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="badge bg-violet-500/10 text-violet-400 border border-violet-500/20 text-xs">
              {proc.category || "general"}
            </span>
            <span className="badge bg-zinc-800 text-zinc-500 border border-zinc-700/50 text-xs">
              priority {proc.priority}
            </span>
            <span className="text-xs text-zinc-600">{proc.hit_count} hit{proc.hit_count !== 1 ? "s" : ""}</span>
          </div>
          <p className="text-xs text-zinc-500 mb-1">
            <span className="font-medium text-zinc-400">Trigger:</span> {proc.trigger}
          </p>
          <p className="text-sm text-zinc-300 leading-relaxed">{proc.instruction}</p>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={toggle}
            disabled={update.isPending}
            className="text-zinc-500 hover:text-violet-400 transition-colors"
            title={proc.is_active ? "Deactivate" : "Activate"}
          >
            {proc.is_active
              ? <ToggleRight className="w-5 h-5 text-violet-400" />
              : <ToggleLeft className="w-5 h-5" />}
          </button>
          <button
            onClick={handleDelete}
            disabled={del.isPending}
            className="w-7 h-7 flex items-center justify-center rounded text-zinc-600
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

export function ProcedureTable({ onNew }: Props) {
  const { data, isLoading, isError, refetch } = useProcedures();
  const procedures = data ?? [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-zinc-500">
          {procedures.length} procedure{procedures.length !== 1 ? "s" : ""}
        </p>
        <button onClick={onNew} className="btn-primary flex items-center gap-1.5 text-xs px-3 py-1.5">
          <Plus className="w-3.5 h-3.5" />
          New procedure
        </button>
      </div>

      {isLoading && (
        <div className="flex items-center gap-2 text-zinc-500 py-8 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading procedures…</span>
        </div>
      )}

      {isError && (
        <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load procedures.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {!isLoading && !isError && procedures.length === 0 && (
        <div className="card text-center py-12">
          <Zap className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-zinc-400 text-sm font-medium">No procedures yet.</p>
          <p className="text-zinc-600 text-xs mt-1">
            Procedures are triggered instructions that run during context building.
          </p>
        </div>
      )}

      {procedures.map((proc) => (
        <ProcedureRow key={proc.procedure_id} proc={proc} />
      ))}
    </div>
  );
}

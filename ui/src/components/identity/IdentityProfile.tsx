"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight, Loader2, User, Brain, Sparkles, Trash2 } from "lucide-react";
import { clsx } from "clsx";
import { useIdentity } from "@/hooks/useIdentity";
import { useBeliefEvidence, useRetractBelief } from "@/hooks/useBeliefs";
import type { IdentityDimension, UserBelief } from "@/types";

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="h-0.5 w-full bg-zinc-200 dark:bg-zinc-800 rounded-full overflow-hidden mt-2">
      <div
        className={clsx(
          "h-full rounded-full transition-all",
          value >= 0.7 ? "bg-emerald-500" : value >= 0.4 ? "bg-amber-500" : "bg-zinc-600",
        )}
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}

function DimensionCard({ dim }: { dim: IdentityDimension }) {
  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-3.5">
      <div className="flex items-start justify-between gap-2 mb-1">
        <p className="section-heading">{dim.category}</p>
        <span className="mono text-zinc-600" title={`${dim.fact_count} facts`}>{dim.fact_count}f</span>
      </div>
      <p className="text-sm text-zinc-800 dark:text-zinc-200 font-medium leading-snug">{dim.dominant_value}</p>
      <ConfidenceBar value={dim.confidence} />
      <p className="text-[10px] text-zinc-700 mt-1.5">{(dim.confidence * 100).toFixed(0)}% confidence</p>
    </div>
  );
}

function BeliefEvidenceList({ beliefId }: { beliefId: string }) {
  const { data, isLoading, isError } = useBeliefEvidence(beliefId);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-zinc-600 py-2 pl-1">
        <Loader2 className="w-3 h-3 animate-spin" />
        <span className="text-xs">Loading evidence…</span>
      </div>
    );
  }
  if (isError || !data) {
    return <p className="text-xs text-rose-400 py-2 pl-1">Failed to load evidence.</p>;
  }
  if (data.evidence_events.length === 0) {
    return (
      <p className="text-xs text-zinc-600 py-2 pl-1">
        No evidence events recorded (belief predates evidence tracking, or its
        source events were pruned).
      </p>
    );
  }

  return (
    <div className="mt-2 mb-1 pl-1 space-y-1.5 border-l-2 border-zinc-200 dark:border-zinc-800">
      <p className="text-[10px] uppercase tracking-wide text-zinc-500 pl-2">Based on these events</p>
      {data.evidence_events.map((e) => (
        <div key={e.event_id} className="pl-2">
          <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">{e.text}</p>
          {e.created_at && (
            <p className="text-[10px] text-zinc-600">{new Date(e.created_at).toLocaleDateString()}</p>
          )}
        </div>
      ))}
      {data.missing_event_ids.length > 0 && (
        <p className="text-[10px] text-zinc-600 pl-2">
          +{data.missing_event_ids.length} source event(s) since pruned.
        </p>
      )}
    </div>
  );
}

function BeliefRow({ belief }: { belief: UserBelief }) {
  const [expanded, setExpanded] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const retract = useRetractBelief();

  const confidenceColor =
    belief.confidence >= 0.7 ? "text-emerald-400" :
    belief.confidence >= 0.4 ? "text-amber-400" :
    "text-zinc-600";

  return (
    <div className="py-3 border-b border-zinc-200 dark:border-zinc-800/60 last:border-0">
      <div className="flex items-start gap-4">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex-shrink-0 mt-0.5 text-zinc-500 hover:text-zinc-300 transition-colors"
          title="Show evidence"
        >
          {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        </button>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{belief.statement}</p>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="badge bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border border-zinc-200 dark:border-zinc-700/50">
              {belief.category}
            </span>
            <span className="text-[10px] text-zinc-600">{belief.evidence_count} evidence</span>
          </div>
        </div>
        <span className={clsx("mono font-semibold flex-shrink-0 mt-0.5", confidenceColor)}>
          {(belief.confidence * 100).toFixed(0)}%
        </span>
        {confirming ? (
          <span className="flex items-center gap-1.5 flex-shrink-0 mt-0.5">
            <button
              onClick={() => retract.mutate(belief.belief_id, { onSettled: () => setConfirming(false) })}
              disabled={retract.isPending}
              className="text-[11px] px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20 transition-colors disabled:opacity-50"
            >
              {retract.isPending ? "Retracting…" : "Confirm"}
            </button>
            <button
              onClick={() => setConfirming(false)}
              className="text-[11px] px-2 py-0.5 rounded text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              Cancel
            </button>
          </span>
        ) : (
          <button
            onClick={() => setConfirming(true)}
            className="flex-shrink-0 mt-0.5 text-zinc-600 hover:text-rose-400 transition-colors"
            title="Retract this belief — it will never be re-inferred"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
      {expanded && <BeliefEvidenceList beliefId={belief.belief_id} />}
    </div>
  );
}

export function IdentityProfilePanel() {
  const { data, isLoading, isError } = useIdentity();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-zinc-600 py-16 justify-center">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">Building identity profile…</span>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="card border-rose-500/20 bg-rose-500/5 text-center py-8">
        <p className="text-rose-400 text-sm">Failed to load identity profile.</p>
      </div>
    );
  }

  if (!data || data.is_empty) {
    return (
      <div className="card text-center py-16">
        <User className="w-10 h-10 text-zinc-800 mx-auto mb-3" />
        <p className="text-zinc-500 text-sm font-medium">No identity profile yet.</p>
        <p className="text-zinc-700 text-xs mt-1">
          Add memories — the system will synthesise a profile automatically.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Summary */}
      <div className="card bg-violet-500/5 border-violet-500/20">
        <div className="flex items-center gap-2 mb-2.5">
          <Sparkles className="w-3.5 h-3.5 text-violet-400" />
          <span className="section-heading text-violet-500">AI Summary</span>
        </div>
        <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{data.summary}</p>
        <p className="mono text-zinc-700 mt-3">
          {data.total_facts} facts · {new Date(data.computed_at).toLocaleDateString()}
        </p>
      </div>

      {/* Dimensions */}
      {data.dimensions.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-3.5 h-3.5 text-zinc-600" />
            <h2 className="section-heading">Identity Dimensions</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {data.dimensions.map((dim, i) => (
              <DimensionCard key={i} dim={dim} />
            ))}
          </div>
        </div>
      )}

      {/* Beliefs */}
      {data.beliefs.length > 0 && (
        <div>
          <h2 className="section-heading mb-4">
            Inferred Beliefs · {data.beliefs.length}
          </h2>
          <div className="card py-0 divide-y divide-zinc-800/60">
            {data.beliefs.map((belief, i) => (
              <BeliefRow key={belief.belief_id || i} belief={belief} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

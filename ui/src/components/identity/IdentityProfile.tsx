"use client";

import { Loader2, User, Brain, Sparkles } from "lucide-react";
import { clsx } from "clsx";
import { useIdentity } from "@/hooks/useIdentity";
import type { IdentityDimension, UserBelief } from "@/types";

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
      <div
        className="h-full bg-violet-500 rounded-full transition-all"
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}

function DimensionCard({ dim }: { dim: IdentityDimension }) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
          {dim.category}
        </span>
        <span className="text-xs text-slate-500">{dim.fact_count} facts</span>
      </div>
      <p className="text-sm text-slate-200 font-medium mb-2">{dim.dominant_value}</p>
      <ConfidenceBar value={dim.confidence} />
      <p className="text-xs text-slate-600 mt-1">{(dim.confidence * 100).toFixed(0)}% confidence</p>
    </div>
  );
}

function BeliefRow({ belief }: { belief: UserBelief }) {
  const color =
    belief.confidence >= 0.7 ? "text-emerald-400" :
    belief.confidence >= 0.4 ? "text-amber-400" : "text-slate-400";

  return (
    <div className="flex items-start gap-3 py-2.5 border-b border-slate-800 last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-slate-300 leading-relaxed">{belief.statement}</p>
        <div className="flex items-center gap-2 mt-1">
          <span className="badge bg-slate-800 text-slate-500 border border-slate-700/50 text-xs">
            {belief.category}
          </span>
          <span className="text-xs text-slate-600">{belief.evidence_count} evidence</span>
        </div>
      </div>
      <span className={clsx("text-sm font-semibold flex-shrink-0", color)}>
        {(belief.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function IdentityProfilePanel() {
  const { data, isLoading, isError } = useIdentity();

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-slate-500 py-12 justify-center">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span className="text-sm">Building your identity profile…</span>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
        <p className="text-rose-400 text-sm">Failed to load identity profile.</p>
      </div>
    );
  }

  if (!data || data.is_empty) {
    return (
      <div className="card text-center py-16">
        <User className="w-12 h-12 text-slate-700 mx-auto mb-3" />
        <p className="text-slate-400 text-sm font-medium">No identity profile yet.</p>
        <p className="text-slate-600 text-xs mt-1">
          Add more memories — the system will synthesise your identity model automatically.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary card */}
      <div className="card bg-violet-500/5 border-violet-500/20">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles className="w-4 h-4 text-violet-400" />
          <span className="text-xs font-medium text-violet-400 uppercase tracking-wide">Summary</span>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">{data.summary}</p>
        <p className="text-xs text-slate-600 mt-2">
          Based on {data.total_facts} facts · computed {new Date(data.computed_at).toLocaleDateString()}
        </p>
      </div>

      {/* Dimensions */}
      {data.dimensions.length > 0 && (
        <div>
          <h2 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
            <Brain className="w-4 h-4" /> Identity Dimensions
          </h2>
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
          <h2 className="text-sm font-medium text-slate-400 mb-3">
            Inferred Beliefs ({data.beliefs.length})
          </h2>
          <div className="card divide-y divide-slate-800 py-0">
            {data.beliefs.map((belief, i) => (
              <BeliefRow key={i} belief={belief} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import React, { useCallback, useMemo, useRef, useState } from "react";
import { Loader2, Network, BookOpen, ExternalLink, X } from "lucide-react";
import Link from "next/link";
import { useFactGraph } from "@/hooks/useFactGraph";
import { useQueries } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import { ForceGraph2D, GRAPH_BG, roundRect } from "@/lib/graph-shared";
import type { MemoryEvent, FactGraphNode, FactGraphEdge } from "@/types";

// ── Colour palettes ───────────────────────────────────────────────────────────
const CATEGORY_STYLES: Record<string, { bg: string; border: string; text: string }> = {
  identity:     { bg: "#431407", border: "#f97316", text: "#fed7aa" },
  location:     { bg: "#042f2e", border: "#14b8a6", text: "#99f6e4" },
  role:         { bg: "#14532d", border: "#22c55e", text: "#86efac" },
  skill:        { bg: "#1e3a4f", border: "#06b6d4", text: "#67e8f9" },
  education:    { bg: "#1e1b4b", border: "#6366f1", text: "#c7d2fe" },
  project:      { bg: "#78350f", border: "#f59e0b", text: "#fcd34d" },
  goal:         { bg: "#4a1942", border: "#a855f7", text: "#d8b4fe" },
  interest:     { bg: "#1e3a5f", border: "#3b82f6", text: "#93c5fd" },
  hobby:        { bg: "#1a2e05", border: "#84cc16", text: "#d9f99d" },
  habit:        { bg: "#422006", border: "#eab308", text: "#fef08a" },
  preference:   { bg: "#3b1f6e", border: "#7c3aed", text: "#c4b5fd" },
  personality:  { bg: "#4c0519", border: "#f43f5e", text: "#fecdd3" },
  relationship: { bg: "#1f2937", border: "#6b7280", text: "#d1d5db" },
  pet:          { bg: "#500724", border: "#ec4899", text: "#fbcfe8" },
  health:       { bg: "#450a0a", border: "#ef4444", text: "#fca5a5" },
  diet:         { bg: "#022c22", border: "#10b981", text: "#6ee7b7" },
  belief:       { bg: "#0f172a", border: "#64748b", text: "#cbd5e1" },
  value:        { bg: "#451a03", border: "#d97706", text: "#fde68a" },
  religion:     { bg: "#1c1917", border: "#a8a29e", text: "#d6d3d1" },
  finance:      { bg: "#052e16", border: "#16a34a", text: "#bbf7d0" },
  lifestyle:    { bg: "#082f49", border: "#0ea5e9", text: "#bae6fd" },
  event:        { bg: "#4a044e", border: "#d946ef", text: "#f0abfc" },
  tool:         { bg: "#18181b", border: "#71717a", text: "#d4d4d8" },
};
const DEFAULT_STYLE = { bg: "#1e293b", border: "#475569", text: "#94a3b8" };

const EDGE_COLORS: Record<string, string> = {
  HAS_IDENTITY:         "#f97316",
  LIVES_IN:             "#14b8a6",
  HAS_ROLE:             "#22c55e",
  HAS_SKILL:            "#06b6d4",
  STUDIED_AT:           "#6366f1",
  WORKS_ON:             "#f59e0b",
  HAS_GOAL:             "#a855f7",
  HAS_INTEREST:         "#3b82f6",
  ENJOYS:               "#84cc16",
  HAS_HABIT:            "#eab308",
  HAS_PREFERENCE:       "#7c3aed",
  HAS_TRAIT:            "#f43f5e",
  KNOWS:                "#6b7280",
  HAS_PET:              "#ec4899",
  HAS_HEALTH_CONDITION: "#ef4444",
  FOLLOWS_DIET:         "#10b981",
  BELIEVES:             "#64748b",
  VALUES:               "#d97706",
  PRACTICES:            "#a8a29e",
  HAS_FINANCE:          "#16a34a",
  HAS_LIFESTYLE:        "#0ea5e9",
  EXPERIENCED:          "#d946ef",
  USES:                 "#71717a",
  RELATED_TO:           "#94a3b8",
};

// ── Types ─────────────────────────────────────────────────────────────────────
interface GraphNode extends FactGraphNode {
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface GraphLink extends FactGraphEdge {}

interface SelectedFact {
  label: string;
  category: string;
  confidence: number | null;
  frequency_count: number | null;
  sourceEventIds: string[];
}


// ── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  const items = Object.entries(CATEGORY_STYLES).map(([cat, s]) => ({ cat, color: s.border }));
  return (
    <div className="absolute bottom-3 left-3 z-10 bg-zinc-900/90 border border-zinc-700/50
                    rounded-xl p-3 backdrop-blur-sm max-h-[460px] overflow-y-auto">
      <p className="text-xs font-medium text-zinc-500 mb-2 uppercase tracking-wide">Categories</p>
      <div className="space-y-1">
        {items.map(({ cat, color }) => (
          <div key={cat} className="flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: color }} />
            <span className="text-xs text-zinc-400 capitalize">{cat}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Source memories panel ─────────────────────────────────────────────────────
function normalizeText(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}

function SourceMemoriesPanel({ fact, onClose }: { fact: SelectedFact; onClose: () => void }) {
  const { data: session } = useSession();
  const token = session?.accessToken;
  const style = CATEGORY_STYLES[fact.category] ?? DEFAULT_STYLE;

  const results = useQueries({
    queries: fact.sourceEventIds.map((id) => ({
      queryKey: ["event", id],
      queryFn: () => createApiClient(token).getEvent(id) as Promise<MemoryEvent>,
      enabled: !!token,
      staleTime: 120_000,
    })),
  });

  const isLoading = results.some((r) => r.isLoading);

  const { unique, skipped } = useMemo(() => {
    const seen = new Set<string>();
    const unique: Array<{ eventId: string; event: MemoryEvent }> = [];
    let skipped = 0;
    results.forEach((r, i) => {
      if (!r.data) return;
      const text = normalizeText(r.data.raw_text);
      if (seen.has(text)) { skipped++; return; }
      seen.add(text);
      unique.push({ eventId: fact.sourceEventIds[i], event: r.data });
    });
    return { unique, skipped };
  }, [results, fact.sourceEventIds]);

  return (
    <div className="rounded-xl border border-zinc-700/50 bg-zinc-900/95 mt-0">
      {/* Header row */}
      <div className="flex items-start justify-between px-4 py-3 border-b border-zinc-700/50">
        <div className="flex items-center gap-3 min-w-0">
          <span className="inline-block w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: style.border }} />
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-zinc-500 capitalize">{fact.category}</span>
              {fact.confidence != null && (
                <span className="text-xs text-zinc-600">{(fact.confidence * 100).toFixed(0)}% confidence</span>
              )}
              {fact.frequency_count != null && (
                <span className="text-xs text-zinc-600">seen {fact.frequency_count}×</span>
              )}
            </div>
            <p className="text-sm font-semibold text-zinc-100 mt-0.5">{fact.label}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="ml-3 p-1 rounded-md text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700/50 transition-colors flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Memories */}
      <div className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <BookOpen className="w-3.5 h-3.5 text-zinc-500" />
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wide">Contributing memories</p>
        </div>
        {isLoading ? (
          <div className="flex items-center gap-2 text-zinc-500 py-4 justify-center">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-xs">Loading…</span>
          </div>
        ) : unique.length === 0 ? (
          <p className="text-xs text-zinc-600 text-center py-4">No source memories tracked for this fact.</p>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {unique.map(({ event }) => (
                <div key={event.event_id} className="rounded-lg border border-zinc-700/50 bg-zinc-800/40 p-3 group">
                  <p className="text-xs text-zinc-300 line-clamp-3 leading-relaxed">{event.raw_text}</p>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs text-zinc-600">{new Date(event.created_at).toLocaleDateString()}</span>
                    <Link
                      href={`/dashboard/memories/${event.event_id}`}
                      className="flex items-center gap-1 text-xs text-violet-400 hover:text-violet-300 transition-colors opacity-0 group-hover:opacity-100"
                    >
                      View <ExternalLink className="w-3 h-3" />
                    </Link>
                  </div>
                </div>
              ))}
            </div>
            {skipped > 0 && (
              <p className="text-xs text-zinc-600 text-center mt-3">
                +{skipped} duplicate{skipped !== 1 ? "s" : ""} hidden
              </p>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export function IdentityFactGraph() {
  const { data, isLoading, isError } = useFactGraph();
  const [selectedFact, setSelectedFact] = useState<SelectedFact | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    return {
      nodes: data.nodes.map((n) => ({ ...n })) as GraphNode[],
      links: data.edges.map((e) => ({ ...e })) as GraphLink[],
    };
  }, [data]);

  const drawNode = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = raw as GraphNode;
      const nx = node.x ?? 0;
      const ny = node.y ?? 0;
      const s = 1 / globalScale;

      if (node.node_type === "user") {
        const r = 24 * s;
        ctx.beginPath();
        ctx.arc(nx, ny, r, 0, 2 * Math.PI);
        ctx.fillStyle = "#1e1533";
        ctx.shadowColor = "#7c3aed";
        ctx.shadowBlur = 14 * s;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = "#7c3aed";
        ctx.lineWidth = 2 * s;
        ctx.stroke();
        ctx.fillStyle = "#e9d5ff";
        ctx.font = `bold ${13 * s}px system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("You", nx, ny);
        return;
      }

      const style = CATEGORY_STYLES[node.category ?? ""] ?? DEFAULT_STYLE;
      const hasSource = (node.source_event_ids?.length ?? 0) > 0;
      const W = 140 * s;
      const H = 40 * s;
      const rx = 7 * s;

      if (hasSource) {
        ctx.shadowColor = style.border;
        ctx.shadowBlur = 10 * s;
      }
      roundRect(ctx, nx - W / 2, ny - H / 2, W, H, rx);
      ctx.fillStyle = style.bg;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = style.border;
      ctx.lineWidth = 1.5 * s;
      ctx.stroke();

      const keyLabel = (node.key ?? "").replace(/_/g, " ").toUpperCase();
      ctx.fillStyle = style.text + "99";
      ctx.font = `${8 * s}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(keyLabel, nx, ny - 11 * s);

      const label = (node.label?.length ?? 0) > 22 ? node.label!.slice(0, 20) + "…" : (node.label ?? "");
      ctx.fillStyle = style.text;
      ctx.font = `${11 * s}px system-ui, sans-serif`;
      ctx.fillText(label, nx, ny + 8 * s);
    },
    [],
  );

  const nodePointerArea = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = raw as GraphNode;
      const s = 1 / globalScale;
      const nx = node.x ?? 0;
      const ny = node.y ?? 0;
      if (node.node_type === "user") {
        ctx.beginPath();
        ctx.arc(nx, ny, 24 * s, 0, 2 * Math.PI);
      } else {
        roundRect(ctx, nx - 70 * s, ny - 20 * s, 140 * s, 40 * s, 7 * s);
      }
      ctx.fillStyle = color;
      ctx.fill();
    },
    [],
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeClick = useCallback((raw: any) => {
    const node = raw as GraphNode;
    if (node.node_type === "user") return;
    setSelectedFact({
      label: node.label,
      category: node.category ?? "other",
      confidence: node.confidence ?? null,
      frequency_count: node.frequency_count ?? null,
      sourceEventIds: node.source_event_ids ?? [],
    });
  }, []);

  const getLinkColor = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => (EDGE_COLORS[(raw as GraphLink).relation] ?? "#475569") + "bb",
    [],
  );

  const getLinkWidth = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => ((raw as GraphLink).relation === "RELATED_TO" ? 0.5 : 1),
    [],
  );

  const getParticleCount = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => ((raw as GraphLink).relation === "RELATED_TO" ? 2 : 0),
    [],
  );

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-zinc-500 py-12 justify-center">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span className="text-sm">Loading fact graph…</span>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
        <p className="text-rose-400 text-sm">Failed to load fact graph.</p>
      </div>
    );
  }

  if (!data || data.nodes.length <= 1) {
    return (
      <div className="card text-center py-12">
        <Network className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
        <p className="text-zinc-400 text-sm font-medium">No fact graph yet.</p>
        <p className="text-zinc-600 text-xs mt-1">
          Facts are extracted as memories are ingested and consolidated.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div
        className="relative rounded-xl overflow-hidden border border-zinc-700/50 bg-zinc-950"
        style={{ height: 560 }}
      >
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={nodePointerArea}
          onNodeClick={handleNodeClick}
          onBackgroundClick={() => setSelectedFact(null)}
          linkColor={getLinkColor}
          linkWidth={getLinkWidth}
          linkDirectionalParticles={getParticleCount}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={getLinkColor}
          backgroundColor={GRAPH_BG}
          warmupTicks={120}
          cooldownTicks={0}
          onEngineStop={() => graphRef.current?.zoomToFit(400, 60)}
        />
        <Legend />
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
          {selectedFact === null && (
            <span className="text-xs text-zinc-500 bg-zinc-900/80 border border-zinc-700/50
                             rounded-lg px-2 py-1 backdrop-blur-sm">
              Click a fact to see source memories
            </span>
          )}
          <span className="text-xs text-zinc-500 bg-zinc-900/80 border border-zinc-700/50
                           rounded-lg px-2 py-1 backdrop-blur-sm">
            {data.nodes.length - 1} fact{data.nodes.length !== 2 ? "s" : ""} · {data.edges.length} link{data.edges.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      {selectedFact && (
        <SourceMemoriesPanel fact={selectedFact} onClose={() => setSelectedFact(null)} />
      )}
    </div>
  );
}

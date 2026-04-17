"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  type Node,
  type Edge,
  type Connection,
  type NodeMouseHandler,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Loader2, Network, X, ExternalLink, BookOpen } from "lucide-react";
import Link from "next/link";
import { useFactGraph } from "@/hooks/useFactGraph";
import { useQueries } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import type { FactGraph, FactGraphNode, FactGraphEdge, MemoryEvent } from "@/types";

// ── Category colour palette ───────────────────────────────────────────────────
const CATEGORY_STYLES: Record<string, { bg: string; border: string; text: string }> = {
  preference:   { bg: "#3b1f6e", border: "#7c3aed", text: "#c4b5fd" },
  interest:     { bg: "#1e3a5f", border: "#3b82f6", text: "#93c5fd" },
  role:         { bg: "#14532d", border: "#22c55e", text: "#86efac" },
  project:      { bg: "#78350f", border: "#f59e0b", text: "#fcd34d" },
  skill:        { bg: "#1e3a4f", border: "#06b6d4", text: "#67e8f9" },
  goal:         { bg: "#4a1942", border: "#a855f7", text: "#d8b4fe" },
  relationship: { bg: "#1f2937", border: "#6b7280", text: "#d1d5db" },
};
const USER_STYLE = { bg: "#1e1533", border: "#7c3aed", text: "#e9d5ff" };
const DEFAULT_STYLE = { bg: "#1e293b", border: "#475569", text: "#94a3b8" };

// ── Edge colour by relation ───────────────────────────────────────────────────
const EDGE_COLORS: Record<string, string> = {
  HAS_PREFERENCE:  "#7c3aed",
  HAS_INTEREST:    "#3b82f6",
  HAS_ROLE:        "#22c55e",
  WORKS_ON:        "#f59e0b",
  HAS_SKILL:       "#06b6d4",
  HAS_GOAL:        "#a855f7",
  KNOWS:           "#6b7280",
  RELATED_TO:      "#f97316",
};

// ── Node extra data stored for panel use ──────────────────────────────────────
interface FactNodeData {
  label: string;
  factLabel: string;
  category: string;
  confidence: number | null;
  frequency_count: number | null;
  sourceEventIds: string[];
  isSelected?: boolean;
}

// ── Layout: radial by category ────────────────────────────────────────────────
function buildLayout(graph: FactGraph): { nodes: Node[]; edges: Edge[] } {
  const factsByCat = new Map<string, FactGraphNode[]>();
  const userNode = graph.nodes.find((n) => n.node_type === "user");
  const factNodes = graph.nodes.filter((n) => n.node_type === "fact");

  for (const fn of factNodes) {
    const cat = fn.category ?? "other";
    if (!factsByCat.has(cat)) factsByCat.set(cat, []);
    factsByCat.get(cat)!.push(fn);
  }

  const categories = [...factsByCat.keys()];
  const CENTER = { x: 0, y: 0 };
  const CAT_RADIUS = 320;
  const FACT_RADIUS = 140;

  const rfNodes: Node[] = [];

  if (userNode) {
    rfNodes.push({
      id: userNode.id,
      type: "default",
      position: CENTER,
      data: { label: userNode.label },
      style: {
        background: USER_STYLE.bg,
        border: `2px solid ${USER_STYLE.border}`,
        color: USER_STYLE.text,
        borderRadius: "50%",
        width: 80,
        height: 80,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 12,
        fontWeight: 700,
        boxShadow: `0 0 20px ${USER_STYLE.border}40`,
      },
    });
  }

  categories.forEach((cat, catIdx) => {
    const catAngle = (2 * Math.PI * catIdx) / categories.length - Math.PI / 2;
    const catX = Math.cos(catAngle) * CAT_RADIUS;
    const catY = Math.sin(catAngle) * CAT_RADIUS;
    const facts = factsByCat.get(cat)!;
    const style = CATEGORY_STYLES[cat] ?? DEFAULT_STYLE;

    facts.forEach((fn, fIdx) => {
      const fanAngle = catAngle + ((fIdx - (facts.length - 1) / 2) * 0.5);
      const x = catX + Math.cos(fanAngle) * FACT_RADIUS;
      const y = catY + Math.sin(fanAngle) * FACT_RADIUS;

      const sourceEventIds = fn.source_event_ids ?? [];
      const hasSource = sourceEventIds.length > 0;

      rfNodes.push({
        id: fn.id,
        type: "default",
        position: { x, y },
        data: {
          label: fn.label.length > 28 ? fn.label.slice(0, 26) + "…" : fn.label,
          factLabel: fn.label,
          category: cat,
          confidence: fn.confidence ?? null,
          frequency_count: fn.frequency_count ?? null,
          sourceEventIds,
        },
        style: {
          background: style.bg,
          border: `1.5px solid ${style.border}`,
          color: style.text,
          borderRadius: 8,
          fontSize: 11,
          padding: "6px 10px",
          maxWidth: 160,
          textAlign: "center" as const,
          boxShadow: hasSource
            ? `0 0 12px ${style.border}60`
            : `0 0 8px ${style.border}30`,
          cursor: hasSource ? "pointer" : "default",
          outline: hasSource ? `1px dashed ${style.border}80` : "none",
        },
      });
    });
  });

  const rfEdges: Edge[] = graph.edges.map((e: FactGraphEdge) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.relation === "RELATED_TO" ? "related" : undefined,
    animated: e.relation === "RELATED_TO",
    style: {
      stroke: EDGE_COLORS[e.relation] ?? "#475569",
      strokeWidth: e.relation === "RELATED_TO" ? 1 : 1.5,
      opacity: e.relation === "RELATED_TO" ? 0.5 : 0.8,
    },
    labelStyle: { fill: "#94a3b8", fontSize: 9 },
    labelBgStyle: { fill: "#0f172a", fillOpacity: 0.8 },
  }));

  return { nodes: rfNodes, edges: rfEdges };
}

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  const items = Object.entries(CATEGORY_STYLES).map(([cat, s]) => ({ cat, color: s.border }));
  return (
    <div className="absolute bottom-3 left-3 z-10 bg-zinc-900/90 border border-zinc-700/50
                    rounded-xl p-3 space-y-1 backdrop-blur-sm">
      <p className="text-xs font-medium text-zinc-500 mb-2 uppercase tracking-wide">Categories</p>
      {items.map(({ cat, color }) => (
        <div key={cat} className="flex items-center gap-2">
          <span
            className="inline-block w-2.5 h-2.5 rounded-sm flex-shrink-0"
            style={{ background: color }}
          />
          <span className="text-xs text-zinc-400 capitalize">{cat}</span>
        </div>
      ))}
    </div>
  );
}

// ── Source memories panel ─────────────────────────────────────────────────────
interface SelectedFact {
  label: string;
  category: string;
  confidence: number | null;
  frequency_count: number | null;
  sourceEventIds: string[];
}

function normalizeText(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ");
}

function SourceMemoriesPanel({
  fact,
  onClose,
}: {
  fact: SelectedFact;
  onClose: () => void;
}) {
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

  // Deduplicate loaded events by normalised text content; keep earliest seen.
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
    <div className="absolute top-0 right-0 h-full w-80 z-20 flex flex-col
                    bg-zinc-900/95 border-l border-zinc-700/50 backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-start justify-between p-4 border-b border-zinc-700/50">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="inline-block w-2 h-2 rounded-sm flex-shrink-0"
              style={{ background: style.border }}
            />
            <span className="text-xs text-zinc-500 capitalize">{fact.category}</span>
          </div>
          <p className="text-sm font-semibold text-zinc-100 truncate">{fact.label}</p>
          <div className="flex items-center gap-3 mt-1.5">
            {fact.confidence != null && (
              <span className="text-xs text-zinc-500">
                {(fact.confidence * 100).toFixed(0)}% confidence
              </span>
            )}
            {fact.frequency_count != null && (
              <span className="text-xs text-zinc-500">
                seen {fact.frequency_count}×
              </span>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="ml-2 p-1 rounded-md text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700/50 transition-colors flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="flex items-center gap-2 mb-3">
          <BookOpen className="w-3.5 h-3.5 text-zinc-500" />
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wide">
            Contributing memories
          </p>
        </div>

        {isLoading ? (
          <div className="flex items-center gap-2 text-zinc-500 py-6 justify-center">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-xs">Loading…</span>
          </div>
        ) : unique.length === 0 ? (
          <p className="text-xs text-zinc-600 text-center py-6">
            No source memories tracked for this fact.
          </p>
        ) : (
          <>
            <div className="space-y-2">
              {unique.map(({ event }) => (
                <div
                  key={event.event_id}
                  className="rounded-lg border border-zinc-700/50 bg-zinc-800/40 p-3 group"
                >
                  <p className="text-xs text-zinc-300 line-clamp-3 leading-relaxed">
                    {event.raw_text}
                  </p>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs text-zinc-600">
                      {new Date(event.created_at).toLocaleDateString()}
                    </span>
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

  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => (data ? buildLayout(data) : { nodes: [], edges: [] }),
    [data],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  useEffect(() => { setNodes(initialNodes); }, [initialNodes, setNodes]);
  useEffect(() => { setEdges(initialEdges); }, [initialEdges, setEdges]);

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges],
  );

  const onNodeClick: NodeMouseHandler = useCallback((_evt, node) => {
    const d = node.data as unknown as FactNodeData;
    if (!d.sourceEventIds || d.sourceEventIds.length === 0) return;
    setSelectedFact({
      label: d.factLabel,
      category: d.category,
      confidence: d.confidence,
      frequency_count: d.frequency_count,
      sourceEventIds: d.sourceEventIds,
    });
  }, []);

  const onPaneClick = useCallback(() => setSelectedFact(null), []);

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
    <div className="relative rounded-xl overflow-hidden border border-zinc-700/50"
         style={{ height: 560 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#1e293b"
        />
        <Controls
          style={{
            background: "#0f172a",
            border: "1px solid #334155",
            borderRadius: 8,
          }}
        />
        <MiniMap
          style={{
            background: "#0f172a",
            border: "1px solid #334155",
            borderRadius: 8,
          }}
          nodeColor={(n) => {
            if (n.style?.border) return n.style.border as string;
            return "#475569";
          }}
          maskColor="#0f172a80"
        />
      </ReactFlow>
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
      {selectedFact && (
        <SourceMemoriesPanel fact={selectedFact} onClose={() => setSelectedFact(null)} />
      )}
    </div>
  );
}

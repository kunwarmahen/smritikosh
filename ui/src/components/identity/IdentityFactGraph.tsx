"use client";

import { useCallback, useEffect, useMemo } from "react";
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
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Loader2, Network } from "lucide-react";
import { useFactGraph } from "@/hooks/useFactGraph";
import type { FactGraph, FactGraphNode, FactGraphEdge } from "@/types";

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

  // User node at center
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

  // Category cluster nodes
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

      const confPct = fn.confidence != null ? ` · ${(fn.confidence * 100).toFixed(0)}%` : "";
      rfNodes.push({
        id: fn.id,
        type: "default",
        position: { x, y },
        data: {
          label: fn.label.length > 28 ? fn.label.slice(0, 26) + "…" : fn.label,
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
          boxShadow: `0 0 8px ${style.border}30`,
        },
      });
    });
  });

  // Edges
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

// ── Main component ────────────────────────────────────────────────────────────
export function IdentityFactGraph() {
  const { data, isLoading, isError } = useFactGraph();

  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => (data ? buildLayout(data) : { nodes: [], edges: [] }),
    [data],
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Sync when data loads (useNodesState only initialises once on mount)
  useEffect(() => { setNodes(initialNodes); }, [initialNodes, setNodes]);
  useEffect(() => { setEdges(initialEdges); }, [initialEdges, setEdges]);

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges],
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
    <div className="relative rounded-xl overflow-hidden border border-zinc-700/50"
         style={{ height: 560 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
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
      <div className="absolute top-3 right-3 z-10">
        <span className="text-xs text-zinc-500 bg-zinc-900/80 border border-zinc-700/50
                         rounded-lg px-2 py-1 backdrop-blur-sm">
          {data.nodes.length - 1} fact{data.nodes.length !== 2 ? "s" : ""} · {data.edges.length} link{data.edges.length !== 1 ? "s" : ""}
        </span>
      </div>
    </div>
  );
}

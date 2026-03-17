"use client";

import { useCallback, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Loader2, Network } from "lucide-react";
import { useMemoryEvent, useMemoryLinks } from "@/hooks/useMemoryGraph";
import { importanceLevel } from "@/types";
import type { MemoryEvent, MemoryLink } from "@/types";

// ── Edge colour palette by relation type ─────────────────────────────────────
const RELATION_COLORS: Record<string, string> = {
  caused:      "#f43f5e",   // rose
  preceded:    "#f59e0b",   // amber
  related:     "#7c3aed",   // violet
  contradicts: "#06b6d4",   // cyan
};

const RELATION_LABELS: Record<string, string> = {
  caused:      "caused",
  preceded:    "preceded",
  related:     "related",
  contradicts: "contradicts",
};

// ── Node styles ───────────────────────────────────────────────────────────────
function anchorStyle() {
  return {
    background: "#1e1533",
    border: "2px solid #7c3aed",
    color: "#e9d5ff",
    borderRadius: 12,
    fontSize: 12,
    fontWeight: 600,
    padding: "10px 14px",
    maxWidth: 240,
    textAlign: "center" as const,
    boxShadow: "0 0 24px #7c3aed40",
  };
}

function linkedStyle(importanceScore: number) {
  const level = importanceLevel(importanceScore);
  const border =
    level === "high"   ? "#22c55e" :
    level === "medium" ? "#f59e0b" : "#ef4444";
  return {
    background: "#0f172a",
    border: `1.5px solid ${border}`,
    color: "#94a3b8",
    borderRadius: 10,
    fontSize: 11,
    padding: "8px 12px",
    maxWidth: 200,
    textAlign: "center" as const,
    cursor: "pointer",
    boxShadow: `0 0 8px ${border}20`,
  };
}

// ── Build React Flow nodes + edges from the data ──────────────────────────────
function buildGraph(
  anchor: MemoryEvent,
  links: MemoryLink[],
): { nodes: Node[]; edges: Edge[] } {
  const anchorId = anchor.event_id;
  const anchorPreview =
    anchor.raw_text.length > 120
      ? anchor.raw_text.slice(0, 118) + "…"
      : anchor.raw_text;

  const nodes: Node[] = [
    {
      id: anchorId,
      position: { x: 0, y: 0 },
      data: { label: anchorPreview },
      style: anchorStyle(),
    },
  ];

  const edges: Edge[] = [];

  // Deduplicate linked events (in case there are multiple link types to the same event)
  const seen = new Set<string>();
  let leftIdx = 0;
  let rightIdx = 0;

  for (const link of links) {
    const isOutgoing = link.from_event_id === anchorId;
    const linkedId   = isOutgoing ? link.to_event_id : link.from_event_id;
    const preview    = isOutgoing ? link.to_event_preview : link.from_event_preview;
    const relation   = link.relation_type;
    const color      = RELATION_COLORS[relation] ?? "#475569";

    // Position: outgoing links go right, incoming go left
    if (!seen.has(linkedId)) {
      seen.add(linkedId);
      const xOffset = isOutgoing ? 400 : -400;
      const yIndex  = isOutgoing ? rightIdx++ : leftIdx++;
      const yOffset = (yIndex - 0) * 160;

      nodes.push({
        id: linkedId,
        position: { x: xOffset, y: yOffset - 80 },
        data: {
          label: (preview || "…").length > 100 ? (preview || "").slice(0, 98) + "…" : (preview || "…"),
        },
        style: linkedStyle(0.5),
      });
    }

    const edgeSource = isOutgoing ? anchorId  : linkedId;
    const edgeTarget = isOutgoing ? linkedId  : anchorId;

    edges.push({
      id: link.link_id,
      source: edgeSource,
      target: edgeTarget,
      label: RELATION_LABELS[relation] ?? relation,
      animated: relation === "caused",
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
      style: { stroke: color, strokeWidth: 1.5 },
      labelStyle: { fill: "#94a3b8", fontSize: 9 },
      labelBgStyle: { fill: "#0f172a", fillOpacity: 0.85 },
    });
  }

  return { nodes, edges };
}

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  return (
    <div className="absolute bottom-3 left-3 z-10 bg-slate-900/90 border border-slate-700/50
                    rounded-xl p-3 space-y-1.5 backdrop-blur-sm">
      <p className="text-xs font-medium text-slate-500 mb-2 uppercase tracking-wide">Relations</p>
      {Object.entries(RELATION_LABELS).map(([key, label]) => (
        <div key={key} className="flex items-center gap-2">
          <span
            className="inline-block w-6 h-0.5 flex-shrink-0"
            style={{ background: RELATION_COLORS[key] }}
          />
          <span className="text-xs text-slate-400 capitalize">{label}</span>
        </div>
      ))}
      <div className="pt-1 border-t border-slate-800">
        <p className="text-xs text-slate-600">← predecessors · successors →</p>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export function MemoryGraphView({ eventId }: { eventId: string }) {
  const router = useRouter();
  const { data: anchor, isLoading: loadingEvent, isError: errorEvent } = useMemoryEvent(eventId);
  const { data: linksData, isLoading: loadingLinks, isError: errorLinks } = useMemoryLinks(eventId);

  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!anchor) return { nodes: [], edges: [] };
    return buildGraph(anchor, linksData?.links ?? []);
  }, [anchor, linksData]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Sync when anchor/links load (useNodesState only initialises once on mount)
  useEffect(() => { setNodes(initialNodes); }, [initialNodes, setNodes]);
  useEffect(() => { setEdges(initialEdges); }, [initialEdges, setEdges]);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (node.id !== eventId) {
        router.push(`/dashboard/memories/${node.id}`);
      }
    },
    [eventId, router],
  );

  const isLoading = loadingEvent || loadingLinks;
  const isError   = errorEvent || errorLinks;

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-slate-500 py-12 justify-center">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span className="text-sm">Loading memory graph…</span>
      </div>
    );
  }

  if (isError || !anchor) {
    return (
      <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
        <p className="text-rose-400 text-sm">Failed to load memory graph.</p>
      </div>
    );
  }

  if (!linksData?.links.length) {
    return (
      <div className="card text-center py-12">
        <Network className="w-10 h-10 text-slate-700 mx-auto mb-3" />
        <p className="text-slate-400 text-sm font-medium">No narrative links yet.</p>
        <p className="text-slate-600 text-xs mt-1">
          Links are created during consolidation when events are causally or temporally connected.
        </p>
      </div>
    );
  }

  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-700/50"
         style={{ height: 500 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
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
          maskColor="#0f172a80"
        />
      </ReactFlow>
      <Legend />
      <div className="absolute top-3 right-3 z-10">
        <span className="text-xs text-slate-500 bg-slate-900/80 border border-slate-700/50
                         rounded-lg px-2 py-1 backdrop-blur-sm">
          {linksData.links.length} link{linksData.links.length !== 1 ? "s" : ""}
           · click a node to navigate
        </span>
      </div>
    </div>
  );
}

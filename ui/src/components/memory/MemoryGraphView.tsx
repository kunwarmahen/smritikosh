"use client";

import { useCallback, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Network } from "lucide-react";
import { useMemoryEvent, useMemoryLinks } from "@/hooks/useMemoryGraph";
import { ForceGraph2D, GRAPH_BG, roundRect } from "@/lib/graph-shared";
import type { MemoryEvent, MemoryLink } from "@/types";

// ── Palette ───────────────────────────────────────────────────────────────────
const RELATION_COLORS: Record<string, string> = {
  caused:      "#f43f5e",
  preceded:    "#f59e0b",
  related:     "#7c3aed",
  contradicts: "#06b6d4",
};
const RELATION_LABELS: Record<string, string> = {
  caused:      "caused",
  preceded:    "preceded",
  related:     "related",
  contradicts: "contradicts",
};

// ── Types ─────────────────────────────────────────────────────────────────────
interface MemGraphNode {
  id: string;
  preview: string;
  isAnchor: boolean;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}
interface MemGraphLink {
  source: string | MemGraphNode;
  target: string | MemGraphNode;
  relation: string;
  link_id: string;
}

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  return (
    <div className="absolute bottom-3 left-3 z-10 bg-zinc-900/90 border border-zinc-700/50
                    rounded-xl p-3 backdrop-blur-sm">
      <p className="text-xs font-medium text-zinc-500 mb-2 uppercase tracking-wide">Relations</p>
      <div className="space-y-1.5">
        {Object.entries(RELATION_LABELS).map(([key, label]) => (
          <div key={key} className="flex items-center gap-2">
            <span className="inline-block w-6 h-0.5 flex-shrink-0" style={{ background: RELATION_COLORS[key] }} />
            <span className="text-xs text-zinc-400 capitalize">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export function MemoryGraphView({ eventId }: { eventId: string }) {
  const router = useRouter();
  const graphRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const { data: anchor, isLoading: loadingEvent, isError: errorEvent } = useMemoryEvent(eventId);
  const { data: linksData, isLoading: loadingLinks, isError: errorLinks } = useMemoryLinks(eventId);

  const graphData = useMemo(() => {
    if (!anchor) return { nodes: [], links: [] };

    const nodesMap = new Map<string, MemGraphNode>();
    const links: MemGraphLink[] = [];

    const anchorPreview = anchor.raw_text.length > 110
      ? anchor.raw_text.slice(0, 108) + "…"
      : anchor.raw_text;

    nodesMap.set(eventId, { id: eventId, preview: anchorPreview, isAnchor: true, fx: 0, fy: 0 });

    for (const link of linksData?.links ?? []) {
      const isOutgoing = link.from_event_id === eventId;
      const linkedId   = isOutgoing ? link.to_event_id   : link.from_event_id;
      const preview    = isOutgoing ? link.to_event_preview : link.from_event_preview;

      if (!nodesMap.has(linkedId)) {
        const p = (preview || "…").length > 90 ? (preview || "").slice(0, 88) + "…" : (preview || "…");
        nodesMap.set(linkedId, { id: linkedId, preview: p, isAnchor: false });
      }

      links.push({
        source: isOutgoing ? eventId  : linkedId,
        target: isOutgoing ? linkedId : eventId,
        relation: link.relation_type,
        link_id: link.link_id,
      });
    }

    return { nodes: Array.from(nodesMap.values()), links };
  }, [anchor, linksData, eventId]);

  const drawNode = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = raw as MemGraphNode;
      const nx = node.x ?? 0;
      const ny = node.y ?? 0;
      const s  = 1 / globalScale;
      const W  = (node.isAnchor ? 200 : 170) * s;
      const H  = 44 * s;
      const rx = 8 * s;

      if (node.isAnchor) {
        ctx.shadowColor = "#7c3aed";
        ctx.shadowBlur  = 16 * s;
      }

      roundRect(ctx, nx - W / 2, ny - H / 2, W, H, rx);
      ctx.fillStyle = node.isAnchor ? "#1e1533" : "#0f172a";
      ctx.fill();
      ctx.shadowBlur  = 0;
      ctx.strokeStyle = node.isAnchor ? "#7c3aed" : "#334155";
      ctx.lineWidth   = (node.isAnchor ? 2 : 1.5) * s;
      ctx.stroke();

      const label = node.isAnchor ? "This memory" : "Linked event";
      ctx.fillStyle = node.isAnchor ? "#a78bfa" : "#64748b";
      ctx.font      = `${8 * s}px system-ui, sans-serif`;
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, nx, ny - 12 * s);

      const maxChars = node.isAnchor ? 32 : 28;
      const text = node.preview.length > maxChars ? node.preview.slice(0, maxChars - 1) + "…" : node.preview;
      ctx.fillStyle = node.isAnchor ? "#e9d5ff" : "#94a3b8";
      ctx.font      = `${10 * s}px system-ui, sans-serif`;
      ctx.fillText(text, nx, ny + 7 * s);
    },
    [],
  );

  const nodePointerArea = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = raw as MemGraphNode;
      const s = 1 / globalScale;
      const W = (node.isAnchor ? 200 : 170) * s;
      roundRect(ctx, (node.x ?? 0) - W / 2, (node.y ?? 0) - 22 * s, W, 44 * s, 8 * s);
      ctx.fillStyle = color;
      ctx.fill();
    },
    [],
  );

  const handleNodeClick = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => {
      const node = raw as MemGraphNode;
      if (!node.isAnchor) router.push(`/dashboard/memories/${node.id}`);
    },
    [router],
  );

  const getLinkColor = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => (RELATION_COLORS[(raw as MemGraphLink).relation] ?? "#475569") + "cc",
    [],
  );

  const getParticleCount = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => ((raw as MemGraphLink).relation === "caused" ? 3 : 0),
    [],
  );

  const isLoading = loadingEvent || loadingLinks;
  const isError   = errorEvent   || errorLinks;

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-zinc-500 py-12 justify-center">
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
        <Network className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
        <p className="text-zinc-400 text-sm font-medium">No narrative links yet.</p>
        <p className="text-zinc-600 text-xs mt-1">
          Links are created during consolidation when events are causally or temporally connected.
        </p>
      </div>
    );
  }

  return (
    <div className="relative rounded-xl overflow-hidden border border-zinc-700/50 bg-zinc-950"
         style={{ height: 500 }}>
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeCanvasObject={drawNode}
        nodePointerAreaPaint={nodePointerArea}
        onNodeClick={handleNodeClick}
        linkColor={getLinkColor}
        linkWidth={1.5}
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor={getLinkColor}
        linkDirectionalParticles={getParticleCount}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleColor={getLinkColor}
        backgroundColor={GRAPH_BG}
        onEngineStop={() => graphRef.current?.zoomToFit(400, 60)}
        nodeLabel=""
      />
      <Legend />
      <div className="absolute top-3 right-3 z-10">
        <span className="text-xs text-zinc-500 bg-zinc-900/80 border border-zinc-700/50
                         rounded-lg px-2 py-1 backdrop-blur-sm">
          {linksData.links.length} link{linksData.links.length !== 1 ? "s" : ""} · click a node to navigate
        </span>
      </div>
    </div>
  );
}

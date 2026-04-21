"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Network } from "lucide-react";
import { useMemoryEvent, useMemoryLinks } from "@/hooks/useMemoryGraph";
import { ForceGraph2D, GRAPH_BG } from "@/lib/graph-shared";
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
const RELATION_PARTICLES: Record<string, number> = {
  caused:      4,
  preceded:    2,
  related:     2,
  contradicts: 3,
};

// ── Types ─────────────────────────────────────────────────────────────────────
interface MemGraphNode {
  id: string;
  preview: string;
  isAnchor: boolean;
  relation?: string;
  val?: number;
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
      <p className="text-[10px] font-semibold text-zinc-500 mb-2 uppercase tracking-widest">Relations</p>
      <div className="space-y-1.5">
        {Object.entries(RELATION_LABELS).map(([key, label]) => (
          <div key={key} className="flex items-center gap-2">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: RELATION_COLORS[key], boxShadow: `0 0 6px ${RELATION_COLORS[key]}` }}
            />
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

    const anchorPreview = anchor.raw_text.length > 60
      ? anchor.raw_text.slice(0, 58) + "…"
      : anchor.raw_text;

    nodesMap.set(eventId, {
      id: eventId, preview: anchorPreview, isAnchor: true, val: 9, fx: 0, fy: 0,
    });

    const allLinks = linksData?.links ?? [];
    const linkedIds = [...new Set(allLinks.map(l =>
      l.from_event_id === eventId ? l.to_event_id : l.from_event_id
    ))];
    const total = linkedIds.length;

    for (const link of allLinks) {
      const isOutgoing = link.from_event_id === eventId;
      const linkedId   = isOutgoing ? link.to_event_id   : link.from_event_id;
      const preview    = isOutgoing ? link.to_event_preview : link.from_event_preview;

      if (!nodesMap.has(linkedId)) {
        const p = (preview || "…");
        const short = p.length > 55 ? p.slice(0, 53) + "…" : p;
        const idx   = linkedIds.indexOf(linkedId);
        const angle = (idx / Math.max(total, 1)) * 2 * Math.PI;
        const dist  = 180;
        nodesMap.set(linkedId, {
          id: linkedId, preview: short, isAnchor: false, relation: link.relation_type, val: 4,
          x: dist * Math.cos(angle), y: dist * Math.sin(angle),
        });
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

  // ── Node orb painter ──────────────────────────────────────────────────────
  const drawNode = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node  = raw as MemGraphNode;
      const nx    = node.x ?? 0;
      const ny    = node.y ?? 0;
      const s     = 1 / globalScale;
      const r     = (node.isAnchor ? 26 : 18) * s;
      const color = node.isAnchor
        ? "#7c3aed"
        : (RELATION_COLORS[node.relation ?? ""] ?? "#475569");

      // Atmospheric halo
      ctx.shadowColor = color;
      ctx.shadowBlur  = (node.isAnchor ? 32 : 22) * s;
      ctx.beginPath();
      ctx.arc(nx, ny, r * 1.25, 0, Math.PI * 2);
      ctx.fillStyle = color + "18";
      ctx.fill();
      ctx.shadowBlur = 0;

      // Orb body — radial gradient for 3-D depth
      const grad = ctx.createRadialGradient(
        nx - r * 0.32, ny - r * 0.32, 0,
        nx, ny, r,
      );
      grad.addColorStop(0,   color + "ff");
      grad.addColorStop(0.5, color + "99");
      grad.addColorStop(1,   color + "28");
      ctx.beginPath();
      ctx.arc(nx, ny, r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Rim
      ctx.strokeStyle = color + "cc";
      ctx.lineWidth   = 1.2 * s;
      ctx.stroke();

      // Specular highlight
      ctx.beginPath();
      ctx.arc(nx - r * 0.32, ny - r * 0.32, r * 0.22, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.fill();

      // Relation / anchor chip
      const chipY = ny + r + 7 * s;
      ctx.font         = `600 ${8 * s}px system-ui, sans-serif`;
      ctx.textAlign    = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle    = color;
      ctx.fillText(
        node.isAnchor ? "anchor" : (RELATION_LABELS[node.relation ?? ""] ?? "linked"),
        nx, chipY,
      );

      // Preview text (two short lines)
      const words   = node.preview.split(" ");
      const line1   = words.slice(0, 4).join(" ") + (words.length > 4 ? " …" : "");
      ctx.font      = `${8.5 * s}px system-ui, sans-serif`;
      ctx.fillStyle = node.isAnchor ? "#ddd6fe" : "#94a3b8";
      ctx.fillText(line1, nx, chipY + 11 * s);
    },
    [],
  );

  // ── Pointer hit area — circle covering orb + label ───────────────────────
  const nodePointerArea = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = raw as MemGraphNode;
      const s    = 1 / globalScale;
      const r    = (node.isAnchor ? 26 : 18) * s;
      ctx.beginPath();
      ctx.arc(node.x ?? 0, node.y ?? 0, r * 1.4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    },
    [],
  );

  // ── Link label pill ───────────────────────────────────────────────────────
  const drawLink = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const link  = raw as MemGraphLink;
      const src   = link.source as MemGraphNode;
      const tgt   = link.target as MemGraphNode;
      if (src.x == null || tgt.x == null) return;

      const mx    = (src.x + tgt.x) / 2;
      const my    = (src.y + tgt.y) / 2;
      const label = RELATION_LABELS[link.relation] ?? link.relation;
      const color = RELATION_COLORS[link.relation] ?? "#475569";
      const s     = 1 / globalScale;

      ctx.font    = `600 ${8 * s}px system-ui, sans-serif`;
      const tw    = ctx.measureText(label).width;
      const pw    = tw + 10 * s;
      const ph    = 13 * s;

      ctx.beginPath();
      ctx.roundRect(mx - pw / 2, my - ph / 2, pw, ph, ph / 2);
      ctx.fillStyle   = "#09090b";
      ctx.fill();
      ctx.strokeStyle = color + "aa";
      ctx.lineWidth   = 0.8 * s;
      ctx.stroke();

      ctx.fillStyle    = color;
      ctx.textAlign    = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, mx, my);
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
    (raw: any) => (RELATION_COLORS[(raw as MemGraphLink).relation] ?? "#475569") + "99",
    [],
  );

  const getLinkWidth = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => {
      const rel = (raw as MemGraphLink).relation;
      return rel === "caused" || rel === "contradicts" ? 2 : 1.5;
    },
    [],
  );

  const getParticleCount = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => RELATION_PARTICLES[(raw as MemGraphLink).relation] ?? 1,
    [],
  );

  // Set d3 forces after the graph mounts so nodes spread out properly
  useEffect(() => {
    const fg = graphRef.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength(-600);
    fg.d3Force("link")?.distance(200);
    fg.d3ReheatSimulation?.();
  }, [graphData]);

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
        nodeRelSize={8}
        onNodeClick={handleNodeClick}
        linkColor={getLinkColor}
        linkWidth={getLinkWidth}
        linkDirectionalArrowLength={0}
        linkDirectionalParticles={getParticleCount}
        linkDirectionalParticleWidth={2.5}
        linkDirectionalParticleColor={getLinkColor}
        linkCanvasObjectMode="after"
        linkCanvasObject={drawLink}
        linkCurvature={0.15}
        backgroundColor={GRAPH_BG}
        cooldownTicks={150}
        onEngineStop={() => graphRef.current?.zoomToFit(400, 80)}
        nodeLabel=""
      />
      <Legend />
      <div className="absolute top-3 right-3 z-10">
        <span className="text-xs text-zinc-500 bg-zinc-900/80 border border-zinc-700/50
                         rounded-lg px-2 py-1 backdrop-blur-sm">
          {linksData.links.length} link{linksData.links.length !== 1 ? "s" : ""} · click to navigate
        </span>
      </div>
    </div>
  );
}

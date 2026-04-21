"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Network, BookOpen, ExternalLink, X, ChevronLeft } from "lucide-react";
import Link from "next/link";
import { useFactGraph } from "@/hooks/useFactGraph";
import { useQueries } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";
import { ForceGraph2D, ForceGraph3D, GRAPH_BG } from "@/lib/graph-shared";
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

// ── Types ─────────────────────────────────────────────────────────────────────
type NodeType = "user" | "fact" | "category";

interface GraphNode extends Omit<FactGraphNode, "node_type"> {
  node_type: NodeType;
  x?: number;
  y?: number;
  z?: number;
  fx?: number;
  fy?: number;
}

interface GraphLink extends FactGraphEdge {
  key_label?: string;
}

interface SelectedFact {
  label: string;
  category: string;
  confidence: number | null;
  frequency_count: number | null;
  sourceEventIds: string[];
}

// ── Graph data transform ──────────────────────────────────────────────────────
function buildGraphData(data: { nodes: FactGraphNode[]; edges: FactGraphEdge[] }) {
  const userNode = data.nodes.find((n) => n.node_type === "user");
  if (!userNode) return { nodes: [] as GraphNode[], links: [] as GraphLink[] };

  const factNodes = data.nodes.filter((n) => n.node_type === "fact");

  // Collect unique categories
  const categories = [...new Set(factNodes.map((n) => n.category ?? "other"))];
  const catCount = categories.length;

  // Position category nodes in a ring around origin, fact nodes fanned from their category
  const catRadius = Math.max(180, catCount * 40);

  const categoryNodes: GraphNode[] = categories.map((cat, i) => {
    const angle = (2 * Math.PI * i) / catCount;
    return {
      id: `cat__${cat}`,
      label: cat,
      node_type: "category",
      category: cat,
      x: catRadius * Math.cos(angle),
      y: catRadius * Math.sin(angle),
    };
  });

  // Fact nodes clustered around their category node
  const factsByCategory: Record<string, FactGraphNode[]> = {};
  for (const f of factNodes) {
    const cat = f.category ?? "other";
    (factsByCategory[cat] ??= []).push(f);
  }

  const positionedFacts: GraphNode[] = [];
  for (const cat of categories) {
    const catIdx = categories.indexOf(cat);
    const catAngle = (2 * Math.PI * catIdx) / catCount;
    const facts = factsByCategory[cat] ?? [];
    const leafRadius = Math.max(120, facts.length * 20);
    const spreadAngle = Math.min(Math.PI * 0.8, (facts.length * Math.PI) / 6);
    facts.forEach((f, j) => {
      const offset = facts.length > 1
        ? catAngle + spreadAngle * ((j / (facts.length - 1)) - 0.5)
        : catAngle;
      const cx = catRadius * Math.cos(catAngle);
      const cy = catRadius * Math.sin(catAngle);
      positionedFacts.push({
        ...f,
        node_type: "fact",
        x: cx + leafRadius * Math.cos(offset),
        y: cy + leafRadius * Math.sin(offset),
      });
    });
  }

  const nodes: GraphNode[] = [
    { ...userNode, node_type: "user", x: 0, y: 0 },
    ...categoryNodes,
    ...positionedFacts,
  ];

  // Build links: user → category, category → fact, keep RELATED_TO cross-links
  const links: GraphLink[] = [];

  for (const cat of categories) {
    links.push({
      id: `edge__user__${cat}`,
      source: userNode.id,
      target: `cat__${cat}`,
      relation: "HAS_CATEGORY",
    });
  }

  for (const f of factNodes) {
    const cat = f.category ?? "other";
    links.push({
      id: `edge__cat__${f.id}`,
      source: `cat__${cat}`,
      target: f.id,
      relation: f.key ?? "",
      key_label: f.key?.replace(/_/g, " ") ?? "",
    });
  }

  for (const e of data.edges) {
    if (e.relation === "RELATED_TO") {
      links.push({ ...e });
    }
  }

  return { nodes, links };
}

// ── 3D sprite helpers ─────────────────────────────────────────────────────────
// Bare text sprite — no background box
function makeTextLabel(label: string, subLabel: string, textColor: string, fontSize: number) {
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-explicit-any
  const THREE = require("three") as any;
  const lineH = fontSize + 3;
  const lines = subLabel ? 2 : 1;
  const canvasW = Math.max(160, Math.max(label.length, subLabel.length) * (fontSize * 0.62) + 8);
  const canvasH = lines * lineH + 4;
  const canvas = document.createElement("canvas");
  canvas.width = canvasW; canvas.height = canvasH;
  const ctx = canvas.getContext("2d")!;
  ctx.textAlign = "center";
  if (subLabel) {
    ctx.fillStyle = textColor + "cc";
    ctx.font = `${fontSize - 3}px system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillText(subLabel.toUpperCase(), canvasW / 2, lineH * 0.5 + 2);
  }
  ctx.fillStyle = textColor;
  ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  const displayLabel = label.length > 22 ? label.slice(0, 20) + "…" : label;
  ctx.fillText(displayLabel, canvasW / 2, (subLabel ? lineH * 1.5 : lineH * 0.5) + 2);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set((canvasW / canvasH) * 16, 16, 1);
  return sprite;
}

function makeNode3D(node: GraphNode) {
  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-explicit-any
  const THREE = require("three") as any;

  if (node.node_type === "user") {
    const group = new THREE.Group();

    // Gem core — low-poly icosahedron
    const coreGeo = new THREE.IcosahedronGeometry(7, 0);
    const coreMat = new THREE.MeshLambertMaterial({ color: 0x7c3aed, emissive: 0x4c1d95, emissiveIntensity: 0.6 });
    group.add(new THREE.Mesh(coreGeo, coreMat));

    // Wireframe shell over the gem
    const wireMat = new THREE.MeshBasicMaterial({ color: 0xa855f7, wireframe: true, transparent: true, opacity: 0.35 });
    group.add(new THREE.Mesh(new THREE.IcosahedronGeometry(7.2, 0), wireMat));

    // Orbiting ring
    const ringGeo = new THREE.TorusGeometry(11, 0.6, 6, 32);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: 0.55 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 3;
    group.add(ring);

    const label = makeTextLabel("You", "", "#e9d5ff", 16);
    label.position.set(0, 18, 0);
    group.add(label);
    return group;
  }

  if (node.node_type === "category") {
    const style = CATEGORY_STYLES[node.category ?? ""] ?? DEFAULT_STYLE;
    const color = parseInt(style.border.slice(1), 16);
    const geo = new THREE.SphereGeometry(5, 8, 6);
    const mat = new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.75 });
    const mesh = new THREE.Mesh(geo, mat);
    const label = makeTextLabel(node.label ?? node.category ?? "", "", style.text, 14);
    label.position.set(0, 10, 0);
    mesh.add(label);
    return mesh;
  }

  // fact node — small orb + bare text label above
  const style = CATEGORY_STYLES[node.category ?? ""] ?? DEFAULT_STYLE;
  const color = parseInt(style.border.slice(1), 16);
  const geo = new THREE.SphereGeometry(3, 12, 12);
  const mat = new THREE.MeshLambertMaterial({ color, emissive: color, emissiveIntensity: 0.4 });
  const mesh = new THREE.Mesh(geo, mat);
  const keyLabel = (node.key ?? "").replace(/_/g, " ");
  const label = makeTextLabel(node.label ?? "", keyLabel, style.text, 13);
  label.position.set(0, 8, 0);
  mesh.add(label);
  return mesh;
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

  const isLoading = !token || results.some((r) => r.isPending);

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
    <div className="absolute top-0 right-0 h-full w-96 z-20 flex flex-col bg-zinc-900/95 border-l border-zinc-700/50 backdrop-blur-sm">
      <div className="flex items-start justify-between p-4 border-b border-zinc-700/50 flex-shrink-0">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-block w-2 h-2 rounded-sm flex-shrink-0" style={{ background: style.border }} />
            <span className="text-xs text-zinc-500 capitalize">{fact.category}</span>
          </div>
          <p className="text-sm font-semibold text-zinc-100 truncate">{fact.label}</p>
          <div className="flex items-center gap-3 mt-1.5">
            {fact.confidence != null && (
              <span className="text-xs text-zinc-500">{(fact.confidence * 100).toFixed(0)}% confidence</span>
            )}
            {fact.frequency_count != null && (
              <span className="text-xs text-zinc-500">seen {fact.frequency_count}×</span>
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

      <div className="flex-1 overflow-y-auto p-4">
        <div className="flex items-center gap-2 mb-3">
          <BookOpen className="w-3.5 h-3.5 text-zinc-500" />
          <p className="text-xs font-medium text-zinc-500 uppercase tracking-wide">Contributing memories</p>
        </div>
        {isLoading ? (
          <div className="flex items-center gap-2 text-zinc-500 py-6 justify-center">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-xs">Loading…</span>
          </div>
        ) : unique.length === 0 ? (
          <p className="text-xs text-zinc-600 text-center py-6">No source memories tracked for this fact.</p>
        ) : (
          <>
            <div className="space-y-2">
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
export function IdentityFactGraph({ onClose }: { onClose?: () => void }) {
  const { data, isLoading, isError } = useFactGraph();
  const [selectedFact, setSelectedFact] = useState<SelectedFact | null>(null);
  const [is3D, setIs3D] = useState(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [graphDims, setGraphDims] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      if (el.clientWidth > 0) setGraphDims({ width: el.clientWidth, height: el.clientHeight });
    });
    ro.observe(el);
    if (el.clientWidth > 0) setGraphDims({ width: el.clientWidth, height: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [] as GraphNode[], links: [] as GraphLink[] };
    return buildGraphData(data);
  }, [data]);

  // ── 2-D coordinate hit-test (avoids shadow-canvas linkedProp bug) ───────────
  const [cursor2D, setCursor2D] = useState<"default" | "pointer">("default");

  const hitTestNode2D = useCallback(
    (gx: number, gy: number, k: number): GraphNode | null => {
      for (const raw of graphData.nodes) {
        const node = raw as GraphNode;
        if (node.x == null || node.y == null) continue;
        const nx = node.x;
        const ny = node.y;
        if (node.node_type === "user") {
          if (Math.hypot(gx - nx, gy - ny) <= 24 / k) return node;
        } else if (node.node_type === "category") {
          if (Math.hypot(gx - nx, gy - ny) <= 20 / k) return node;
        } else {
          // rect covers orb + key/value labels above
          if (
            gx >= nx - 80 / k && gx <= nx + 80 / k &&
            gy >= ny - 28 / k && gy <= ny + 8 / k
          ) return node;
        }
      }
      return null;
    },
    [graphData.nodes],
  );

  const handle2DClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const fg = graphRef.current;
      if (!fg) return;
      const rect = e.currentTarget.getBoundingClientRect();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const coords = (fg as any).screen2GraphCoords(e.clientX - rect.left, e.clientY - rect.top);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const k: number = (fg as any).zoom?.() ?? 1;
      const node = hitTestNode2D(coords.x, coords.y, k);
      if (node?.node_type === "fact") {
        setSelectedFact({
          label: node.label,
          category: node.category ?? "other",
          confidence: node.confidence ?? null,
          frequency_count: node.frequency_count ?? null,
          sourceEventIds: node.source_event_ids ?? [],
        });
      }
    },
    [hitTestNode2D],
  );

  const handle2DMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const fg = graphRef.current;
      if (!fg) return;
      const rect = e.currentTarget.getBoundingClientRect();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const coords = (fg as any).screen2GraphCoords(e.clientX - rect.left, e.clientY - rect.top);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const k: number = (fg as any).zoom?.() ?? 1;
      const node = hitTestNode2D(coords.x, coords.y, k);
      setCursor2D(node?.node_type === "fact" ? "pointer" : "default");
    },
    [hitTestNode2D],
  );

  // ── 2-D drawing ─────────────────────────────────────────────────────────────
  const drawNode2D = useCallback(
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

      if (node.node_type === "category") {
        const style = CATEGORY_STYLES[node.category ?? ""] ?? DEFAULT_STYLE;
        const r = 20 * s;
        ctx.beginPath();
        ctx.arc(nx, ny, r, 0, 2 * Math.PI);
        ctx.fillStyle = style.bg;
        ctx.shadowColor = style.border;
        ctx.shadowBlur = 8 * s;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = style.border;
        ctx.lineWidth = 2 * s;
        ctx.stroke();
        ctx.fillStyle = style.text;
        ctx.font = `bold ${10 * s}px system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText((node.label ?? "").toUpperCase(), nx, ny);
        return;
      }

      // fact node — solid orb + floating text
      const style = CATEGORY_STYLES[node.category ?? ""] ?? DEFAULT_STYLE;
      const hasSource = (node.source_event_ids?.length ?? 0) > 0;
      const orbR = 6 * s;

      ctx.beginPath();
      ctx.arc(nx, ny, orbR, 0, 2 * Math.PI);
      ctx.fillStyle = style.border;
      if (hasSource) {
        ctx.shadowColor = style.border;
        ctx.shadowBlur = 12 * s;
      }
      ctx.fill();
      ctx.shadowBlur = 0;

      const keyLabel = (node.key ?? "").replace(/_/g, " ").toUpperCase();
      ctx.fillStyle = style.text + "cc";
      ctx.font = `${10 * s}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(keyLabel, nx, ny - orbR - 12 * s);

      const label = (node.label?.length ?? 0) > 22 ? node.label!.slice(0, 20) + "…" : (node.label ?? "");
      ctx.fillStyle = style.text;
      ctx.font = `bold ${13 * s}px system-ui, sans-serif`;
      ctx.fillText(label, nx, ny - orbR - 1 * s);
    },
    [],
  );

  // ── 2-D pointer area (defines drag/hover hit zones matching visual sizes) ──────
  const paintNodeArea2D = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any, color: string, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = raw as GraphNode;
      const nx = node.x ?? 0;
      const ny = node.y ?? 0;
      const s = 1 / globalScale;
      ctx.beginPath();
      if (node.node_type === "user") {
        ctx.arc(nx, ny, 24 * s, 0, 2 * Math.PI);
      } else if (node.node_type === "category") {
        ctx.arc(nx, ny, 20 * s, 0, 2 * Math.PI);
      } else {
        ctx.arc(nx, ny, 6 * s, 0, 2 * Math.PI);
      }
      ctx.fillStyle = color;
      ctx.fill();
    },
    [],
  );

  // ── 3-D node click ──────────────────────────────────────────────────────────
  const handleNodeClick3D = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => {
      if (!raw) return;
      const node = raw as GraphNode;
      if (node.node_type === "fact") {
        setSelectedFact({
          label: node.label,
          category: node.category ?? "other",
          confidence: node.confidence ?? null,
          frequency_count: node.frequency_count ?? null,
          sourceEventIds: node.source_event_ids ?? [],
        });
      }
    },
    [],
  );

  // ── Link styling ─────────────────────────────────────────────────────────────
  // 2D canvas supports 8-char hex alpha; Three.js LineBasicMaterial does not honour
  // alpha from hex without transparent:true, so 3D gets 6-char hex only.
  const getLinkColor = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => {
      const link = raw as GraphLink;
      if (link.relation === "HAS_CATEGORY") return "#94a3b8cc";
      if (link.relation === "RELATED_TO") return "#64748baa";
      const node = graphData.nodes.find((n) => n.id === link.target);
      if (node?.node_type === "fact") {
        const style = CATEGORY_STYLES[node.category ?? ""] ?? DEFAULT_STYLE;
        return style.border + "cc";
      }
      return "#94a3b8cc";
    },
    [graphData],
  );

  const getLinkColor3D = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => {
      const link = raw as GraphLink;
      if (link.relation === "HAS_CATEGORY") return "#94a3b8";
      if (link.relation === "RELATED_TO") return "#64748b";
      const node = graphData.nodes.find((n) => n.id === link.target);
      if (node?.node_type === "fact") {
        const style = CATEGORY_STYLES[node.category ?? ""] ?? DEFAULT_STYLE;
        return style.border;
      }
      return "#94a3b8";
    },
    [graphData],
  );

  const getLinkWidth = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => {
      const link = raw as GraphLink;
      if (link.relation === "RELATED_TO") return 0.5;
      if (link.relation === "HAS_CATEGORY") return 1;
      return 1.5;
    },
    [],
  );

  const getParticleCount = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (raw: any) => ((raw as GraphLink).relation === "RELATED_TO" ? 2 : 0),
    [],
  );

  // ── 3-D object factory ───────────────────────────────────────────────────────
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeThreeObject = useCallback((raw: any) => makeNode3D(raw as GraphNode), []);

  // ── Zoom to fit after graph settles ─────────────────────────────────────────
  // useEffect is more reliable than onEngineStop; 600ms covers warmup + first render.
  useEffect(() => {
    if (is3D || !graphData.nodes.length) return;
    // Clear fx/fy that the library pins on dragged nodes — prevents frozen nodes after toggle
    for (const n of graphData.nodes) {
      delete (n as GraphNode).fx;
      delete (n as GraphNode).fy;
    }
    const id = setTimeout(() => {
      const fg = graphRef.current;
      if (!fg) return;
      fg.centerAt(0, 0, 400);
      fg.zoom(4.0, 400);
    }, 600);
    return () => clearTimeout(id);
  }, [is3D, graphData]);

  // ── Props ────────────────────────────────────────────────────────────────────
  const baseProps = {
    graphData,
    linkWidth: getLinkWidth,
    linkDirectionalParticles: getParticleCount,
    linkDirectionalParticleWidth: 1.5,
    backgroundColor: GRAPH_BG,
    warmupTicks: 300,
    cooldownTicks: 0,
  };

  const props2D = {
    ...baseProps,
    linkColor: getLinkColor,
    linkDirectionalParticleColor: getLinkColor,
  };

  const props3D = {
    ...baseProps,
    linkColor: getLinkColor3D,
    linkDirectionalParticleColor: getLinkColor3D,
  };

  // ── Early returns ────────────────────────────────────────────────────────────
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

  const factCount = data.nodes.filter((n) => n.node_type === "fact").length;
  const linkCount = data.edges.length;

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden bg-zinc-950"
      style={{ height: "100%" }}
    >
      {is3D ? (
        <ForceGraph3D
          ref={graphRef}
          {...props3D}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          onNodeClick={handleNodeClick3D}
        />
      ) : (
        <div
          style={{ width: "100%", height: "100%", cursor: cursor2D }}
          onClick={handle2DClick}
          onMouseMove={handle2DMouseMove}
        >
          <ForceGraph2D
            ref={graphRef}
            {...props2D}
            width={selectedFact ? graphDims.width - 384 : graphDims.width}
            height={graphDims.height}
            nodeCanvasObject={drawNode2D}
            nodePointerAreaPaint={paintNodeArea2D}
          />
        </div>
      )}

      <Legend />

      {/* Back button */}
      {onClose && (
        <button
          onClick={onClose}
          className="absolute top-3 left-3 z-10 flex items-center gap-1.5 text-xs px-2.5 py-1.5
                     bg-zinc-900/80 border border-zinc-700/50 rounded-lg text-zinc-400
                     hover:text-zinc-200 backdrop-blur-sm transition-colors"
        >
          <ChevronLeft className="w-3.5 h-3.5" />
          Back
        </button>
      )}

      {/* Top-right controls */}
      <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
        {selectedFact === null && (
          <span className="text-xs text-zinc-500 bg-zinc-900/80 border border-zinc-700/50
                           rounded-lg px-2 py-1 backdrop-blur-sm">
            Click a fact to see source memories
          </span>
        )}
        <span className="text-xs text-zinc-500 bg-zinc-900/80 border border-zinc-700/50
                         rounded-lg px-2 py-1 backdrop-blur-sm">
          {factCount} fact{factCount !== 1 ? "s" : ""} · {linkCount} link{linkCount !== 1 ? "s" : ""}
        </span>
        {/* 2D / 3D toggle */}
        <div className="flex items-center bg-zinc-900/80 border border-zinc-700/50 rounded-lg overflow-hidden backdrop-blur-sm">
          <button
            onClick={() => setIs3D(false)}
            className={`px-2.5 py-1 text-xs font-medium transition-colors ${
              !is3D
                ? "bg-violet-600 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            2D
          </button>
          <button
            onClick={() => setIs3D(true)}
            className={`px-2.5 py-1 text-xs font-medium transition-colors ${
              is3D
                ? "bg-violet-600 text-white"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            3D
          </button>
        </div>
      </div>

      {selectedFact && (
        <SourceMemoriesPanel fact={selectedFact} onClose={() => setSelectedFact(null)} />
      )}
    </div>
  );
}

import dynamic from "next/dynamic";

// Single dynamic import shared by all graph views — avoids bundling aframe (3D/VR variant)
export const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });
export const ForceGraph3D = dynamic(
  () =>
    import("@/components/graph/ForceGraph3DWrapper").then((m) => ({
      default: m.ForceGraph3DWrapper,
    })),
  { ssr: false },
);

export const GRAPH_BG = "#09090b";

// D3 force simulation defaults used across graph views
export const GRAPH_FORCE_CONFIG = {
  chargeStrength: -200,
  linkDistance: 100,
} as const;

export function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  w: number, h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

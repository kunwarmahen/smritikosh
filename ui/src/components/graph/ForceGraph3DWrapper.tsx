"use client";

/**
 * Thin React wrapper around the vanilla `3d-force-graph` library.
 * Avoids importing `react-force-graph` (the combined bundle) which pulls in
 * aframe-forcegraph-component and crashes with "AFRAME is not defined".
 */

import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyFn = (...args: any[]) => any;

export interface ForceGraph3DWrapperProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  graphData: { nodes: any[]; links: any[] };
  backgroundColor?: string;
  warmupTicks?: number;
  cooldownTicks?: number;
  onEngineStop?: () => void;
  linkColor?: AnyFn | string;
  linkWidth?: AnyFn | number;
  linkDirectionalParticles?: AnyFn | number;
  linkDirectionalParticleWidth?: AnyFn | number;
  linkDirectionalParticleColor?: AnyFn | string;
  nodeThreeObject?: AnyFn;
  nodeThreeObjectExtend?: boolean;
  onNodeClick?: AnyFn;
}

export interface ForceGraph3DHandle {
  zoomToFit(durationMs?: number, padding?: number): void;
}

export const ForceGraph3DWrapper = forwardRef<
  ForceGraph3DHandle,
  ForceGraph3DWrapperProps
>(function ForceGraph3DWrapper(props, ref) {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const prevPropsRef = useRef<Partial<ForceGraph3DWrapperProps>>({});

  useImperativeHandle(ref, () => ({
    zoomToFit(durationMs = 400, padding = 60) {
      fgRef.current?.zoomToFit(durationMs, padding);
    },
  }));

  // Mount: create the 3d-force-graph instance once
  useEffect(() => {
    let cancelled = false;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    import("3d-force-graph").then(({ default: ForceGraph3D }: { default: any }) => {
      if (cancelled || !containerRef.current) return;

      const el = containerRef.current;
      const w = el.clientWidth || 800;
      const h = el.clientHeight || 600;

      // Set dimensions BEFORE mounting so the WebGL renderer never initialises at window size
      const fg = ForceGraph3D({ controlType: "orbit" });
      fg.width(w).height(h).showNavInfo(false);
      applyAll(fg, props);
      prevPropsRef.current = { ...props };
      fg(el);

      fgRef.current = fg;

      // Zoom to fit after warmup — called on the vanilla instance directly
      // so it works regardless of when the parent ref resolves.
      // setTimeout(() => fg.zoomToFit(400, -600), 600);
      setTimeout(() => {
        // .cameraPosition( { x, y, z }, lookAt, transitionDuration )
        fg.cameraPosition(
          { z: 300 }, // Move camera back on Z-axis (larger number = further away/half-zoom)
          { x: 0, y: 0, z: 0 }, // Look at the center of the graph
          400 // Transition duration
        );
      }, 600);

      const ro = new ResizeObserver(([entry]) => {
        fg.width(entry.contentRect.width);
        fg.height(entry.contentRect.height);
      });
      ro.observe(el);

      // Cleanup stored so the outer closure can reach it
      (containerRef.current as HTMLDivElement & { _cleanup?: () => void })._cleanup = () => {
        ro.disconnect();
        fg._destructor?.();
        fgRef.current = null;
      };
    });

    return () => {
      cancelled = true;
      const cleanup = (containerRef.current as (HTMLDivElement & { _cleanup?: () => void }) | null)?._cleanup;
      cleanup?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update only changed props after mount
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    const prev = prevPropsRef.current;
    const p = props;

    if (p.graphData !== prev.graphData) fg.graphData(p.graphData);
    if (p.backgroundColor !== prev.backgroundColor && p.backgroundColor !== undefined) fg.backgroundColor(p.backgroundColor);
    if (p.warmupTicks !== prev.warmupTicks && p.warmupTicks !== undefined) fg.warmupTicks(p.warmupTicks);
    if (p.cooldownTicks !== prev.cooldownTicks && p.cooldownTicks !== undefined) fg.cooldownTicks(p.cooldownTicks);
    if (p.onEngineStop !== prev.onEngineStop && p.onEngineStop !== undefined) fg.onEngineStop(p.onEngineStop);
    if (p.linkColor !== prev.linkColor && p.linkColor !== undefined) fg.linkColor(p.linkColor);
    if (p.linkWidth !== prev.linkWidth && p.linkWidth !== undefined) fg.linkWidth(p.linkWidth);
    if (p.linkDirectionalParticles !== prev.linkDirectionalParticles && p.linkDirectionalParticles !== undefined) fg.linkDirectionalParticles(p.linkDirectionalParticles);
    if (p.linkDirectionalParticleWidth !== prev.linkDirectionalParticleWidth && p.linkDirectionalParticleWidth !== undefined) fg.linkDirectionalParticleWidth(p.linkDirectionalParticleWidth);
    if (p.linkDirectionalParticleColor !== prev.linkDirectionalParticleColor && p.linkDirectionalParticleColor !== undefined) fg.linkDirectionalParticleColor(p.linkDirectionalParticleColor);
    if (p.nodeThreeObject !== prev.nodeThreeObject && p.nodeThreeObject !== undefined) fg.nodeThreeObject(p.nodeThreeObject);
    if (p.nodeThreeObjectExtend !== prev.nodeThreeObjectExtend && p.nodeThreeObjectExtend !== undefined) fg.nodeThreeObjectExtend(p.nodeThreeObjectExtend);
    if (p.onNodeClick !== prev.onNodeClick && p.onNodeClick !== undefined) fg.onNodeClick(p.onNodeClick);

    prevPropsRef.current = { ...p };
  });

  return <div ref={containerRef} style={{ width: "100%", height: "100%", overflow: "hidden" }} />;
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function applyAll(fg: any, props: ForceGraph3DWrapperProps) {
  fg.graphData(props.graphData);
  if (props.backgroundColor !== undefined) fg.backgroundColor(props.backgroundColor);
  if (props.warmupTicks !== undefined) fg.warmupTicks(props.warmupTicks);
  if (props.cooldownTicks !== undefined) fg.cooldownTicks(props.cooldownTicks);
  if (props.onEngineStop !== undefined) fg.onEngineStop(props.onEngineStop);
  if (props.linkColor !== undefined) fg.linkColor(props.linkColor);
  if (props.linkWidth !== undefined) fg.linkWidth(props.linkWidth);
  if (props.linkDirectionalParticles !== undefined) fg.linkDirectionalParticles(props.linkDirectionalParticles);
  if (props.linkDirectionalParticleWidth !== undefined) fg.linkDirectionalParticleWidth(props.linkDirectionalParticleWidth);
  if (props.linkDirectionalParticleColor !== undefined) fg.linkDirectionalParticleColor(props.linkDirectionalParticleColor);
  if (props.nodeThreeObject !== undefined) fg.nodeThreeObject(props.nodeThreeObject);
  if (props.nodeThreeObjectExtend !== undefined) fg.nodeThreeObjectExtend(props.nodeThreeObjectExtend);
  if (props.onNodeClick !== undefined) fg.onNodeClick(props.onNodeClick);
}

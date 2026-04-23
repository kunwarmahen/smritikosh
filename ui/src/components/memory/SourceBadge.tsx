"use client";

import {
  Code2,
  PenLine,
  Sparkles,
  Activity,
  Zap,
  Layers,
  Globe,
  Wrench,
  GitMerge,
  Mic,
  Volume2,
  ImageIcon,
  FileText,
} from "lucide-react";
import { clsx } from "clsx";

interface SourceConfig {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  className: string;
}

const SOURCE_CONFIG: Record<string, SourceConfig> = {
  api_explicit:         { label: "API",        icon: Code2,      className: "bg-zinc-800 text-zinc-400 border-zinc-700/60" },
  ui_manual:            { label: "Manual",     icon: PenLine,    className: "bg-blue-500/10 text-blue-400 border-blue-500/20" },
  passive_distillation: { label: "Distilled",  icon: Sparkles,   className: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
  passive_streaming:    { label: "Streaming",  icon: Activity,   className: "bg-orange-500/10 text-orange-400 border-orange-500/20" },
  trigger_word:         { label: "Triggered",  icon: Zap,        className: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20" },
  sdk_middleware:       { label: "SDK",        icon: Layers,     className: "bg-sky-500/10 text-sky-400 border-sky-500/20" },
  webhook_ingest:       { label: "Webhook",    icon: Globe,      className: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20" },
  tool_use:             { label: "Tool",       icon: Wrench,     className: "bg-purple-500/10 text-purple-400 border-purple-500/20" },
  cross_system:         { label: "Synthesized",icon: GitMerge,   className: "bg-teal-500/10 text-teal-400 border-teal-500/20" },
  media_voice:          { label: "Voice",      icon: Mic,        className: "bg-rose-500/10 text-rose-400 border-rose-500/20" },
  media_audio:          { label: "Audio",      icon: Volume2,    className: "bg-pink-500/10 text-pink-400 border-pink-500/20" },
  media_image:          { label: "Image",      icon: ImageIcon,  className: "bg-violet-500/10 text-violet-400 border-violet-500/20" },
  media_document:       { label: "Doc",        icon: FileText,   className: "bg-slate-500/10 text-slate-400 border-slate-500/20" },
};

interface Props {
  sourceType: string;
  hideLabel?: boolean;
  className?: string;
}

export function SourceBadge({ sourceType, hideLabel = false, className }: Props) {
  const config = SOURCE_CONFIG[sourceType] ?? SOURCE_CONFIG.api_explicit;
  const Icon = config.icon;

  return (
    <span className={clsx("badge border", config.className, className)}>
      <Icon className="w-3 h-3 flex-shrink-0" />
      {!hideLabel && config.label}
    </span>
  );
}

export function isAutoExtracted(sourceType: string | undefined): boolean {
  if (!sourceType) return false;
  return !["api_explicit", "ui_manual"].includes(sourceType);
}

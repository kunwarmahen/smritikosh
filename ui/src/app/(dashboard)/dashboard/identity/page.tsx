"use client";

import { useState } from "react";
import { Network } from "lucide-react";
import { IdentityProfilePanel } from "@/components/identity/IdentityProfile";
import { IdentityFactGraph } from "@/components/identity/IdentityFactGraph";

export default function IdentityPage() {
  const [showGraph, setShowGraph] = useState(false);

  if (showGraph) {
    return (
      <div className="-mx-8 -mt-8 h-screen">
        <IdentityFactGraph onClose={() => setShowGraph(false)} />
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-start justify-between mb-6 gap-4">
        <div>
          <h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100 tracking-tight">Identity Model</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Your synthesised personality profile, beliefs, and core dimensions.
          </p>
        </div>
        <button
          onClick={() => setShowGraph(true)}
          className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg border transition-colors flex-shrink-0 bg-zinc-100 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"
        >
          <Network className="w-3.5 h-3.5" />
          Fact graph
        </button>
      </div>

      <IdentityProfilePanel />
    </div>
  );
}

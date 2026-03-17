"use client";

import { useState } from "react";
import { Network } from "lucide-react";
import { IdentityProfilePanel } from "@/components/identity/IdentityProfile";
import { IdentityFactGraph } from "@/components/identity/IdentityFactGraph";

export default function IdentityPage() {
  const [showGraph, setShowGraph] = useState(false);

  return (
    <div>
      <div className="flex items-start justify-between mb-6 gap-4">
        <div>
          <h1 className="text-base font-semibold text-zinc-100 tracking-tight">Identity Model</h1>
          <p className="text-sm text-zinc-500 mt-1">
            Your synthesised personality profile, beliefs, and core dimensions.
          </p>
        </div>
        <button
          onClick={() => setShowGraph((v) => !v)}
          className={`flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg border transition-colors flex-shrink-0 ${
            showGraph
              ? "bg-violet-600/20 border-violet-500/40 text-violet-300"
              : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-zinc-200"
          }`}
        >
          <Network className="w-3.5 h-3.5" />
          {showGraph ? "Hide graph" : "Fact graph"}
        </button>
      </div>

      {showGraph ? (
        <div className="mb-6">
          <IdentityFactGraph />
        </div>
      ) : null}

      <IdentityProfilePanel />
    </div>
  );
}

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
          <h1 className="text-xl font-semibold text-slate-100">Identity Model</h1>
          <p className="text-sm text-slate-500 mt-1">
            Your synthesised personality profile, beliefs, and core dimensions.
          </p>
        </div>
        <button
          onClick={() => setShowGraph((v) => !v)}
          className={`flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-lg border transition-colors flex-shrink-0 ${
            showGraph
              ? "bg-violet-600/20 border-violet-500/40 text-violet-300"
              : "bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200"
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

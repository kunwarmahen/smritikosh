"use client";

import { useState } from "react";
import { X, Loader2 } from "lucide-react";
import { useCreateProcedure } from "@/hooks/useProcedures";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function NewProcedureDrawer({ open, onClose }: Props) {
  const create = useCreateProcedure();
  const [trigger, setTrigger]       = useState("");
  const [instruction, setInstruction] = useState("");
  const [category, setCategory]     = useState("general");
  const [priority, setPriority]     = useState(5);
  const [error, setError]           = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (!trigger.trim() || !instruction.trim()) {
      setError("Trigger and instruction are required.");
      return;
    }
    try {
      await create.mutateAsync({ trigger, instruction, category, priority });
      setTrigger("");
      setInstruction("");
      setCategory("general");
      setPriority(5);
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create procedure.");
    }
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-zinc-950/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="relative z-10 w-full sm:max-w-lg bg-zinc-900 border border-zinc-700
                      rounded-t-2xl sm:rounded-2xl shadow-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-base font-semibold text-zinc-100">New Procedure</h2>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300">
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="label">Trigger</label>
            <input
              type="text"
              className="input mt-1"
              placeholder="when user asks about diet…"
              value={trigger}
              onChange={(e) => setTrigger(e.target.value)}
            />
            <p className="text-xs text-zinc-600 mt-1">
              A short phrase or condition that activates this procedure.
            </p>
          </div>

          <div>
            <label className="label">Instruction</label>
            <textarea
              className="input mt-1 h-24 resize-none"
              placeholder="Always include their dietary restrictions and preferred cuisine…"
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label">Category</label>
              <input
                type="text"
                className="input mt-1"
                placeholder="general"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              />
            </div>
            <div>
              <label className="label">Priority (1–10)</label>
              <input
                type="number"
                className="input mt-1"
                min={1}
                max={10}
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value))}
              />
            </div>
          </div>

          {error && (
            <p className="text-xs text-rose-400">{error}</p>
          )}

          <div className="flex gap-3 pt-1">
            <button type="button" onClick={onClose} className="btn-secondary flex-1">
              Cancel
            </button>
            <button
              type="submit"
              disabled={create.isPending}
              className="btn-primary flex-1 flex items-center justify-center gap-2"
            >
              {create.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

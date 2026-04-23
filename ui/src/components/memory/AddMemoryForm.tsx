"use client";

import { useState } from "react";
import { X, PenLine, Loader2, Check } from "lucide-react";
import { useStoreFact } from "@/hooks/useMemory";

const FACT_CATEGORIES = [
  { value: "identity",     label: "Identity" },
  { value: "location",     label: "Location" },
  { value: "role",         label: "Role" },
  { value: "skill",        label: "Skill" },
  { value: "education",    label: "Education" },
  { value: "project",      label: "Project" },
  { value: "goal",         label: "Goal" },
  { value: "interest",     label: "Interest" },
  { value: "hobby",        label: "Hobby" },
  { value: "habit",        label: "Habit" },
  { value: "preference",   label: "Preference" },
  { value: "personality",  label: "Personality" },
  { value: "relationship", label: "Relationship" },
  { value: "pet",          label: "Pet" },
  { value: "health",       label: "Health" },
  { value: "diet",         label: "Diet" },
  { value: "belief",       label: "Belief" },
  { value: "value",        label: "Value" },
  { value: "religion",     label: "Religion" },
  { value: "finance",      label: "Finance" },
  { value: "lifestyle",    label: "Lifestyle" },
  { value: "event",        label: "Event" },
  { value: "tool",         label: "Tool" },
];

interface Props {
  onClose: () => void;
}

export function AddMemoryForm({ onClose }: Props) {
  const [category, setCategory] = useState("skill");
  const [key, setKey] = useState("");
  const [value, setValue] = useState("");
  const [note, setNote] = useState("");
  const [saved, setSaved] = useState(false);

  const storeFact = useStoreFact();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!key.trim() || !value.trim()) return;

    await storeFact.mutateAsync({
      category,
      key: key.trim().toLowerCase().replace(/\s+/g, "_"),
      value: value.trim(),
      note: note.trim() || undefined,
    });

    setSaved(true);
    setTimeout(() => {
      setSaved(false);
      setKey("");
      setValue("");
      setNote("");
    }, 1500);
  }

  return (
    <div
      className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-md shadow-2xl animate-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 bg-blue-500/15 rounded-md flex items-center justify-center">
              <PenLine className="w-3.5 h-3.5 text-blue-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-100">Add a memory</h2>
              <p className="text-xs text-zinc-500">Stored as a verified fact</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded text-zinc-600
                       hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          {/* Category */}
          <div>
            <label className="label">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="input"
            >
              {FACT_CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>{c.label}</option>
              ))}
            </select>
          </div>

          {/* Key + Value side by side */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label">
                Key
                <span className="ml-1 text-zinc-600 normal-case font-normal">(e.g. editor)</span>
              </label>
              <input
                type="text"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder="editor"
                className="input"
                required
              />
            </div>
            <div>
              <label className="label">Value</label>
              <input
                type="text"
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder="neovim"
                className="input"
                required
              />
            </div>
          </div>

          {/* Note */}
          <div>
            <label className="label">
              Note
              <span className="ml-1 text-zinc-600 normal-case font-normal">(optional context)</span>
            </label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Add context about this fact…"
              rows={2}
              className="input resize-none"
            />
          </div>

          {storeFact.isError && (
            <p className="text-xs text-rose-400">
              {storeFact.error?.message ?? "Failed to save — try again."}
            </p>
          )}

          {/* Actions */}
          <div className="flex items-center justify-end gap-2 pt-1">
            <button type="button" onClick={onClose} className="btn-secondary">
              Cancel
            </button>
            <button
              type="submit"
              disabled={storeFact.isPending || saved || !key.trim() || !value.trim()}
              className="btn-primary"
            >
              {saved ? (
                <><Check className="w-4 h-4" /> Saved</>
              ) : storeFact.isPending ? (
                <><Loader2 className="w-4 h-4 animate-spin" /> Saving…</>
              ) : "Save memory"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

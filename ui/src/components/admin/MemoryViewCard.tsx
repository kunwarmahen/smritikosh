"use client";

/**
 * What an agent's prompt is given about a user, and what keeps a memory
 * from ageing -- for one user, or for everyone ("*"). Each switch can
 * also follow the default: a user's row overrides only what it sets.
 */

import { Loader2 } from "lucide-react";
import { clsx } from "clsx";
import { useMemoryView, useSetMemoryView } from "@/hooks/useAdmin";
import type { MemoryViewOverrides } from "@/lib/api-client";

type Switch = "show_facts" | "show_beliefs" | "show_memories";

const SWITCHES: { key: Switch; label: string; hint: string }[] = [
  { key: "show_facts", label: "Facts",
    hint: "Who they are, where they live, what they prefer, and the facts that match the question." },
  { key: "show_beliefs", label: "Beliefs",
    hint: "Patterns found across their memories, given as guesses." },
  { key: "show_memories", label: "Memories",
    hint: "What they told the agent that no fact covers yet." },
];

export function MemoryViewCard({ userId }: { userId: string }) {
  const everyone = userId === "*";
  const { data, isLoading, isError } = useMemoryView(userId);
  const save = useSetMemoryView();

  function set(change: Partial<MemoryViewOverrides>) {
    if (!data) return;
    save.mutate({ userId, body: { ...data.overrides, ...change } });
  }

  // Everyone's row has nothing above it but the built-in default, so
  // "inherit" there reads "default (on)".
  const inherited = (key: keyof MemoryViewOverrides) => {
    if (!data) return "";
    const value = data.effective[key];
    const shown = typeof value === "boolean" ? (value ? "on" : "off") : value;
    return data.source[key] === "user" ? "" : `${everyone ? "default" : "everyone's"}: ${shown}`;
  };

  return (
    <div className="card">
      <p className="section-heading mb-1">In the agent&apos;s prompt</p>
      <p className="text-xs text-zinc-600 mb-3">
        {everyone
          ? "For everyone who has no setting of their own."
          : "For this user. Anything left on the default follows everyone's setting."}
      </p>
      {isLoading && <Loader2 className="w-4 h-4 animate-spin text-zinc-500" />}
      {isError && <p className="text-xs text-rose-400">Couldn&apos;t load these settings.</p>}
      {data && (
        <div className="space-y-0">
          {SWITCHES.map(({ key, label, hint }) => (
            <Row key={key} label={label} hint={hint} note={inherited(key)}>
              <Choice
                value={data.overrides[key] === null ? "inherit" : data.overrides[key] ? "on" : "off"}
                options={[["inherit", "Default"], ["on", "On"], ["off", "Off"]]}
                disabled={save.isPending}
                onChange={(v) => set({ [key]: v === "inherit" ? null : v === "on" })}
              />
            </Row>
          ))}
          <Row label="What keeps a memory"
               hint="Used: only an answer that leaned on it keeps it from ageing. Shown: appearing in a prompt keeps it too, so nothing shown is ever pruned."
               note={inherited("keep_by")} last>
            <Choice
              value={data.overrides.keep_by ?? "inherit"}
              options={[["inherit", "Default"], ["used", "Used"], ["shown", "Shown"]]}
              disabled={save.isPending}
              onChange={(v) => set({ keep_by: v === "inherit" ? null : (v as "used" | "shown") })}
            />
          </Row>
        </div>
      )}
      {save.isError && <p className="text-xs text-rose-400 mt-2">Not saved: {String(save.error)}</p>}
    </div>
  );
}

function Row({ label, hint, note, last = false, children }: {
  label: string; hint: string; note: string; last?: boolean; children: React.ReactNode;
}) {
  return (
    <div className={clsx("flex items-start justify-between gap-4 py-3",
                         !last && "border-b border-zinc-200 dark:border-zinc-800/60")}>
      <div className="min-w-0">
        <p className="text-sm text-zinc-700 dark:text-zinc-300">{label}</p>
        <p className="text-xs text-zinc-600 mt-0.5">{hint}</p>
        {note && <p className="text-xs text-zinc-500 mt-0.5 italic">{note}</p>}
      </div>
      {children}
    </div>
  );
}

function Choice({ value, options, disabled, onChange }: {
  value: string; options: [string, string][]; disabled: boolean; onChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-shrink-0 rounded-lg border border-zinc-200 dark:border-zinc-700 overflow-hidden">
      {options.map(([v, label]) => (
        <button key={v} disabled={disabled} onClick={() => onChange(v)}
                className={clsx("text-xs px-2.5 py-1.5 transition-colors",
                  value === v
                    ? "bg-violet-500/15 text-violet-500"
                    : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200")}>
          {label}
        </button>
      ))}
    </div>
  );
}

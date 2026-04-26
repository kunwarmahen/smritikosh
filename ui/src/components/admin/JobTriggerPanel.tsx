"use client";

import { useState } from "react";
import { Loader2, Play, CheckCircle2, XCircle } from "lucide-react";
import { clsx } from "clsx";
import {
  useAdminConsolidate,
  useAdminPrune,
  useAdminCluster,
  useAdminMineBeliefs,
} from "@/hooks/useAdmin";

type JobStatus = "idle" | "running" | "ok" | "error";

function JobRow({
  title,
  description,
  status,
  onRun,
}: {
  title: string;
  description: string;
  status: JobStatus;
  onRun: () => void;
}) {
  return (
    <div className="flex items-center gap-4 py-3.5 border-b border-zinc-200 dark:border-zinc-800/60 last:border-0">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">{title}</p>
        <p className="text-xs text-zinc-600 mt-0.5">{description}</p>
      </div>
      <div className="flex items-center gap-2.5 flex-shrink-0">
        {status === "ok" && (
          <span className="flex items-center gap-1 text-xs text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" /> done
          </span>
        )}
        {status === "error" && (
          <span className="flex items-center gap-1 text-xs text-rose-400">
            <XCircle className="w-3.5 h-3.5" /> failed
          </span>
        )}
        <button
          onClick={onRun}
          disabled={status === "running"}
          className={clsx(
            "w-7 h-7 flex items-center justify-center rounded-lg transition-colors",
            status === "running"
              ? "bg-zinc-200 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-600 cursor-not-allowed"
              : "bg-violet-600 hover:bg-violet-500 text-white",
          )}
          title={`Run ${title}`}
        >
          {status === "running"
            ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
            : <Play    className="w-3 h-3 ml-0.5" />}
        </button>
      </div>
    </div>
  );
}

export function JobTriggerPanel({ userId }: { userId: string }) {
  const consolidate = useAdminConsolidate();
  const prune       = useAdminPrune();
  const cluster     = useAdminCluster();
  const mineBeliefs = useAdminMineBeliefs();

  type JobKey = "consolidate" | "prune" | "cluster" | "mine";
  const [statuses, setStatuses] = useState<Record<JobKey, JobStatus>>({
    consolidate: "idle",
    prune:       "idle",
    cluster:     "idle",
    mine:        "idle",
  });

  async function runJob(key: JobKey, fn: () => Promise<unknown>) {
    setStatuses((s) => ({ ...s, [key]: "running" }));
    try {
      await fn();
      setStatuses((s) => ({ ...s, [key]: "ok" }));
    } catch {
      setStatuses((s) => ({ ...s, [key]: "error" }));
    }
  }

  const jobs: { key: JobKey; title: string; description: string; fn: () => Promise<unknown> }[] = [
    {
      key: "consolidate",
      title: "Consolidate",
      description: "Merge raw events into summaries and extract facts into Neo4j.",
      fn: () => consolidate.mutateAsync({ userId }),
    },
    {
      key: "prune",
      title: "Prune",
      description: "Remove low-importance memories past retention threshold.",
      fn: () => prune.mutateAsync({ userId }),
    },
    {
      key: "cluster",
      title: "Cluster",
      description: "Group memories by topic using semantic embeddings.",
      fn: () => cluster.mutateAsync({ userId }),
    },
    {
      key: "mine",
      title: "Mine beliefs",
      description: "Infer higher-order beliefs and values from memory patterns.",
      fn: () => mineBeliefs.mutateAsync({ userId }),
    },
  ];

  return (
    <div className="card p-0">
      <div className="px-4 pt-3.5 pb-1">
        <p className="section-heading">Pipeline jobs · {userId}</p>
      </div>
      <div className="px-4">
        {jobs.map(({ key, title, description, fn }) => (
          <JobRow
            key={key}
            title={title}
            description={description}
            status={statuses[key]}
            onRun={() => runJob(key, fn)}
          />
        ))}
      </div>
    </div>
  );
}

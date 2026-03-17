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

interface JobCardProps {
  title: string;
  description: string;
  status: JobStatus;
  onRun: () => void;
}

function JobCard({ title, description, status, onRun }: JobCardProps) {
  return (
    <div className="card flex items-center gap-4">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-200">{title}</p>
        <p className="text-xs text-slate-500 mt-0.5">{description}</p>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        {status === "ok"    && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
        {status === "error" && <XCircle className="w-4 h-4 text-rose-400" />}
        <button
          onClick={onRun}
          disabled={status === "running"}
          className={clsx(
            "w-8 h-8 flex items-center justify-center rounded-lg transition-colors",
            status === "running"
              ? "bg-slate-800 text-slate-500 cursor-not-allowed"
              : "bg-violet-600 hover:bg-violet-500 text-white",
          )}
          title={`Run ${title}`}
        >
          {status === "running"
            ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
            : <Play className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );
}

export function JobTriggerPanel({ userId }: { userId: string }) {
  const consolidate  = useAdminConsolidate();
  const prune        = useAdminPrune();
  const cluster      = useAdminCluster();
  const mineBeliefs  = useAdminMineBeliefs();

  type JobKey = "consolidate" | "prune" | "cluster" | "mine";
  const [statuses, setStatuses] = useState<Record<JobKey, JobStatus>>({
    consolidate: "idle",
    prune: "idle",
    cluster: "idle",
    mine: "idle",
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
      description: "Merge semantically similar memories and extract facts.",
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
      title: "Mine Beliefs",
      description: "Infer higher-order beliefs and values from memory patterns.",
      fn: () => mineBeliefs.mutateAsync({ userId }),
    },
  ];

  return (
    <div className="space-y-3">
      {jobs.map(({ key, title, description, fn }) => (
        <JobCard
          key={key}
          title={title}
          description={description}
          status={statuses[key]}
          onRun={() => runJob(key, fn)}
        />
      ))}
    </div>
  );
}

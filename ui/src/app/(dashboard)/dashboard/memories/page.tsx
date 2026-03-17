import { MemoryTimeline } from "@/components/memory/MemoryTimeline";

export default function MemoriesPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">Memories</h1>
        <p className="text-sm text-slate-500 mt-1">
          Your stored memories, searchable and manageable.
        </p>
      </div>
      <MemoryTimeline />
    </div>
  );
}

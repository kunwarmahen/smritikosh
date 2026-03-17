import { MemoryTimeline } from "@/components/memory/MemoryTimeline";

export default function MemoriesPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100 tracking-tight">Memories</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Your stored memories, searchable and manageable.
        </p>
      </div>
      <MemoryTimeline />
    </div>
  );
}

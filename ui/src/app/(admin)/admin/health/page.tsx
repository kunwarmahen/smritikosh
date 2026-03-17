import { HealthPanel } from "@/components/admin/HealthPanel";

export default function AdminHealthPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100 tracking-tight">System Health</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Live status of all backend services.
        </p>
      </div>
      <div className="max-w-md">
        <HealthPanel />
      </div>
    </div>
  );
}

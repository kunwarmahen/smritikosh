import { HealthPanel } from "@/components/admin/HealthPanel";

export default function AdminHealthPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">System Health</h1>
        <p className="text-sm text-slate-500 mt-1">
          Live status of all backend services.
        </p>
      </div>
      <div className="max-w-md">
        <HealthPanel />
      </div>
    </div>
  );
}

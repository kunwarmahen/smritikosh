import { AuditStatsBar } from "@/components/audit/AuditStatsBar";
import { AuditTimeline } from "@/components/audit/AuditTimeline";

export default function AuditPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">Audit Trail</h1>
        <p className="text-sm text-slate-500 mt-1">
          Every memory operation recorded — filter by event type.
        </p>
      </div>
      <AuditStatsBar />
      <AuditTimeline />
    </div>
  );
}

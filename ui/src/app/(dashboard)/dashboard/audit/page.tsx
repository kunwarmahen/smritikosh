import { AuditStatsBar } from "@/components/audit/AuditStatsBar";
import { AuditTimeline } from "@/components/audit/AuditTimeline";

export default function AuditPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100 tracking-tight">Audit Trail</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Every memory operation recorded — filter by event type.
        </p>
      </div>
      <AuditStatsBar />
      <AuditTimeline />
    </div>
  );
}

"use client";

import { useState } from "react";
import { AuditStatsBar } from "@/components/audit/AuditStatsBar";
import { AuditTimeline } from "@/components/audit/AuditTimeline";

export default function AdminAuditPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100 tracking-tight">Audit Log</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Full system-wide audit trail for all pipeline events.
        </p>
      </div>
      <AuditStatsBar />
      <AuditTimeline />
    </div>
  );
}

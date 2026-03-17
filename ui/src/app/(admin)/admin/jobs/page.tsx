"use client";

import { useState } from "react";
import { JobTriggerPanel } from "@/components/admin/JobTriggerPanel";
import { useSession } from "next-auth/react";

export default function AdminJobsPage() {
  const { data: session } = useSession();
  const [userId, setUserId] = useState(session?.user?.id ?? "");

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">Pipeline Jobs</h1>
        <p className="text-sm text-slate-500 mt-1">
          Manually trigger memory pipeline jobs for any user.
        </p>
      </div>

      <div className="mb-4 max-w-sm">
        <label className="label mb-1 block">Target user ID</label>
        <input
          type="text"
          className="input"
          placeholder="alice"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
        />
      </div>

      {userId ? (
        <JobTriggerPanel userId={userId} />
      ) : (
        <p className="text-sm text-slate-600">Enter a user ID to enable job triggers.</p>
      )}
    </div>
  );
}

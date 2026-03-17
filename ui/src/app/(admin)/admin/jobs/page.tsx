"use client";

import { useState } from "react";
import { JobTriggerPanel } from "@/components/admin/JobTriggerPanel";
import { useSession } from "next-auth/react";

export default function AdminJobsPage() {
  const [userId, setUserId] = useState("");

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100 tracking-tight">Pipeline Jobs</h1>
        <p className="text-xs text-zinc-500 mt-1">
          Manually trigger memory pipeline jobs for any user.
        </p>
      </div>

      <div className="mb-5 max-w-xs">
        <label className="label">Target user</label>
        <input
          type="text"
          className="input"
          placeholder="alice"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          autoFocus
        />
      </div>

      {userId
        ? <JobTriggerPanel userId={userId} />
        : <p className="text-xs text-zinc-700">Enter a user ID above to enable job triggers.</p>
      }
    </div>
  );
}

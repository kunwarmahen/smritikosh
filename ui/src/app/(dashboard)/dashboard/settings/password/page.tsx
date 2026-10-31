"use client";

import { useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { useSession } from "next-auth/react";
import { createApiClient } from "@/lib/api-client";

// Change your own password. The current one is asked for even while signed
// in; API keys and other open sign-ins keep working.
export default function PasswordPage() {
  const { data: session } = useSession();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setDone(false);
    if (next.length < 8) { setError("The new password needs 8 characters or more."); return; }
    if (next !== again) { setError("The two new passwords are not the same."); return; }
    setBusy(true);
    try {
      await createApiClient(session?.accessToken).changePassword({
        current_password: current, new_password: next,
      });
      setDone(true);
      setCurrent(""); setNext(""); setAgain("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "The password was not changed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-md">
      <div className="mb-6">
        <h1 className="page-title">Password</h1>
        <p className="text-xs text-zinc-500 mt-1">
          The password you sign in here with. Your API keys keep working.
        </p>
      </div>
      <form onSubmit={submit} className="card space-y-3">
        <div>
          <label className="label" htmlFor="current">Current password</label>
          <input id="current" type="password" className="input" autoComplete="current-password"
                 value={current} onChange={(e) => setCurrent(e.target.value)} required />
        </div>
        <div>
          <label className="label" htmlFor="next">New password</label>
          <input id="next" type="password" className="input" autoComplete="new-password"
                 value={next} onChange={(e) => setNext(e.target.value)} required />
        </div>
        <div>
          <label className="label" htmlFor="again">New password again</label>
          <input id="again" type="password" className="input" autoComplete="new-password"
                 value={again} onChange={(e) => setAgain(e.target.value)} required />
        </div>
        {error && <p className="text-xs text-rose-400">{error}</p>}
        {done && (
          <p className="text-xs text-emerald-500 flex items-center gap-1">
            <Check className="w-3.5 h-3.5" /> Changed. Use it the next time you sign in.
          </p>
        )}
        <button type="submit" className="btn-primary w-full justify-center" disabled={busy}>
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : "Change password"}
        </button>
      </form>
    </div>
  );
}

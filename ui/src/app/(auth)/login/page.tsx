"use client";

import { Suspense } from "react";
import { signIn } from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Brain, Loader2, Eye, EyeOff } from "lucide-react";

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get("callbackUrl") ?? "/dashboard/memories";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    const result = await signIn("credentials", { username, password, redirect: false });
    setLoading(false);
    if (result?.error) {
      setError("Incorrect username or password.");
    } else {
      router.push(callbackUrl);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="label" htmlFor="username">Username</label>
        <input
          id="username"
          type="text"
          className="input"
          placeholder="alice"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          autoFocus
          required
        />
      </div>

      <div>
        <label className="label" htmlFor="password">Password</label>
        <div className="relative">
          <input
            id="password"
            type={showPassword ? "text" : "password"}
            className="input pr-10"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {error && (
        <p className="text-rose-400 text-xs bg-rose-500/8 border border-rose-500/20 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={loading || !username || !password}
        className="btn-primary w-full justify-center mt-1"
      >
        {loading ? (
          <><Loader2 className="w-4 h-4 animate-spin" /> Signing in…</>
        ) : (
          "Continue"
        )}
      </button>
    </form>
  );
}

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-50 dark:bg-zinc-950 px-4">
      {/* Ambient glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                        w-[600px] h-[600px] rounded-full"
          style={{
            background: "radial-gradient(circle, rgba(109,40,217,0.06) 0%, transparent 70%)",
          }}
        />
      </div>

      <div className="relative w-full max-w-[340px]">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12
                          bg-violet-600 rounded-2xl mb-5 shadow-lg shadow-violet-950">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-xl font-semibold text-zinc-900 dark:text-zinc-100 tracking-tight">Smritikosh</h1>
          <p className="text-zinc-600 text-sm mt-1">Memory Dashboard</p>
        </div>

        {/* Form card */}
        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6">
          <Suspense fallback={
            <div className="flex justify-center py-10">
              <Loader2 className="w-5 h-5 animate-spin text-zinc-600" />
            </div>
          }>
            <LoginForm />
          </Suspense>
        </div>

        <p className="text-center text-zinc-700 text-xs mt-5">
          Contact your administrator for access
        </p>
      </div>
    </div>
  );
}

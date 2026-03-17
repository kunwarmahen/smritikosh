"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut, useSession } from "next-auth/react";
import {
  Brain,
  Users,
  Cpu,
  HeartPulse,
  ActivitySquare,
  ArrowLeft,
  LogOut,
  ChevronRight,
} from "lucide-react";
import { clsx } from "clsx";

const ADMIN_NAV = [
  { href: "/admin/users",  icon: Users,           label: "Users" },
  { href: "/admin/jobs",   icon: Cpu,             label: "Background jobs" },
  { href: "/admin/audit",  icon: ActivitySquare,  label: "Audit log" },
  { href: "/admin/health", icon: HeartPulse,      label: "Health" },
];

export function AdminShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data: session } = useSession();

  return (
    <div className="flex min-h-screen">
      <aside className="fixed left-0 top-0 h-screen w-[240px] bg-slate-900 border-r border-slate-800
                        flex flex-col z-20">
        {/* Logo */}
        <div className="px-4 py-5 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-amber-600/20 border border-amber-500/30 rounded-lg
                            flex items-center justify-center flex-shrink-0">
              <Brain className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-100">Admin Panel</p>
              <p className="text-xs text-slate-500">Smritikosh</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
          {/* Back to user dashboard */}
          <Link
            href="/dashboard/memories"
            className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium
                       text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            My dashboard
          </Link>

          <div className="pt-1 pb-1 px-3">
            <p className="text-xs font-medium text-slate-600 uppercase tracking-wider">Admin</p>
          </div>

          {ADMIN_NAV.map(({ href, icon: Icon, label }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                  active
                    ? "bg-amber-600/10 text-amber-300 border border-amber-500/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800",
                )}
              >
                <Icon className="w-4 h-4 flex-shrink-0" />
                {label}
                {active && <ChevronRight className="w-3 h-3 ml-auto text-amber-400/60" />}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-slate-800 p-3">
          <div className="flex items-center gap-3 px-2 py-2 mb-1">
            <div className="w-7 h-7 bg-amber-600/20 border border-amber-500/30 rounded-full
                            flex items-center justify-center text-amber-400 text-xs font-bold flex-shrink-0">
              {session?.user?.id?.[0]?.toUpperCase() ?? "A"}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-slate-200 truncate">{session?.user?.id}</p>
              <p className="text-xs text-amber-500">Administrator</p>
            </div>
          </div>
          <button
            onClick={() => signOut({ callbackUrl: "/login" })}
            className="flex items-center gap-2 w-full px-3 py-1.5 text-xs text-slate-500
                       hover:text-slate-300 hover:bg-slate-800 rounded-lg transition-colors"
          >
            <LogOut className="w-3.5 h-3.5" />
            Sign out
          </button>
        </div>
      </aside>

      <main className="ml-[240px] flex-1 min-h-screen">
        {children}
      </main>
    </div>
  );
}

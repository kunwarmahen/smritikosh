"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut, useSession } from "next-auth/react";
import {
  Brain,
  Clock,
  User2,
  Grid3X3,
  ActivitySquare,
  Zap,
  ShieldCheck,
  LogOut,
  ChevronRight,
} from "lucide-react";
import { clsx } from "clsx";

const USER_NAV = [
  { href: "/dashboard/memories",   icon: Clock,           label: "Memories" },
  { href: "/dashboard/identity",   icon: User2,           label: "Identity" },
  { href: "/dashboard/clusters",   icon: Grid3X3,         label: "Clusters" },
  { href: "/dashboard/audit",      icon: ActivitySquare,  label: "Audit trail" },
  { href: "/dashboard/procedures", icon: Zap,             label: "Procedures" },
];

const ADMIN_NAV = [
  { href: "/admin/users",   icon: ShieldCheck, label: "Admin panel" },
];

export function UserShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data: session } = useSession();
  const isAdmin = session?.user?.role === "admin";

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-screen w-[240px] bg-slate-900 border-r border-slate-800
                        flex flex-col z-20">
        {/* Logo */}
        <div className="px-4 py-5 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-violet-600/20 border border-violet-500/30 rounded-lg
                            flex items-center justify-center flex-shrink-0">
              <Brain className="w-4 h-4 text-violet-400" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-100">Smritikosh</p>
              <p className="text-xs text-slate-500">स्मृतिकोश</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
          {USER_NAV.map(({ href, icon: Icon, label }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                  active
                    ? "bg-violet-600/20 text-violet-300 border border-violet-500/20"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800",
                )}
              >
                <Icon className="w-4 h-4 flex-shrink-0" />
                {label}
                {active && <ChevronRight className="w-3 h-3 ml-auto text-violet-400/60" />}
              </Link>
            );
          })}

          {isAdmin && (
            <>
              <div className="pt-3 pb-1 px-3">
                <p className="text-xs font-medium text-slate-600 uppercase tracking-wider">Admin</p>
              </div>
              {ADMIN_NAV.map(({ href, icon: Icon, label }) => (
                <Link
                  key={href}
                  href={href}
                  className={clsx(
                    "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                    pathname.startsWith("/admin")
                      ? "bg-amber-600/10 text-amber-300 border border-amber-500/20"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800",
                  )}
                >
                  <Icon className="w-4 h-4 flex-shrink-0" />
                  {label}
                </Link>
              ))}
            </>
          )}
        </nav>

        {/* User footer */}
        <div className="border-t border-slate-800 p-3">
          <div className="flex items-center gap-3 px-2 py-2 mb-1">
            <div className="w-7 h-7 bg-violet-600/20 border border-violet-500/30 rounded-full
                            flex items-center justify-center text-violet-400 text-xs font-bold flex-shrink-0">
              {session?.user?.id?.[0]?.toUpperCase() ?? "?"}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-slate-200 truncate">{session?.user?.id}</p>
              <p className="text-xs text-slate-500 capitalize">{session?.user?.role}</p>
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

      {/* Main content */}
      <main className="ml-[240px] flex-1 min-h-screen">
        {children}
      </main>
    </div>
  );
}

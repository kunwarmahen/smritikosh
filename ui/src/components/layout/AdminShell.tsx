"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut, useSession } from "next-auth/react";
import {
  Brain,
  Users,
  Cpu,
  HeartPulse,
  ScrollText,
  ArrowLeft,
  LogOut,
} from "lucide-react";
import { clsx } from "clsx";

const ADMIN_NAV = [
  { href: "/admin/users",  icon: Users,      label: "Users" },
  { href: "/admin/jobs",   icon: Cpu,        label: "Jobs" },
  { href: "/admin/audit",  icon: ScrollText, label: "Audit log" },
  { href: "/admin/health", icon: HeartPulse, label: "Health" },
];

export function AdminShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data: session } = useSession();
  const username = session?.user?.id ?? "";

  return (
    <div className="flex min-h-screen">
      <aside
        className="fixed left-0 top-0 h-screen flex flex-col z-20"
        style={{ width: "var(--sidebar-width)" }}
      >
        <div className="flex flex-col h-full bg-zinc-950 border-r border-zinc-800/80">

          {/* Logo */}
          <div className="px-5 py-4 border-b border-zinc-800/80">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 bg-amber-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <Brain className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className="text-sm font-semibold text-zinc-100 leading-none">Admin</p>
                <p className="text-[10px] text-zinc-600 mt-0.5">Smritikosh</p>
              </div>
            </div>
          </div>

          {/* Nav */}
          <nav className="flex-1 px-3 py-3 space-y-0.5 overflow-y-auto">
            {/* Back link */}
            <Link
              href="/dashboard/memories"
              className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm
                         text-zinc-600 hover:text-zinc-400 hover:bg-zinc-900 transition-all duration-100 mb-2"
            >
              <ArrowLeft className="w-4 h-4" />
              My dashboard
            </Link>

            <p className="px-3 mb-1.5 text-[10px] font-medium text-zinc-700 uppercase tracking-widest">
              System
            </p>

            {ADMIN_NAV.map(({ href, icon: Icon, label }) => {
              const active = pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={clsx(
                    "flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-100",
                    active
                      ? "bg-zinc-800 text-zinc-100 font-medium"
                      : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
                  )}
                >
                  <Icon
                    className={clsx(
                      "w-4 h-4 flex-shrink-0",
                      active ? "text-amber-400" : "text-zinc-600",
                    )}
                  />
                  {label}
                </Link>
              );
            })}
          </nav>

          {/* User footer */}
          <div className="px-3 py-3 border-t border-zinc-800/80">
            <div className="flex items-center gap-2.5 px-2 py-1.5 mb-1 rounded-lg">
              <div className="w-6 h-6 rounded-full bg-amber-500 flex items-center justify-center
                              text-[11px] font-bold text-white flex-shrink-0">
                {username[0]?.toUpperCase() ?? "A"}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-300 truncate leading-none">{username}</p>
                <p className="text-[10px] text-amber-600 mt-0.5">Administrator</p>
              </div>
            </div>
            <button
              onClick={() => signOut({ callbackUrl: "/login" })}
              className="flex items-center gap-2 w-full px-2 py-1.5 text-xs text-zinc-600
                         hover:text-zinc-400 hover:bg-zinc-900 rounded-lg transition-colors"
            >
              <LogOut className="w-3.5 h-3.5" />
              Sign out
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main
        className="flex-1 min-h-screen bg-zinc-950"
        style={{ marginLeft: "var(--sidebar-width)" }}
      >
        <div
          className="fixed inset-0 pointer-events-none"
          style={{
            marginLeft: "var(--sidebar-width)",
            backgroundImage: "radial-gradient(circle, #27272a 1px, transparent 1px)",
            backgroundSize: "24px 24px",
            opacity: 0.4,
          }}
        />
        <div className="relative px-8 py-8">
          {children}
        </div>
      </main>
    </div>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut, useSession } from "next-auth/react";
import {
  Brain,
  Clock,
  User2,
  Grid3X3,
  ScrollText,
  Zap,
  Key,
  Mic,
  ShieldCheck,
  LogOut,
  ChevronLeft,
  ChevronRight,
  ScanEye,
} from "lucide-react";
import { clsx } from "clsx";
import { useState } from "react";

const USER_NAV = [
  { href: "/dashboard/memories",   icon: Clock,      label: "Memories" },
  { href: "/dashboard/review",     icon: ScanEye,    label: "Review" },
  { href: "/dashboard/identity",   icon: User2,      label: "Identity" },
  { href: "/dashboard/clusters",   icon: Grid3X3,    label: "Clusters" },
  { href: "/dashboard/audit",      icon: ScrollText, label: "Audit trail" },
  { href: "/dashboard/procedures", icon: Zap,        label: "Procedures" },
];

const SETTINGS_NAV = [
  { href: "/dashboard/settings/api-keys",          icon: Key, label: "API Keys" },
  { href: "/dashboard/settings/voice-enrollment",  icon: Mic, label: "Voice" },
];

const ADMIN_NAV = [
  { href: "/admin/users", icon: ShieldCheck, label: "Admin" },
];

const EXPANDED_W = 220;
const COLLAPSED_W = 56;

export function UserShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data: session } = useSession();
  const isAdmin = session?.user?.role === "admin";
  const username = session?.user?.id ?? "";

  const [collapsed, setCollapsed] = useState(true);

  const sidebarW = collapsed ? COLLAPSED_W : EXPANDED_W;

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside
        className="fixed left-0 top-0 h-screen flex flex-col z-20 transition-[width] duration-200 ease-in-out"
        style={{ width: sidebarW }}
      >
        <div className="flex flex-col h-full bg-zinc-950 border-r border-zinc-800/80 overflow-hidden">

          {/* Logo */}
          <div className={clsx("border-b border-zinc-800/80 flex-shrink-0", collapsed ? "px-3 py-4" : "px-5 py-4")}>
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 bg-violet-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <Brain className="w-4 h-4 text-white" />
              </div>
              {!collapsed && (
                <div>
                  <p className="text-sm font-semibold text-zinc-100 leading-none">Smritikosh</p>
                  <p className="text-[10px] text-zinc-600 mt-0.5">स्मृतिकोश</p>
                </div>
              )}
            </div>
          </div>

          {/* Nav */}
          <nav className={clsx("flex-1 py-3 space-y-0.5 overflow-y-auto", collapsed ? "px-2" : "px-3")}>
            {USER_NAV.map(({ href, icon: Icon, label }) => {
              const active = pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  title={collapsed ? label : undefined}
                  className={clsx(
                    "flex items-center rounded-lg text-sm transition-all duration-100",
                    collapsed ? "justify-center p-2.5" : "gap-2.5 px-3 py-2",
                    active
                      ? "bg-zinc-800 text-zinc-100 font-medium"
                      : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
                  )}
                >
                  <Icon className={clsx("w-4 h-4 flex-shrink-0", active ? "text-violet-400" : "text-zinc-600")} />
                  {!collapsed && label}
                </Link>
              );
            })}

            {!collapsed && (
              <div className="pt-4">
                <p className="px-3 mb-1.5 text-[10px] font-medium text-zinc-600 uppercase tracking-widest">
                  Settings
                </p>
              </div>
            )}
            {collapsed && <div className="pt-3 border-t border-zinc-800/60 mx-1" />}

            {SETTINGS_NAV.map(({ href, icon: Icon, label }) => {
              const active = pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  title={collapsed ? label : undefined}
                  className={clsx(
                    "flex items-center rounded-lg text-sm transition-all duration-100",
                    collapsed ? "justify-center p-2.5" : "gap-2.5 px-3 py-2",
                    active
                      ? "bg-zinc-800 text-zinc-100 font-medium"
                      : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
                  )}
                >
                  <Icon className={clsx("w-4 h-4 flex-shrink-0", active ? "text-violet-400" : "text-zinc-600")} />
                  {!collapsed && label}
                </Link>
              );
            })}

            {isAdmin && (
              <>
                {!collapsed && (
                  <div className="pt-4">
                    <p className="px-3 mb-1.5 text-[10px] font-medium text-zinc-600 uppercase tracking-widest">
                      System
                    </p>
                  </div>
                )}
                {ADMIN_NAV.map(({ href, icon: Icon, label }) => {
                  const active = pathname.startsWith("/admin");
                  return (
                    <Link
                      key={href}
                      href={href}
                      title={collapsed ? label : undefined}
                      className={clsx(
                        "flex items-center rounded-lg text-sm transition-all duration-100",
                        collapsed ? "justify-center p-2.5" : "gap-2.5 px-3 py-2",
                        active
                          ? "bg-zinc-800 text-zinc-100 font-medium"
                          : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900",
                      )}
                    >
                      <Icon className={clsx("w-4 h-4 flex-shrink-0", active ? "text-amber-400" : "text-zinc-600")} />
                      {!collapsed && label}
                    </Link>
                  );
                })}
              </>
            )}
          </nav>

          {/* User footer */}
          <div className={clsx("border-t border-zinc-800/80 flex-shrink-0", collapsed ? "px-2 py-3" : "px-3 py-3")}>
            {!collapsed && (
              <div className="flex items-center gap-2.5 px-2 py-1.5 mb-1 rounded-lg">
                <div className="w-6 h-6 rounded-full bg-violet-600 flex items-center justify-center text-[11px] font-bold text-white flex-shrink-0">
                  {username[0]?.toUpperCase() ?? "?"}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-zinc-300 truncate leading-none">{username}</p>
                  <p className="text-[10px] text-zinc-600 mt-0.5 capitalize">{session?.user?.role}</p>
                </div>
              </div>
            )}

            {collapsed ? (
              <div className="flex flex-col items-center gap-1">
                <div
                  title={username}
                  className="w-6 h-6 rounded-full bg-violet-600 flex items-center justify-center text-[11px] font-bold text-white mb-1"
                >
                  {username[0]?.toUpperCase() ?? "?"}
                </div>
                <button
                  onClick={() => signOut({ callbackUrl: "/login" })}
                  title="Sign out"
                  className="p-1.5 text-zinc-600 hover:text-zinc-400 hover:bg-zinc-900 rounded-lg transition-colors"
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => signOut({ callbackUrl: "/login" })}
                className="flex items-center gap-2 w-full px-2 py-1.5 text-xs text-zinc-600 hover:text-zinc-400 hover:bg-zinc-900 rounded-lg transition-colors"
              >
                <LogOut className="w-3.5 h-3.5" />
                Sign out
              </button>
            )}

            {/* Collapse / expand toggle */}
            <button
              onClick={() => setCollapsed((c) => !c)}
              className={clsx(
                "mt-2 flex items-center justify-center w-full py-1 text-zinc-600 hover:text-zinc-400 hover:bg-zinc-900 rounded-lg transition-colors",
              )}
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {collapsed
                ? <ChevronRight className="w-3.5 h-3.5" />
                : <ChevronLeft  className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main
        className="flex-1 min-h-screen bg-zinc-950 transition-[margin-left] duration-200 ease-in-out"
        style={{ marginLeft: sidebarW }}
      >
        <div
          className="fixed inset-0 pointer-events-none transition-[margin-left] duration-200 ease-in-out"
          style={{
            marginLeft: sidebarW,
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

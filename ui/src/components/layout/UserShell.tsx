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
  Menu,
  X,
  Sun,
  Moon,
} from "lucide-react";
import { clsx } from "clsx";
import { useState, useEffect } from "react";
import { useTheme } from "@/hooks/useTheme";

const USER_NAV = [
  { href: "/dashboard/memories",   icon: Clock,      label: "Memories" },
  { href: "/dashboard/review",     icon: ScanEye,    label: "Review" },
  { href: "/dashboard/identity",   icon: User2,      label: "Identity" },
  { href: "/dashboard/clusters",   icon: Grid3X3,    label: "Clusters" },
  { href: "/dashboard/audit",      icon: ScrollText, label: "Audit trail" },
  { href: "/dashboard/procedures", icon: Zap,        label: "Procedures" },
];

const SETTINGS_NAV = [
  { href: "/dashboard/settings/api-keys",         icon: Key, label: "API Keys" },
  { href: "/dashboard/settings/voice-enrollment", icon: Mic, label: "Voice" },
];

const ADMIN_NAV = [
  { href: "/admin/users", icon: ShieldCheck, label: "Admin" },
];

const EXPANDED_W = 220;
const COLLAPSED_W = 56;

export function UserShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data: session } = useSession();
  const isAdmin  = session?.user?.role === "admin";
  const username = session?.user?.id ?? "";
  const { dark, toggle: toggleTheme } = useTheme();

  const [collapsed,  setCollapsed]  = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isMobile,   setIsMobile]   = useState(false);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  useEffect(() => { setMobileOpen(false); }, [pathname]);

  const sidebarW   = collapsed ? COLLAPSED_W : EXPANDED_W;
  const mainMargin = isMobile ? 0 : sidebarW;

  function navLink(
    href: string,
    Icon: React.ElementType,
    label: string,
    c: boolean,
    activeOverride?: boolean,
  ) {
    const active = activeOverride ?? pathname.startsWith(href);
    return (
      <Link
        key={href}
        href={href}
        title={c ? label : undefined}
        className={clsx(
          "flex items-center rounded-lg text-sm transition-all duration-100",
          c ? "justify-center p-2.5" : "gap-2.5 px-3 py-2",
          active
            ? "bg-violet-600/10 dark:bg-zinc-800 text-violet-700 dark:text-zinc-100 font-medium"
            : "text-zinc-600 dark:text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-900",
        )}
      >
        <Icon className={clsx("w-4 h-4 flex-shrink-0", active ? "text-violet-500 dark:text-violet-400" : "text-zinc-400 dark:text-zinc-600")} />
        {!c && label}
      </Link>
    );
  }

  const sidebarBg    = "bg-white dark:bg-zinc-950";
  const sidebarBorder = "border-zinc-200 dark:border-zinc-800/80";
  const footerText   = "text-zinc-500 dark:text-zinc-600 hover:text-zinc-700 dark:hover:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-900";

  return (
    <div className="flex min-h-screen bg-zinc-50 dark:bg-zinc-950">

      {/* ── Desktop sidebar (md+) ──────────────────────────────────────────── */}
      <aside
        className="hidden md:flex fixed left-0 top-0 h-screen flex-col z-20 transition-[width] duration-200 ease-in-out"
        style={{ width: sidebarW }}
      >
        <div className={clsx("flex flex-col h-full overflow-hidden border-r", sidebarBg, sidebarBorder)}>

          {/* Logo */}
          <div className={clsx("border-b flex-shrink-0", sidebarBorder, collapsed ? "px-3 py-4" : "px-5 py-4")}>
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 bg-violet-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <Brain className="w-4 h-4 text-white" />
              </div>
              {!collapsed && (
                <div>
                  <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 leading-none">Smritikosh</p>
                  <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-0.5">स्मृतिकोश</p>
                </div>
              )}
            </div>
          </div>

          {/* Nav */}
          <nav className={clsx("flex-1 py-3 space-y-0.5 overflow-y-auto", collapsed ? "px-2" : "px-3")}>
            {USER_NAV.map(({ href, icon, label }) => navLink(href, icon, label, collapsed))}

            {!collapsed && (
              <div className="pt-4">
                <p className="px-3 mb-1.5 text-[10px] font-medium text-zinc-400 dark:text-zinc-600 uppercase tracking-widest">Settings</p>
              </div>
            )}
            {collapsed && <div className={clsx("pt-3 border-t mx-1", sidebarBorder)} />}

            {SETTINGS_NAV.map(({ href, icon, label }) => navLink(href, icon, label, collapsed))}

            {isAdmin && (
              <>
                {!collapsed && (
                  <div className="pt-4">
                    <p className="px-3 mb-1.5 text-[10px] font-medium text-zinc-400 dark:text-zinc-600 uppercase tracking-widest">System</p>
                  </div>
                )}
                {ADMIN_NAV.map(({ href, icon, label }) =>
                  navLink(href, icon, label, collapsed, pathname.startsWith("/admin"))
                )}
              </>
            )}
          </nav>

          {/* Footer */}
          <div className={clsx("border-t flex-shrink-0", sidebarBorder, collapsed ? "px-2 py-3" : "px-3 py-3")}>
            {!collapsed && (
              <div className="flex items-center gap-2.5 px-2 py-1.5 mb-1 rounded-lg">
                <div className="w-6 h-6 rounded-full bg-violet-600 flex items-center justify-center text-[11px] font-bold text-white flex-shrink-0">
                  {username[0]?.toUpperCase() ?? "?"}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-zinc-800 dark:text-zinc-300 truncate leading-none">{username}</p>
                  <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-0.5 capitalize">{session?.user?.role}</p>
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
                  className={clsx("p-1.5 rounded-lg transition-colors", footerText)}
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={toggleTheme}
                  title={dark ? "Switch to light mode" : "Switch to dark mode"}
                  className={clsx("p-1.5 rounded-lg transition-colors", footerText)}
                >
                  {dark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
                </button>
              </div>
            ) : (
              <div className="space-y-0.5">
                <button
                  onClick={() => signOut({ callbackUrl: "/login" })}
                  className={clsx("flex items-center gap-2 w-full px-2 py-1.5 text-xs rounded-lg transition-colors", footerText)}
                >
                  <LogOut className="w-3.5 h-3.5" />
                  Sign out
                </button>
                <button
                  onClick={toggleTheme}
                  className={clsx("flex items-center gap-2 w-full px-2 py-1.5 text-xs rounded-lg transition-colors", footerText)}
                >
                  {dark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
                  {dark ? "Light mode" : "Dark mode"}
                </button>
              </div>
            )}

            {/* Collapse / expand toggle */}
            <button
              onClick={() => setCollapsed((c) => !c)}
              className={clsx("mt-2 flex items-center justify-center w-full py-1 rounded-lg transition-colors", footerText)}
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {collapsed ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </aside>

      {/* ── Mobile backdrop ──────────────────────────────────────────────── */}
      {mobileOpen && (
        <div
          className="md:hidden fixed inset-0 z-30 bg-black/60"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* ── Mobile drawer ────────────────────────────────────────────────── */}
      <aside
        className={clsx(
          "md:hidden fixed left-0 top-0 h-screen flex flex-col z-40 w-56",
          "transition-transform duration-200 ease-in-out",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className={clsx("flex flex-col h-full overflow-hidden border-r", sidebarBg, sidebarBorder)}>

          {/* Header row */}
          <div className={clsx("px-5 py-4 border-b flex-shrink-0 flex items-center gap-2.5", sidebarBorder)}>
            <div className="w-7 h-7 bg-violet-600 rounded-lg flex items-center justify-center flex-shrink-0">
              <Brain className="w-4 h-4 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 leading-none">Smritikosh</p>
              <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-0.5">स्मृतिकोश</p>
            </div>
            <button
              onClick={() => setMobileOpen(false)}
              className={clsx("p-1.5 rounded-lg transition-colors", footerText)}
              aria-label="Close navigation"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Nav */}
          <nav className="flex-1 px-3 py-3 space-y-0.5 overflow-y-auto">
            {USER_NAV.map(({ href, icon, label }) => navLink(href, icon, label, false))}
            <div className="pt-4">
              <p className="px-3 mb-1.5 text-[10px] font-medium text-zinc-400 dark:text-zinc-600 uppercase tracking-widest">Settings</p>
            </div>
            {SETTINGS_NAV.map(({ href, icon, label }) => navLink(href, icon, label, false))}
            {isAdmin && (
              <>
                <div className="pt-4">
                  <p className="px-3 mb-1.5 text-[10px] font-medium text-zinc-400 dark:text-zinc-600 uppercase tracking-widest">System</p>
                </div>
                {ADMIN_NAV.map(({ href, icon, label }) =>
                  navLink(href, icon, label, false, pathname.startsWith("/admin"))
                )}
              </>
            )}
          </nav>

          {/* Footer */}
          <div className={clsx("px-3 py-3 border-t flex-shrink-0 space-y-0.5", sidebarBorder)}>
            <div className="flex items-center gap-2.5 px-2 py-1.5 mb-1 rounded-lg">
              <div className="w-6 h-6 rounded-full bg-violet-600 flex items-center justify-center text-[11px] font-bold text-white flex-shrink-0">
                {username[0]?.toUpperCase() ?? "?"}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-800 dark:text-zinc-300 truncate leading-none">{username}</p>
                <p className="text-[10px] text-zinc-400 dark:text-zinc-600 mt-0.5 capitalize">{session?.user?.role}</p>
              </div>
            </div>
            <button
              onClick={() => signOut({ callbackUrl: "/login" })}
              className={clsx("flex items-center gap-2 w-full px-2 py-1.5 text-xs rounded-lg transition-colors", footerText)}
            >
              <LogOut className="w-3.5 h-3.5" />
              Sign out
            </button>
            <button
              onClick={toggleTheme}
              className={clsx("flex items-center gap-2 w-full px-2 py-1.5 text-xs rounded-lg transition-colors", footerText)}
            >
              {dark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
              {dark ? "Light mode" : "Dark mode"}
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main content ──────────────────────────────────────────────────── */}
      <main
        className="flex-1 min-h-screen bg-zinc-50 dark:bg-zinc-950 md:transition-[margin-left] duration-200 ease-in-out"
        style={{ marginLeft: mainMargin }}
      >
        {/* Mobile top bar */}
        <div className={clsx(
          "md:hidden sticky top-0 z-10 flex items-center gap-3 px-4 h-12 border-b",
          sidebarBg, sidebarBorder,
        )}>
          <button
            onClick={() => setMobileOpen(true)}
            className={clsx("p-1.5 rounded-lg transition-colors", footerText)}
            aria-label="Open navigation"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 bg-violet-600 rounded flex items-center justify-center">
              <Brain className="w-3 h-3 text-white" />
            </div>
            <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">Smritikosh</span>
          </div>
          <button
            onClick={toggleTheme}
            className={clsx("ml-auto p-1.5 rounded-lg transition-colors", footerText)}
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
        </div>

        {/* Dot-grid background */}
        <div
          className="fixed inset-0 pointer-events-none transition-[margin-left] duration-200 ease-in-out"
          style={{
            marginLeft: mainMargin,
            backgroundImage: `radial-gradient(circle, ${dark ? "#27272a" : "#d4d4d8"} 1px, transparent 1px)`,
            backgroundSize: "24px 24px",
            opacity: 0.4,
          }}
        />
        <div className="relative px-6 py-6 md:px-8 md:py-8">
          {children}
        </div>
      </main>
    </div>
  );
}

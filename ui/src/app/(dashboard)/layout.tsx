import { UserShell } from "@/components/layout/UserShell";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return <UserShell>{children}</UserShell>;
}

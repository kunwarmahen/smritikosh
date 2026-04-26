import { UserTable } from "@/components/admin/UserTable";

export default function AdminUsersPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-900 dark:text-zinc-100 tracking-tight">Users</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Register and manage user accounts.
        </p>
      </div>
      <UserTable />
    </div>
  );
}

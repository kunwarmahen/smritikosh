import { UserTable } from "@/components/admin/UserTable";

export default function AdminUsersPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">Users</h1>
        <p className="text-sm text-slate-500 mt-1">
          Register and manage user accounts.
        </p>
      </div>
      <UserTable />
    </div>
  );
}

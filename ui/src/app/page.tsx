import { redirect } from "next/navigation";

// Root → send authenticated users to dashboard, unauthenticated to login
// (middleware handles the actual auth check; this just prevents a blank page)
export default function RootPage() {
  redirect("/dashboard/memories");
}

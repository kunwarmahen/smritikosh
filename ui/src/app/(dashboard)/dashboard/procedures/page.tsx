"use client";

import { useState } from "react";
import { ProcedureTable } from "@/components/procedures/ProcedureTable";
import { NewProcedureDrawer } from "@/components/procedures/NewProcedureDrawer";

export default function ProceduresPage() {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">Procedures</h1>
        <p className="text-sm text-slate-500 mt-1">
          Triggered instructions injected during context building.
        </p>
      </div>
      <ProcedureTable onNew={() => setDrawerOpen(true)} />
      <NewProcedureDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </div>
  );
}

"use client";

import { useState, useEffect, useRef } from "react";
import { Search, X, Loader2 } from "lucide-react";
import { useSearchMemory } from "@/hooks/useMemory";
import type { SearchResultItem } from "@/types";

interface Props {
  onResults: (results: SearchResultItem[] | null) => void;
}

export function MemorySearch({ onResults }: Props) {
  const [query, setQuery] = useState("");
  const search = useSearchMemory();
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    clearTimeout(debounceRef.current);

    if (!query.trim()) {
      onResults(null);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      const result = await search.mutateAsync({ query, limit: 20 });
      onResults(result.results);
    }, 400);

    return () => clearTimeout(debounceRef.current);
  }, [query]); // eslint-disable-line react-hooks/exhaustive-deps

  function clear() {
    setQuery("");
    onResults(null);
  }

  return (
    <div className="relative">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
      <input
        type="text"
        className="input pl-9 pr-9"
        placeholder="Search memories…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="absolute right-3 top-1/2 -translate-y-1/2">
        {search.isPending ? (
          <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
        ) : query ? (
          <button onClick={clear} className="text-zinc-500 hover:text-zinc-300">
            <X className="w-4 h-4" />
          </button>
        ) : null}
      </div>
    </div>
  );
}

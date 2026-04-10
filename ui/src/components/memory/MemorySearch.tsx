"use client";

import { useState, useEffect, useRef } from "react";
import { Search, X, Loader2, AlertCircle } from "lucide-react";
import { useSearchMemory } from "@/hooks/useMemory";
import type { SearchResultItem } from "@/types";

interface Props {
  onResults: (results: SearchResultItem[] | null) => void;
}

export function MemorySearch({ onResults }: Props) {
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const search = useSearchMemory();
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    clearTimeout(debounceRef.current);
    setError(null);

    if (!query.trim()) {
      onResults(null);
      return;
    }

    debounceRef.current = setTimeout(async () => {
      try {
        const result = await search.mutateAsync({ query, limit: 20 });
        onResults(result.results);
      } catch {
        setError("Search failed — please try again.");
        onResults(null);
      }
    }, 400);

    return () => clearTimeout(debounceRef.current);
  }, [query]); // eslint-disable-line react-hooks/exhaustive-deps

  function clear() {
    setQuery("");
    setError(null);
    onResults(null);
  }

  return (
    <div className="space-y-1.5">
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
      {error && (
        <p className="flex items-center gap-1.5 text-xs text-rose-400 px-1">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          {error}
        </p>
      )}
    </div>
  );
}

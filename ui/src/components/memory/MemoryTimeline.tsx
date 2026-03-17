"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Inbox, RefreshCw } from "lucide-react";
import { useRecentEvents } from "@/hooks/useMemory";
import { MemoryCard } from "./MemoryCard";
import { MemorySearch } from "./MemorySearch";
import type { MemoryEvent, SearchResultItem } from "@/types";

export function MemoryTimeline() {
  const router = useRouter();
  const [limit, setLimit] = useState(20);
  const [searchResults, setSearchResults] = useState<SearchResultItem[] | null>(null);

  const { data, isLoading, isError, refetch, isFetching } = useRecentEvents({ limit });

  const isSearching = searchResults !== null;

  // Convert SearchResultItem → MemoryEvent shape for display
  const searchEvents: MemoryEvent[] = (searchResults ?? []).map((r) => ({
    event_id:        r.event_id,
    user_id:         "",
    raw_text:        r.raw_text,
    importance_score: r.importance_score,
    consolidated:    r.consolidated,
    created_at:      r.created_at,
  }));

  const events: MemoryEvent[] = isSearching ? searchEvents : (data?.events ?? []);

  return (
    <div className="space-y-4">
      {/* Search + header */}
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <MemorySearch onResults={setSearchResults} />
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="btn-secondary px-3 py-2"
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 ${isFetching ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Status line */}
      {isSearching ? (
        <p className="text-xs text-zinc-500">
          {searchResults!.length} result{searchResults!.length !== 1 ? "s" : ""} found
        </p>
      ) : data && (
        <p className="text-xs text-zinc-500">
          Showing {data.events.length} most recent memories
        </p>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center gap-2 text-zinc-500 py-8 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading memories…</span>
        </div>
      )}

      {/* Error */}
      {isError && (
        <div className="card border-rose-500/30 bg-rose-500/5 text-center py-8">
          <p className="text-rose-400 text-sm">Failed to load memories.</p>
          <button onClick={() => refetch()} className="btn-secondary mt-3">Retry</button>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !isError && events.length === 0 && (
        <div className="card text-center py-12">
          <Inbox className="w-10 h-10 text-zinc-700 mx-auto mb-3" />
          <p className="text-zinc-400 text-sm font-medium">
            {isSearching ? "No memories match your search." : "No memories yet."}
          </p>
          <p className="text-zinc-600 text-xs mt-1">
            {isSearching ? "Try a different query." : "Start a conversation to build your memory."}
          </p>
        </div>
      )}

      {/* Event list */}
      <div className="space-y-3">
        {events.map((event) => (
          <MemoryCard
            key={event.event_id}
            event={event}
            onViewGraph={(id) => router.push(`/dashboard/memories/${id}`)}
          />
        ))}
      </div>

      {/* Load more */}
      {!isSearching && events.length > 0 && events.length >= limit && (
        <div className="text-center pt-2">
          <button
            onClick={() => setLimit((l) => l + 20)}
            disabled={isFetching}
            className="btn-secondary"
          >
            {isFetching ? <Loader2 className="w-4 h-4 animate-spin" /> : "Load more"}
          </button>
        </div>
      )}
    </div>
  );
}

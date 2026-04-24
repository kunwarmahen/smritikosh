"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Inbox, RefreshCw, Plus, Upload } from "lucide-react";
import { useRecentEvents } from "@/hooks/useMemory";
import { MemoryCard } from "./MemoryCard";
import { MemorySearch } from "./MemorySearch";
import { AddMemoryForm } from "./AddMemoryForm";
import { UploadMediaForm } from "./UploadMediaForm";
import type { MemoryEvent, SearchResultItem } from "@/types";

export function MemoryTimeline() {
  const router = useRouter();
  const [limit, setLimit] = useState(20);
  const [searchResults, setSearchResults] = useState<SearchResultItem[] | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showUploadForm, setShowUploadForm] = useState(false);

  const { data, isLoading, isError, refetch, isFetching } = useRecentEvents({ limit });

  const isSearching = searchResults !== null;

  // Convert SearchResultItem → MemoryEvent shape for display
  const searchEvents: MemoryEvent[] = (searchResults ?? []).map((r) => ({
    event_id:         r.event_id,
    user_id:          "",
    raw_text:         r.raw_text,
    summary:          r.summary ?? null,
    importance_score: r.importance_score,
    consolidated:     r.consolidated,
    created_at:       r.created_at,
    hybrid_score:     r.hybrid_score,
    similarity_score: r.similarity_score,
  }));

  const events: MemoryEvent[] = isSearching ? searchEvents : (data?.events ?? []);

  return (
    <div className="space-y-4">
      {showAddForm && <AddMemoryForm onClose={() => setShowAddForm(false)} />}
      {showUploadForm && <UploadMediaForm onClose={() => setShowUploadForm(false)} />}

      {/* Search + header */}
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <MemorySearch onResults={setSearchResults} />
        </div>
        <button
          onClick={() => setShowUploadForm(true)}
          className="btn-secondary px-3 py-2 flex-shrink-0 flex items-center gap-1.5 text-xs"
          title="Upload media"
        >
          <Upload className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Upload</span>
        </button>
        <button
          onClick={() => setShowAddForm(true)}
          className="btn-secondary px-3 py-2 flex-shrink-0"
          title="Add memory"
        >
          <Plus className="w-4 h-4" />
        </button>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="btn-secondary px-3 py-2 flex-shrink-0"
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

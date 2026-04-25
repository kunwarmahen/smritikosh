'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { FileUp, CheckCircle2, AlertCircle, Loader2, Upload, X, Image, Mic } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useUploadMedia, useMediaStatus, useConfirmMediaFacts, useVoiceEnrollmentStatus } from '@/hooks/useMemory';
import { PendingFact } from '@/types';

type MediaCategory = 'voice_note' | 'meeting' | 'document' | 'image';
type ContentType = 'voice_note' | 'meeting_recording' | 'document' | 'receipt' | 'screenshot' | 'whiteboard';
type Step = 'upload' | 'processing' | 'review' | 'success' | 'nothing_found' | 'error';

const IMAGE_SUBTYPES: { value: ContentType; label: string; hint: string }[] = [
  { value: 'receipt', label: '🧾 Receipt', hint: 'Extracts purchase/lifestyle signals' },
  { value: 'screenshot', label: '🖥 Screenshot', hint: 'Extracts tool/tech/workflow signals' },
  { value: 'whiteboard', label: '📋 Whiteboard', hint: 'Extracts project/goal/decision signals' },
];

interface UploadMediaFormProps {
  onClose: () => void;
}

export function UploadMediaForm({ onClose }: UploadMediaFormProps) {
  const { data: session } = useSession();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const enrollmentStatus = useVoiceEnrollmentStatus();

  const [step, setStep] = useState<Step>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [mediaCategory, setMediaCategory] = useState<MediaCategory>('voice_note');
  const [imageSubtype, setImageSubtype] = useState<ContentType>('receipt');
  const [contextNote, setContextNote] = useState('');
  const [mediaId, setMediaId] = useState<string | null>(null);
  const [selectedFactIndices, setSelectedFactIndices] = useState<Set<number>>(new Set());
  const [savedCount, setSavedCount] = useState(0);

  const contentType: ContentType =
    mediaCategory === 'image'
      ? imageSubtype
      : mediaCategory === 'meeting'
      ? 'meeting_recording'
      : (mediaCategory as ContentType);

  const uploadMedia = useUploadMedia();
  const mediaStatus = useMediaStatus(mediaId);
  const confirmFacts = useConfirmMediaFacts();

  // Auto-transition when processing completes
  useEffect(() => {
    if (!mediaStatus.data) return;

    const status = mediaStatus.data.status;
    if (status === 'processing') return; // Still processing

    if (status === 'nothing_found') {
      setStep('nothing_found');
    } else if (status === 'failed') {
      setStep('error');
    } else if (status === 'complete') {
      if (mediaStatus.data.facts_pending_review > 0) {
        setSelectedFactIndices(new Set(mediaStatus.data.pending_facts.map((_, i) => i)));
        setStep('review');
      } else {
        setSavedCount(mediaStatus.data.facts_extracted);
        setStep('success');
      }
    }
  }, [mediaStatus.data?.status, mediaStatus.data?.facts_pending_review]);

  const handleFileSelect = (files: FileList | null) => {
    if (files && files[0]) {
      setSelectedFile(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.add('border-violet-400', 'bg-violet-950');
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.currentTarget.classList.remove('border-violet-400', 'bg-violet-950');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-violet-400', 'bg-violet-950');
    handleFileSelect(e.dataTransfer.files);
  };

  const handleSubmitUpload = async () => {
    if (!selectedFile || !session) return;

    const formData = new FormData();
    formData.append('user_id', session.user.id);
    formData.append('app_id', 'default');
    formData.append('content_type', contentType);
    formData.append('context_note', contextNote);
    formData.append('file', selectedFile);

    setStep('processing');
    const result = await uploadMedia.mutateAsync(formData);
    setMediaId(result.media_id);
  };

  const handleConfirmFacts = async () => {
    if (!mediaId || !session) return;

    const confirmedIndices = Array.from(selectedFactIndices);
    await confirmFacts.mutateAsync({
      mediaId,
      user_id: session.user.id,
      app_id: 'default',
      confirmed_indices: confirmedIndices,
    });

    const autoSaved = mediaStatus.data?.facts_extracted ?? 0;
    setSavedCount(autoSaved + confirmedIndices.length);
    setStep('success');
  };

  // ── Step: Upload ──────────────────────────────────────────────────────
  if (step === 'upload') {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-xl shadow-2xl">
          {/* Header */}
          <div className="border-b border-zinc-800 p-5 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Upload className="h-5 w-5 text-violet-400" />
              <h2 className="text-lg font-semibold text-zinc-100">Upload Media</h2>
            </div>
            <button
              onClick={onClose}
              className="text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Body */}
          <div className="p-5 space-y-4">
            {/* Dropzone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center transition-colors cursor-pointer hover:border-zinc-600"
              onClick={() => fileInputRef.current?.click()}
            >
              <FileUp className="h-12 w-12 text-zinc-600 mx-auto mb-3" />
              <p className="text-sm font-medium text-zinc-300 mb-1">
                {selectedFile ? selectedFile.name : 'Drop file or click to browse'}
              </p>
              <p className="text-xs text-zinc-500">
                {mediaCategory === 'voice_note'
                  ? 'MP3, WAV, M4A, WebM (max 25 MB)'
                  : mediaCategory === 'meeting'
                  ? 'MP3, WAV, M4A, WebM, MP4 (max 500 MB)'
                  : mediaCategory === 'image'
                  ? 'JPG, PNG, WebP, GIF (max 20 MB)'
                  : 'PDF, TXT, MD, CSV (max 10 MB)'}
              </p>
              <input
                ref={fileInputRef}
                type="file"
                onChange={(e) => handleFileSelect(e.target.files)}
                accept={
                  mediaCategory === 'voice_note' || mediaCategory === 'meeting'
                    ? 'audio/*,video/*'
                    : mediaCategory === 'image'
                    ? 'image/jpeg,image/png,image/gif,image/webp'
                    : '.pdf,.txt,.md,.csv'
                }
                className="hidden"
              />
            </div>

            {/* Category Selector */}
            <div>
              <label className="label">Content Type</label>
              <div className="flex gap-2 mt-2 flex-wrap">
                {(
                  [
                    { value: 'voice_note', label: '🎙 Voice Note' },
                    { value: 'meeting', label: '🎧 Meeting' },
                    { value: 'document', label: '📄 Document' },
                    { value: 'image', label: '🖼 Image' },
                  ] as { value: MediaCategory; label: string }[]
                ).map(({ value, label }) => (
                  <button
                    key={value}
                    onClick={() => {
                      setMediaCategory(value);
                      setSelectedFile(null);
                    }}
                    className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      mediaCategory === value
                        ? 'bg-violet-600 text-white'
                        : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Meeting diarization info */}
            {mediaCategory === 'meeting' && (
              <div className={`rounded-lg px-4 py-3 text-xs ${
                enrollmentStatus.data?.has_embedding
                  ? 'bg-emerald-950/40 border border-emerald-800/50 text-emerald-300'
                  : 'bg-zinc-800 border border-zinc-700 text-zinc-400'
              }`}>
                {enrollmentStatus.data?.has_embedding ? (
                  <>
                    <p className="font-medium mb-0.5">Speaker matching enabled</p>
                    <p>Your enrolled voice will be used to identify your speech segments automatically.</p>
                  </>
                ) : enrollmentStatus.data?.enrolled ? (
                  <>
                    <p className="font-medium mb-0.5 text-amber-300">Enrolled without speaker embedding</p>
                    <p>
                      Install resemblyzer for speaker matching:{' '}
                      <code className="font-mono bg-zinc-700 px-1 rounded">pip install resemblyzer</code>
                    </p>
                  </>
                ) : (
                  <>
                    <p className="font-medium mb-0.5">No voice enrolled</p>
                    <p>
                      First-person filter will be applied to the full transcript.{' '}
                      <Link
                        href="/dashboard/settings/voice-enrollment"
                        className="text-violet-400 hover:text-violet-300 underline"
                        onClick={onClose}
                      >
                        Enroll your voice
                      </Link>{' '}
                      for accurate speaker matching.
                    </p>
                  </>
                )}
              </div>
            )}

            {/* Image subtype selector */}
            {mediaCategory === 'image' && (
              <div>
                <label className="label">Image Type</label>
                <div className="flex flex-col gap-2 mt-2">
                  {IMAGE_SUBTYPES.map(({ value, label, hint }) => (
                    <button
                      key={value}
                      onClick={() => setImageSubtype(value)}
                      className={`flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm transition-colors text-left ${
                        imageSubtype === value
                          ? 'bg-violet-600/20 border border-violet-500 text-zinc-100'
                          : 'bg-zinc-800 border border-transparent text-zinc-400 hover:bg-zinc-700'
                      }`}
                    >
                      <span className="font-medium">{label}</span>
                      <span className="text-xs text-zinc-500 ml-auto">{hint}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Context Note */}
            <div>
              <label className="label">Context (Optional)</label>
              <textarea
                value={contextNote}
                onChange={(e) => setContextNote(e.target.value)}
                placeholder="What should I know about this file?"
                className="input w-full min-h-20 resize-none"
              />
            </div>
          </div>

          {/* Footer */}
          <div className="border-t border-zinc-800 p-4 flex justify-end gap-3">
            <button
              onClick={onClose}
              className="btn-secondary"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmitUpload}
              disabled={!selectedFile || uploadMedia.isPending}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {uploadMedia.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading…
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4" />
                  Upload
                </>
              )}
            </button>
          </div>

          {uploadMedia.isError && (
            <div className="bg-rose-950 border-t border-rose-800 px-5 py-3 text-sm text-rose-300">
              {uploadMedia.error?.message || 'Upload failed'}
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Step: Processing ──────────────────────────────────────────────────
  if (step === 'processing') {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-md shadow-2xl p-8 text-center">
          <Loader2 className="h-12 w-12 text-violet-400 mx-auto mb-4 animate-spin" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">Processing…</h3>
          <p className="text-sm text-zinc-400">
            {mediaCategory === 'voice_note'
              ? 'Transcribing'
              : mediaCategory === 'meeting'
              ? 'Transcribing and identifying speakers in'
              : mediaCategory === 'image'
              ? 'Analysing image'
              : 'Analysing'}{' '}
            your file
          </p>
        </div>
      </div>
    );
  }

  // ── Step: Review ──────────────────────────────────────────────────────
  if (step === 'review' && mediaStatus.data) {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-lg shadow-2xl">
          {/* Header */}
          <div className="border-b border-zinc-800 p-5 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-zinc-100">Review Facts</h2>
            <button
              onClick={onClose}
              className="text-zinc-500 hover:text-zinc-300"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Body */}
          <div className="p-5 space-y-4">
            <p className="text-sm text-zinc-300">
              Found {mediaStatus.data.facts_pending_review} thing
              {mediaStatus.data.facts_pending_review !== 1 ? 's' : ''} that might be worth remembering:
            </p>

            <div className="space-y-2 max-h-96 overflow-y-auto">
              {mediaStatus.data.pending_facts.map((fact, idx) => (
                <label
                  key={idx}
                  className="flex items-start gap-3 p-3 rounded-lg bg-zinc-800 hover:bg-zinc-700 cursor-pointer transition-colors group"
                >
                  <input
                    type="checkbox"
                    checked={selectedFactIndices.has(idx)}
                    onChange={(e) => {
                      const newSet = new Set(selectedFactIndices);
                      if (e.target.checked) {
                        newSet.add(idx);
                      } else {
                        newSet.delete(idx);
                      }
                      setSelectedFactIndices(newSet);
                    }}
                    className="mt-0.5"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-100 break-words">
                      {fact.content}
                    </p>
                    <div className="flex gap-2 mt-1 flex-wrap">
                      <span className="text-xs bg-zinc-700 text-zinc-300 px-2 py-0.5 rounded">
                        {fact.category}
                      </span>
                      <span className="text-xs text-zinc-400">
                        {Math.round(fact.relevance_score * 100)}% relevant
                      </span>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="border-t border-zinc-800 p-4 flex justify-end gap-3">
            <button
              onClick={() => {
                setSelectedFactIndices(new Set());
                handleConfirmFacts();
              }}
              className="btn-secondary"
            >
              Dismiss All
            </button>
            <button
              onClick={handleConfirmFacts}
              disabled={selectedFactIndices.size === 0 || confirmFacts.isPending}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {confirmFacts.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Saving…
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-4 w-4" />
                  Save {selectedFactIndices.size > 0 ? selectedFactIndices.size : 0}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Step: Success ─────────────────────────────────────────────────────
  if (step === 'success') {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-md shadow-2xl p-8 text-center">
          <CheckCircle2 className="h-12 w-12 text-emerald-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">Memories Saved!</h3>
          <p className="text-sm text-zinc-400 mb-6">
            Saved {savedCount} fact{savedCount !== 1 ? 's' : ''}
          </p>
          <button
            onClick={onClose}
            className="btn-primary w-full"
          >
            Done
          </button>
        </div>
      </div>
    );
  }

  // ── Step: Nothing Found ───────────────────────────────────────────────
  if (step === 'nothing_found') {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-md shadow-2xl p-8 text-center">
          <AlertCircle className="h-12 w-12 text-amber-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">Nothing Found</h3>
          <p className="text-sm text-zinc-400 mb-6">
            No extractable content was found in this file. Try adding a context note to help guide the analysis.
          </p>
          <button
            onClick={onClose}
            className="btn-primary w-full"
          >
            Back
          </button>
        </div>
      </div>
    );
  }

  // ── Step: Error ───────────────────────────────────────────────────────
  if (step === 'error' && mediaStatus.data?.message) {
    return (
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl w-full max-w-md shadow-2xl p-8 text-center">
          <AlertCircle className="h-12 w-12 text-rose-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">Processing Error</h3>
          <p className="text-sm text-zinc-400 mb-6">
            {mediaStatus.data.message}
          </p>
          <button
            onClick={onClose}
            className="btn-primary w-full"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return null;
}

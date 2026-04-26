'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Mic, MicOff, CheckCircle2, AlertCircle, Loader2, Trash2, RefreshCw, Radio } from 'lucide-react';
import { useSession } from 'next-auth/react';
import { useVoiceEnrollmentStatus, useEnrollVoice, useDeleteVoiceEnrollment } from '@/hooks/useMemory';

const MAX_RECORD_SECONDS = 30;

type RecordState = 'idle' | 'recording' | 'recorded' | 'uploading' | 'done' | 'error';

export default function VoiceEnrollmentPage() {
  const { data: session } = useSession();
  const enrollmentStatus = useVoiceEnrollmentStatus();
  const enrollVoice = useEnrollVoice();
  const deleteEnrollment = useDeleteVoiceEnrollment();

  const [recordState, setRecordState] = useState<RecordState>('idle');
  const [secondsLeft, setSecondsLeft] = useState(MAX_RECORD_SECONDS);
  const [errorMsg, setErrorMsg] = useState('');
  const [waveformData, setWaveformData] = useState<number[]>(Array(40).fill(2));

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recordedBlobRef = useRef<Blob | null>(null);

  // Animate waveform during recording
  const animateWaveform = useCallback(() => {
    if (!analyserRef.current) return;
    const analyser = analyserRef.current;
    const buf = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(buf);

    // Downsample to 40 bars
    const barCount = 40;
    const step = Math.floor(buf.length / barCount);
    const bars = Array.from({ length: barCount }, (_, i) => {
      const slice = buf.slice(i * step, (i + 1) * step);
      const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
      return Math.max(2, Math.round((avg / 255) * 64));
    });

    setWaveformData(bars);
    animFrameRef.current = requestAnimationFrame(animateWaveform);
  }, []);

  const stopAnimation = useCallback(() => {
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setWaveformData(Array(40).fill(2));
  }, []);

  const stopRecording = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    stopAnimation();
    mediaRecorderRef.current?.stop();
  }, [stopAnimation]);

  const startRecording = useCallback(async () => {
    try {
      setErrorMsg('');
      chunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Set up Web Audio for waveform
      const ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        recordedBlobRef.current = new Blob(chunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach((t) => t.stop());
        ctx.close();
        setRecordState('recorded');
      };

      recorder.start(100);
      setRecordState('recording');
      setSecondsLeft(MAX_RECORD_SECONDS);
      animFrameRef.current = requestAnimationFrame(animateWaveform);

      // Auto-stop after MAX_RECORD_SECONDS
      let remaining = MAX_RECORD_SECONDS;
      timerRef.current = setInterval(() => {
        remaining -= 1;
        setSecondsLeft(remaining);
        if (remaining <= 0) stopRecording();
      }, 1000);
    } catch (err) {
      setErrorMsg('Microphone access denied. Please allow microphone access in your browser settings.');
      setRecordState('error');
    }
  }, [animateWaveform, stopRecording]);

  const handleUpload = async () => {
    if (!recordedBlobRef.current || !session) return;
    setRecordState('uploading');
    setErrorMsg('');

    const formData = new FormData();
    formData.append('app_id', 'default');
    formData.append('file', recordedBlobRef.current, 'enrollment.webm');

    try {
      await enrollVoice.mutateAsync(formData);
      setRecordState('done');
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'Enrollment failed');
      setRecordState('error');
    }
  };

  const handleDelete = async () => {
    await deleteEnrollment.mutateAsync();
    setRecordState('idle');
    setSecondsLeft(MAX_RECORD_SECONDS);
    recordedBlobRef.current = null;
  };

  const handleReRecord = () => {
    recordedBlobRef.current = null;
    setRecordState('idle');
    setSecondsLeft(MAX_RECORD_SECONDS);
    setErrorMsg('');
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      stopAnimation();
    };
  }, [stopAnimation]);

  const isEnrolled = enrollmentStatus.data?.enrolled ?? false;
  const hasEmbedding = enrollmentStatus.data?.has_embedding ?? false;

  return (
    <div className="max-w-2xl">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-100 mb-1">Voice Enrollment</h1>
        <p className="text-sm text-zinc-400">
          Record a 30-second voice sample to enable speaker identification in meeting recordings.
          Your voice profile is used to find your speech segments and extract only what you said.
        </p>
      </div>

      {/* Current enrollment status */}
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 mb-6">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Enrollment Status</h2>
          {enrollmentStatus.isLoading && (
            <Loader2 className="h-4 w-4 text-zinc-500 animate-spin" />
          )}
        </div>

        {!enrollmentStatus.isLoading && (
          <div className="flex items-center gap-3">
            {isEnrolled ? (
              <>
                <CheckCircle2 className="h-5 w-5 text-emerald-400 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                    {hasEmbedding ? 'Enrolled with speaker embedding' : 'Enrolled (no embedding)'}
                  </p>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    {hasEmbedding
                      ? `Speaker d-vector: ${enrollmentStatus.data?.embedding_dim ?? '?'} dimensions · Enrolled ${enrollmentStatus.data?.enrolled_at ? new Date(enrollmentStatus.data.enrolled_at).toLocaleDateString() : ''}`
                      : 'Install resemblyzer for speaker matching: pip install resemblyzer'}
                  </p>
                </div>
                <button
                  onClick={handleDelete}
                  disabled={deleteEnrollment.isPending}
                  className="ml-auto text-zinc-500 hover:text-rose-400 transition-colors p-1.5 rounded-lg hover:bg-rose-100 dark:hover:bg-rose-950"
                  title="Remove enrollment"
                >
                  {deleteEnrollment.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                </button>
              </>
            ) : (
              <>
                <MicOff className="h-5 w-5 text-zinc-600 flex-shrink-0" />
                <p className="text-sm text-zinc-500">Not enrolled — record a sample below</p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Recorder */}
      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5">
        <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-4">
          {isEnrolled ? 'Re-enroll' : 'Record Voice Sample'}
        </h2>

        {/* Waveform visualiser */}
        <div className="flex items-end justify-center gap-0.5 h-16 mb-5 bg-zinc-100 dark:bg-zinc-950 rounded-lg px-4 py-2">
          {waveformData.map((h, i) => (
            <div
              key={i}
              className={`w-1.5 rounded-sm transition-all duration-75 ${
                recordState === 'recording' ? 'bg-violet-400' : 'bg-zinc-700'
              }`}
              style={{ height: `${h}px` }}
            />
          ))}
        </div>

        {/* Recording controls */}
        <div className="flex items-center justify-center gap-4">
          {recordState === 'idle' && (
            <button
              onClick={startRecording}
              className="flex items-center gap-2 px-6 py-3 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-medium text-sm transition-colors"
            >
              <Mic className="h-4 w-4" />
              Start Recording
            </button>
          )}

          {recordState === 'recording' && (
            <>
              <div className="flex items-center gap-2 text-sm text-rose-400">
                <Radio className="h-4 w-4 animate-pulse" />
                Recording — {secondsLeft}s left
              </div>
              <button
                onClick={stopRecording}
                className="flex items-center gap-2 px-5 py-2.5 bg-rose-600 hover:bg-rose-500 text-white rounded-xl font-medium text-sm transition-colors"
              >
                <MicOff className="h-4 w-4" />
                Stop
              </button>
            </>
          )}

          {recordState === 'recorded' && (
            <div className="flex items-center gap-3">
              <button
                onClick={handleReRecord}
                className="flex items-center gap-2 px-4 py-2.5 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 rounded-xl font-medium text-sm transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                Re-record
              </button>
              <button
                onClick={handleUpload}
                className="flex items-center gap-2 px-5 py-2.5 bg-violet-600 hover:bg-violet-500 text-white rounded-xl font-medium text-sm transition-colors"
              >
                <CheckCircle2 className="h-4 w-4" />
                Save Enrollment
              </button>
            </div>
          )}

          {recordState === 'uploading' && (
            <div className="flex items-center gap-2 text-sm text-zinc-400">
              <Loader2 className="h-4 w-4 animate-spin text-violet-400" />
              Computing speaker embedding…
            </div>
          )}

          {recordState === 'done' && (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <CheckCircle2 className="h-4 w-4" />
                Enrolled successfully
              </div>
              <button
                onClick={handleReRecord}
                className="flex items-center gap-2 px-4 py-2.5 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 rounded-xl font-medium text-sm transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                Re-record
              </button>
            </div>
          )}

          {recordState === 'error' && (
            <div className="flex flex-col items-center gap-3">
              <div className="flex items-center gap-2 text-sm text-rose-400">
                <AlertCircle className="h-4 w-4" />
                {errorMsg || 'Something went wrong'}
              </div>
              <button
                onClick={handleReRecord}
                className="flex items-center gap-2 px-4 py-2.5 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 rounded-xl font-medium text-sm transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                Try again
              </button>
            </div>
          )}
        </div>

        <p className="text-xs text-zinc-600 text-center mt-4">
          Read anything aloud — a passage of text, today's plans, anything natural. 30 seconds is enough.
        </p>
      </div>

      {/* How it works */}
      <div className="mt-6 bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800/50 rounded-xl p-5 space-y-2">
        <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-widest mb-3">How it works</h3>
        <div className="flex gap-3 text-sm text-zinc-400">
          <span className="text-violet-400 font-mono text-xs mt-0.5">1</span>
          Record 30 seconds of your voice
        </div>
        <div className="flex gap-3 text-sm text-zinc-400">
          <span className="text-violet-400 font-mono text-xs mt-0.5">2</span>
          A speaker d-vector embedding is computed from your sample
        </div>
        <div className="flex gap-3 text-sm text-zinc-400">
          <span className="text-violet-400 font-mono text-xs mt-0.5">3</span>
          When you upload a meeting recording, your voice is identified automatically
        </div>
        <div className="flex gap-3 text-sm text-zinc-400">
          <span className="text-violet-400 font-mono text-xs mt-0.5">4</span>
          Only your speech is extracted — other participants are ignored
        </div>
        <p className="text-xs text-zinc-600 pt-2">
          Requires resemblyzer (pip install resemblyzer) and optionally pyannote.audio for full diarization.
        </p>
      </div>
    </div>
  );
}

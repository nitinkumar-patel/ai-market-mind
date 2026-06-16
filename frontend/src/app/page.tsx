'use client';

import { useResearch } from '@/hooks/use-research';
import { ProgressTimeline } from '@/components/progress-timeline';
import { ResearchForm } from '@/components/research-form';
import { ResultsCard } from '@/components/results-card';

export default function HomePage() {
  const { state, run, reset } = useResearch();

  const isStreaming = state.status === 'streaming';
  const isDone      = state.status === 'done';
  const isError     = state.status === 'error';

  return (
    <main className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-4">
      <ResearchForm onSubmit={run} isLoading={isStreaming} />

      {isError && (
        <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger flex items-start gap-2.5">
          <svg className="w-4 h-4 mt-0.5 shrink-0" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
            <path d="M8 5v4M8 10.5v.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          {state.error}
        </div>
      )}

      {(isStreaming || isDone) && (
        <ProgressTimeline steps={state.steps} isDone={isDone} topic={state.topic} />
      )}

      {isDone && state.result && (
        <ResultsCard topic={state.topic} result={state.result} onReset={reset} />
      )}
    </main>
  );
}

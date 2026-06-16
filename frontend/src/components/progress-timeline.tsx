'use client';

import type { StepUpdate } from '@/lib/types';

const STEP_LABELS: Record<string, string> = {
  memory_check: 'Memory Check',
  memory_hit:   'Cache Hit',
  memory_miss:  'Cache Miss',
  planner:      'Research Planner',
  search:       'Web Search',
  ingest:       'Ingest & Embed',
  writer:       'Report Writer',
  review:       'Guardrail Review',
};

interface Props {
  steps: StepUpdate[];
  isDone: boolean;
  topic: string;
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className ?? 'w-3 h-3'} viewBox="0 0 12 12" fill="none">
      <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function StepRow({
  step,
  isActive,
  isDone,
  isLast,
}: {
  step: StepUpdate;
  isActive: boolean;
  isDone: boolean;
  isLast: boolean;
}) {
  return (
    <div className="flex gap-3 animate-fade-up">
      {/* Track: dot + connector */}
      <div className="flex flex-col items-center w-5 shrink-0">
        <div
          className={`
            w-5 h-5 rounded-full flex items-center justify-center border-2 shrink-0 transition-all duration-300
            ${isDone
              ? 'bg-emerald-500 border-emerald-500 text-white'
              : isActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-slate-200 bg-white'}
          `}
        >
          {isDone ? (
            <CheckIcon />
          ) : isActive ? (
            <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse-dot" />
          ) : (
            <div className="w-1.5 h-1.5 rounded-full bg-slate-300" />
          )}
        </div>

        {!isLast && (
          <div className={`w-px flex-1 my-1 min-h-[12px] transition-colors duration-300 ${isDone ? 'bg-emerald-200' : 'bg-slate-100'}`} />
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 min-w-0 ${isLast ? 'pb-0' : 'pb-3'}`}>
        <div
          className={`text-xs font-semibold transition-colors duration-200 ${
            isDone ? 'text-emerald-700' : isActive ? 'text-blue-600' : 'text-slate-400'
          }`}
        >
          {STEP_LABELS[step.step] ?? step.step}
        </div>
        <div className="text-xs text-slate-400 mt-0.5 leading-relaxed">{step.message}</div>
      </div>
    </div>
  );
}

export function ProgressTimeline({ steps, isDone, topic }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 animate-fade-up">
      {/* Header */}
      <div className="flex items-start gap-3 mb-5">
        <div className="mt-0.5 shrink-0">
          {isDone ? (
            <div className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center text-white">
              <CheckIcon />
            </div>
          ) : (
            <div className="w-5 h-5 border-2 border-slate-200 border-t-blue-500 rounded-full animate-spin" />
          )}
        </div>
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-slate-900">
            {isDone ? 'Research Complete' : 'Running Research Agent…'}
          </h3>
          <p className="text-xs text-slate-400 mt-0.5 truncate">
            &ldquo;{topic}&rdquo;
          </p>
        </div>
      </div>

      {/* Steps */}
      {steps.length > 0 && (
        <div className="ml-1">
          {steps.map((step, i) => {
            const isLast   = i === steps.length - 1;
            const isActive = !isDone && isLast;
            const stepDone = isDone || !isLast;
            return (
              <StepRow
                key={`${step.step}-${i}`}
                step={step}
                isActive={isActive}
                isDone={stepDone}
                isLast={isLast}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

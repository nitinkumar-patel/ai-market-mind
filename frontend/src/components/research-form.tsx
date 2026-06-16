'use client';

import { type FormEvent, useState } from 'react';
import type { ResearchDepth, ResearchRequest } from '@/lib/types';

const EXAMPLES = [
  'GenAI in Marketing 2025',
  'Nike brand strategy',
  'DTC skincare market trends',
  'Streaming wars competitive landscape',
];

interface Props {
  onSubmit: (req: ResearchRequest) => void;
  isLoading: boolean;
}

export function ResearchForm({ onSubmit, isLoading }: Props) {
  const [topic, setTopic] = useState('');
  const [depth, setDepth] = useState<ResearchDepth>('quick');

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = topic.trim();
    if (!trimmed || isLoading) return;
    onSubmit({ topic: trimmed, depth });
  }

  return (
    <div className="bg-bg-surface rounded-2xl border border-stroke shadow-sm p-6">
      <h1 className="text-lg font-bold font-display text-ink-primary mb-0.5">Market Research Agent</h1>
      <p className="text-sm text-ink-secondary mb-5">
        Enter a brand, topic, or trend — the AI agent researches, synthesizes, and delivers an
        executive-ready report in real time.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="flex gap-2.5 flex-wrap sm:flex-nowrap">
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. GenAI in Marketing 2025…"
            disabled={isLoading}
            autoFocus
            className="
              flex-1 min-w-0 px-3.5 py-2.5 text-sm rounded-lg
              bg-bg-raised border border-stroke
              text-ink-primary placeholder:text-ink-tertiary
              outline-none focus:border-accent focus:ring-2 focus:ring-accent/10
              disabled:opacity-50 disabled:cursor-not-allowed transition
            "
          />

          <select
            value={depth}
            onChange={(e) => setDepth(e.target.value as ResearchDepth)}
            disabled={isLoading}
            className="
              px-3 py-2.5 text-sm rounded-lg
              bg-bg-raised border border-stroke text-ink-primary
              outline-none focus:border-accent focus:ring-2 focus:ring-accent/10
              disabled:opacity-50 cursor-pointer transition
            "
          >
            <option value="quick">Quick</option>
            <option value="detailed">Detailed</option>
          </select>

          <button
            type="submit"
            disabled={!topic.trim() || isLoading}
            className="
              px-5 py-2.5 bg-accent hover:bg-accent-light active:scale-95
              text-accent-subtle text-sm font-semibold rounded-lg transition
              disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100
              flex items-center gap-2 whitespace-nowrap
            "
          >
            {isLoading ? (
              <>
                <svg className="animate-spin w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity=".25" />
                  <path fill="currentColor" fillOpacity=".75" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Researching…
              </>
            ) : (
              <>
                Research
                <svg className="w-4 h-4 shrink-0" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </>
            )}
          </button>
        </div>
      </form>

      <div className="flex flex-wrap items-center gap-2 mt-3.5">
        <span className="text-xs text-ink-tertiary font-mono">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setTopic(ex)}
            disabled={isLoading}
            className="
              text-xs text-accent bg-accent/10 hover:bg-accent/20 border border-accent/20
              rounded-full px-2.5 py-0.5 transition disabled:opacity-40 disabled:cursor-not-allowed
            "
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}

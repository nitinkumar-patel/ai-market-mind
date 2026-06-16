'use client';

import { useRef } from 'react';
import { useResearch } from '@/hooks/use-research';
import { ProgressTimeline } from '@/components/progress-timeline';
import { ResearchForm } from '@/components/research-form';
import { ResultsCard } from '@/components/results-card';

// ── Why ──────────────────────────────────────────────────────────────────────
const WHY_ITEMS = [
  {
    icon: '⏰',
    title: 'Weeks of waiting',
    body: 'Traditional market research takes 2–4 weeks to brief, research, write, and review. By delivery, the opportunity is gone.',
  },
  {
    icon: '💸',
    title: 'Expensive consultants',
    body: 'Agency reports start at $5,000+. Building an in-house team costs even more — and the turnaround is still slow.',
  },
  {
    icon: '📅',
    title: 'Already stale',
    body: "By the time the report lands in your inbox, the market has moved. You're making decisions on yesterday's data.",
  },
];

// ── How ──────────────────────────────────────────────────────────────────────
const HOW_STEPS = [
  {
    step: '01',
    label: 'You enter a topic',
    detail: 'A brand, trend, competitor, or market question — anything you need intelligence on.',
  },
  {
    step: '02',
    label: 'Memory check (instant)',
    detail: 'MarketMind checks its vector cache first. If a recent report exists, it\'s returned in milliseconds.',
  },
  {
    step: '03',
    label: 'Live web research',
    detail: 'The AI agent searches and reads live sources via Tavily. No hallucinated data — every fact is grounded in a real URL.',
  },
  {
    step: '04',
    label: 'Synthesise & embed',
    detail: 'Content is chunked, embedded with pgvector, and synthesised by GPT-4 into a coherent narrative.',
  },
  {
    step: '05',
    label: 'Executive report delivered',
    detail: 'A structured report appears in seconds — executive summary, key findings, and cited sources.',
  },
];

// ── What ─────────────────────────────────────────────────────────────────────
const WHAT_ITEMS = [
  {
    icon: '📋',
    label: 'Executive Summary',
    detail: 'A concise synthesis of the market landscape — ready to paste directly into a deck or memo.',
  },
  {
    icon: '🔍',
    label: 'Key Findings',
    detail: 'Numbered insights ranked by importance. Scannable in under 60 seconds.',
  },
  {
    icon: '🔗',
    label: 'Cited Sources',
    detail: 'Every claim backed by a live source URL. No black-box conclusions — full transparency.',
  },
  {
    icon: '⚡',
    label: 'Smart Caching',
    detail: 'Repeat queries are served from memory instantly. Fast and cost-efficient at scale.',
  },
];

// ── Shared primitives ─────────────────────────────────────────────────────────
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-bold font-mono uppercase tracking-[.16em] text-accent mb-3">
      {children}
    </p>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-display text-3xl sm:text-4xl font-bold text-ink-primary leading-tight">
      {children}
    </h2>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const { state, run, reset } = useResearch();
  const toolRef = useRef<HTMLElement>(null);

  const isIdle      = state.status === 'idle';
  const isStreaming = state.status === 'streaming';
  const isDone      = state.status === 'done';
  const isError     = state.status === 'error';

  function scrollToTool() {
    toolRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  return (
    <div className="min-h-screen">

      {isIdle && (
        <>
          {/* ── Hero ─────────────────────────────────────────────────────── */}
          <section className="relative px-4 sm:px-6 pt-20 pb-24 text-center overflow-hidden">
            {/* Radial glow behind headline */}
            <div
              className="absolute inset-0 pointer-events-none"
              style={{
                background:
                  'radial-gradient(ellipse 80% 50% at 50% -10%, rgba(125,211,252,0.07) 0%, transparent 70%)',
              }}
            />

            <div className="relative max-w-3xl mx-auto">
              {/* Pill badge */}
              <span className="inline-flex items-center gap-2 text-[10px] font-semibold font-mono uppercase tracking-widest text-accent bg-accent/10 border border-accent/20 rounded-full px-3 py-1 mb-7">
                <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-dot" />
                AI-Powered Market Intelligence
              </span>

              <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold text-ink-primary leading-[1.1] tracking-tight mb-5">
                Stop guessing.
                <br />
                <span className="text-accent">Start knowing.</span>
              </h1>

              <p className="text-base sm:text-lg text-ink-secondary leading-relaxed max-w-xl mx-auto mb-8">
                MarketMind is an AI agent that researches any market, brand, or
                trend and delivers an executive-ready report in seconds — not weeks.
              </p>

              <div className="flex items-center justify-center gap-3 flex-wrap">
                <button
                  onClick={scrollToTool}
                  className="px-6 py-3 bg-accent hover:bg-accent-light text-accent-subtle font-semibold rounded-lg transition active:scale-95"
                >
                  Try it free →
                </button>
                <button
                  onClick={scrollToTool}
                  className="px-6 py-3 text-ink-secondary hover:text-ink-primary border border-stroke hover:border-accent/40 rounded-lg transition font-medium"
                >
                  See how it works ↓
                </button>
              </div>

              {/* Social proof strip */}
              <div className="mt-12 flex items-center justify-center gap-6 flex-wrap">
                {['LangGraph', 'GPT-4o', 'pgvector', 'Tavily'].map((tech) => (
                  <span key={tech} className="text-xs text-ink-tertiary font-mono">
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </section>

          {/* ── Why ──────────────────────────────────────────────────────── */}
          <section className="px-4 sm:px-6 py-20 border-t border-stroke">
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-12">
                <SectionLabel>Why it matters</SectionLabel>
                <SectionHeading>Traditional market research is broken</SectionHeading>
                <p className="text-ink-secondary mt-4 max-w-lg mx-auto text-sm leading-relaxed">
                  The way companies gather market intelligence hasn't changed in decades.
                  MarketMind changes that.
                </p>
              </div>

              <div className="grid sm:grid-cols-3 gap-5">
                {WHY_ITEMS.map((item) => (
                  <div
                    key={item.title}
                    className="bg-bg-surface border border-stroke rounded-2xl p-6 hover:border-accent/30 transition group"
                  >
                    <div className="text-2xl mb-4">{item.icon}</div>
                    <h3 className="font-semibold text-ink-primary mb-2 group-hover:text-accent transition-colors">
                      {item.title}
                    </h3>
                    <p className="text-sm text-ink-secondary leading-relaxed">{item.body}</p>
                  </div>
                ))}
              </div>

              {/* Transition line */}
              <div className="mt-14 text-center">
                <p className="text-sm text-ink-tertiary font-mono">
                  There is a better way ↓
                </p>
              </div>
            </div>
          </section>

          {/* ── How ──────────────────────────────────────────────────────── */}
          <section className="px-4 sm:px-6 py-20 border-t border-stroke">
            <div className="max-w-3xl mx-auto">
              <div className="text-center mb-14">
                <SectionLabel>How it works</SectionLabel>
                <SectionHeading>Five steps. Seconds to complete.</SectionHeading>
                <p className="text-ink-secondary mt-4 max-w-lg mx-auto text-sm leading-relaxed">
                  A LangGraph-powered AI agent handles the full research pipeline —
                  from query to polished report — autonomously.
                </p>
              </div>

              <div className="space-y-2">
                {HOW_STEPS.map((s, i) => (
                  <div key={s.step} className="flex gap-5 items-start group">
                    {/* Step indicator + connector */}
                    <div className="flex flex-col items-center shrink-0 pt-1">
                      <div className="w-10 h-10 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center text-accent text-[11px] font-bold font-mono group-hover:bg-accent/20 transition-colors">
                        {s.step}
                      </div>
                      {i < HOW_STEPS.length - 1 && (
                        <div className="w-px h-10 bg-stroke mt-1" />
                      )}
                    </div>

                    {/* Content */}
                    <div className="pt-1.5 pb-6">
                      <h3 className="font-semibold text-ink-primary mb-1">{s.label}</h3>
                      <p className="text-sm text-ink-secondary leading-relaxed">{s.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ── What ─────────────────────────────────────────────────────── */}
          <section className="px-4 sm:px-6 py-20 border-t border-stroke">
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-12">
                <SectionLabel>What you get</SectionLabel>
                <SectionHeading>A complete intelligence report, instantly.</SectionHeading>
                <p className="text-ink-secondary mt-4 max-w-lg mx-auto text-sm leading-relaxed">
                  Every report is structured for immediate use — no editing, no formatting,
                  no waiting for someone to synthesise the findings for you.
                </p>
              </div>

              <div className="grid sm:grid-cols-2 gap-5">
                {WHAT_ITEMS.map((item) => (
                  <div
                    key={item.label}
                    className="flex gap-4 bg-bg-surface border border-stroke rounded-2xl p-6 hover:border-accent/30 transition group"
                  >
                    <div className="text-2xl shrink-0 mt-0.5">{item.icon}</div>
                    <div>
                      <h3 className="font-semibold text-ink-primary mb-1 group-hover:text-accent transition-colors">
                        {item.label}
                      </h3>
                      <p className="text-sm text-ink-secondary leading-relaxed">{item.detail}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ── CTA banner ───────────────────────────────────────────────── */}
          <section className="px-4 sm:px-6 py-16 border-t border-stroke">
            <div className="max-w-2xl mx-auto text-center">
              <SectionLabel>Ready to try it?</SectionLabel>
              <SectionHeading>Your first report in under 30 seconds.</SectionHeading>
              <p className="text-ink-secondary mt-4 mb-8 text-sm leading-relaxed">
                No signup. No credit card. Just enter a topic below and watch the agent work.
              </p>
              <button
                onClick={scrollToTool}
                className="px-7 py-3.5 bg-accent hover:bg-accent-light text-accent-subtle font-semibold rounded-lg transition active:scale-95 text-sm"
              >
                Run your first research →
              </button>
            </div>
          </section>
        </>
      )}

      {/* ── Tool ─────────────────────────────────────────────────────────── */}
      <section
        ref={toolRef}
        id="tool"
        className={`px-4 sm:px-6 py-10 ${isIdle ? 'border-t border-stroke' : ''}`}
      >
        <div className="max-w-3xl mx-auto space-y-4">

          {/* Compact heading shown only when active */}
          {!isIdle && (
            <div className="flex items-center justify-between mb-1">
              <div>
                <p className="text-[10px] font-bold font-mono uppercase tracking-[.16em] text-accent">
                  MarketMind
                </p>
                <p className="text-sm text-ink-tertiary">AI Market Research Agent</p>
              </div>
              {isDone && (
                <button
                  onClick={reset}
                  className="text-xs text-ink-tertiary hover:text-ink-primary border border-stroke hover:border-accent/40 rounded-lg px-3 py-1.5 transition"
                >
                  ← Back to home
                </button>
              )}
            </div>
          )}

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

        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      {isIdle && (
        <footer className="border-t border-stroke px-4 sm:px-6 py-6 text-center">
          <p className="text-xs text-ink-tertiary font-mono">
            Built with LangGraph · GPT-4o · pgvector · Tavily · Next.js
          </p>
        </footer>
      )}

    </div>
  );
}

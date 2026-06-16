'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ResearchResult } from '@/lib/types';

interface Props {
  topic: string;
  result: ResearchResult;
  onReset: () => void;
}

// Strip any residual LLM section-header lines that leaked through the parser.
const SECTION_HEADER_RE = /^\*{0,2}\s*(executive\s+summary|key\s+findings)\s*\*{0,2}:?\s*$/im;
function cleanContent(text: string): string {
  return text
    .split('\n')
    .filter(l => !SECTION_HEADER_RE.test(l.trim()))
    .join('\n')
    .trim();
}

function CopyIcon() {
  return (
    <svg className="w-3.5 h-3.5 shrink-0" viewBox="0 0 14 14" fill="none">
      <rect x="4.5" y="4.5" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
      <path d="M2.5 9.5H2A1.5 1.5 0 01.5 8V2A1.5 1.5 0 012 .5h6A1.5 1.5 0 019.5 2v1" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg className="w-3.5 h-3.5 shrink-0 text-success" viewBox="0 0 14 14" fill="none">
      <path d="M2 7l4 4 6-6" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PrintIcon() {
  return (
    <svg className="w-3.5 h-3.5 shrink-0" viewBox="0 0 14 14" fill="none">
      <path d="M4 4V1.5h6V4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      <rect x=".5" y="4" width="13" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
      <path d="M4 8h6M4 10h4" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
    </svg>
  );
}

function ExternalLinkIcon() {
  return (
    <svg className="w-3 h-3 shrink-0 opacity-60" viewBox="0 0 12 12" fill="none">
      <path d="M5 2H2a1 1 0 00-1 1v7a1 1 0 001 1h7a1 1 0 001-1V7" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
      <path d="M8 1h3v3M11 1L6.5 5.5" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function ResultsCard({ topic, result, onReset }: Props) {
  const [copied, setCopied] = useState(false);

  const summary  = cleanContent(result.executive_summary);
  const findings = result.key_findings
    .map(cleanContent)
    .filter(f => f.length > 0);

  async function copyReport() {
    const text = [
      `MARKET RESEARCH REPORT: ${topic}`,
      `Generated: ${new Date().toLocaleDateString('en-US', { dateStyle: 'long' })}`,
      `Source: ${result.reused_from_memory ? 'Memory cache' : 'Live web research'}`,
      '',
      'EXECUTIVE SUMMARY',
      '─'.repeat(40),
      summary,
      '',
      'KEY FINDINGS',
      '─'.repeat(40),
      findings.map((f, i) => `${i + 1}. ${f}`).join('\n'),
    ].join('\n');

    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2200);
  }

  const dateStr = new Date().toLocaleDateString('en-US', { dateStyle: 'long' });

  return (
    <div className="rounded-2xl border border-stroke shadow-sm overflow-hidden animate-fade-up print-card">

      {/* ── Report header ── */}
      <div className="bg-bg-raised px-6 py-5 border-b border-stroke">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-[.14em] text-accent font-mono mb-1.5">
              Market Research Report
            </p>
            <h2 className="text-xl font-bold font-display text-ink-primary leading-snug break-words">
              {topic}
            </h2>
            <p className="text-xs text-ink-tertiary font-mono mt-1.5">
              {dateStr}&nbsp;·&nbsp;
              {result.reused_from_memory ? 'Served from memory cache' : 'Live web research'}
            </p>
          </div>

          <span
            className={`text-xs font-semibold px-2.5 py-1 rounded-full shrink-0 font-mono ${
              result.reused_from_memory
                ? 'bg-accent-gold/10 text-accent-gold border border-accent-gold/20'
                : 'bg-success/10 text-success border border-success/20'
            }`}
          >
            {result.reused_from_memory ? '⚡ Cached' : '🔍 Fresh'}
          </span>
        </div>
      </div>

      {/* ── Report body ── */}
      <div className="bg-bg-surface px-6 py-7 space-y-8">

        {/* Executive Summary */}
        {summary && (
          <section>
            <SectionLabel>Executive Summary</SectionLabel>
            <div className="mt-4 pl-4 border-l-2 border-accent/30">
              <div className="prose-report">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{summary}</ReactMarkdown>
              </div>
            </div>
          </section>
        )}

        {/* Key Findings */}
        {findings.length > 0 && (
          <>
            {summary && <Divider />}
            <section>
              <SectionLabel>Key Findings</SectionLabel>
              <ol className="mt-4 space-y-4">
                {findings.map((finding, i) => (
                  <li key={i} className="flex gap-3.5 items-start group">
                    <span className="mt-0.5 w-5 h-5 shrink-0 flex items-center justify-center rounded-full bg-accent/10 text-accent text-[10px] font-bold font-mono ring-1 ring-accent/20 group-hover:bg-accent/20 transition-colors">
                      {i + 1}
                    </span>
                    <div className="prose-report flex-1 min-w-0 pt-px">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{finding}</ReactMarkdown>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          </>
        )}

        {/* Citations */}
        {result.citations.length > 0 && (
          <>
            <Divider />
            <section>
              <SectionLabel>Sources</SectionLabel>
              <ul className="mt-4 space-y-2">
                {result.citations.map((c, i) => (
                  <li key={i}>
                    <a
                      href={c.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-accent hover:text-accent-light transition-colors inline-flex items-center gap-1.5"
                    >
                      <ExternalLinkIcon />
                      {c.title || c.source_url}
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          </>
        )}
      </div>

      {/* ── Action bar ── */}
      <div className="no-print bg-bg-raised border-t border-stroke px-6 py-3.5 flex items-center gap-2 flex-wrap">
        <button
          onClick={copyReport}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-secondary hover:text-ink-primary bg-bg-surface hover:bg-bg-raised border border-stroke rounded-lg px-3 py-1.5 transition"
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? 'Copied!' : 'Copy Report'}
        </button>

        <button
          onClick={() => window.print()}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-secondary hover:text-ink-primary bg-bg-surface hover:bg-bg-raised border border-stroke rounded-lg px-3 py-1.5 transition"
        >
          <PrintIcon />
          Print
        </button>

        <button
          onClick={onReset}
          className="ml-auto inline-flex items-center gap-1.5 text-xs font-medium text-ink-tertiary hover:text-ink-primary hover:bg-bg-surface rounded-lg px-3 py-1.5 transition"
        >
          ← New Research
        </button>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <h3 className="text-[10px] font-bold uppercase tracking-[.14em] text-accent font-mono">
        {children}
      </h3>
      <div className="flex-1 h-px bg-stroke" />
    </div>
  );
}

function Divider() {
  return <hr className="border-stroke/60" />;
}

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
    <svg className="w-3.5 h-3.5 shrink-0 text-emerald-500" viewBox="0 0 14 14" fill="none">
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

  async function copyReport() {
    const findings = result.key_findings.map((f, i) => `${i + 1}. ${f}`).join('\n');
    const text = [
      `MARKET RESEARCH REPORT: ${topic}`,
      `Generated: ${new Date().toLocaleDateString('en-US', { dateStyle: 'long' })}`,
      `Source: ${result.reused_from_memory ? 'Memory cache' : 'Live web research'}`,
      '',
      'EXECUTIVE SUMMARY',
      '─'.repeat(40),
      result.executive_summary,
      '',
      'KEY FINDINGS',
      '─'.repeat(40),
      findings,
    ].join('\n');

    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2200);
  }

  const dateStr = new Date().toLocaleDateString('en-US', { dateStyle: 'long' });

  return (
    <div className="rounded-2xl border border-slate-200 shadow-sm overflow-hidden animate-fade-up print-card">
      {/* ── Report header (dark) ── */}
      <div className="bg-navy px-6 py-5">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div className="min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-[.14em] text-blue-400 mb-1.5">
              Market Research Report
            </p>
            <h2 className="text-xl font-bold text-white leading-snug break-words">
              {topic}
            </h2>
            <p className="text-xs text-slate-400 mt-1.5">
              {dateStr}&nbsp;·&nbsp;
              {result.reused_from_memory ? 'Served from memory cache' : 'Live web research'}
            </p>
          </div>

          <span
            className={`text-xs font-semibold px-2.5 py-1 rounded-full shrink-0 ${
              result.reused_from_memory
                ? 'bg-amber-400/10 text-amber-300 border border-amber-400/20'
                : 'bg-emerald-400/10 text-emerald-300 border border-emerald-400/20'
            }`}
          >
            {result.reused_from_memory ? '⚡ Cached' : '🔍 Fresh'}
          </span>
        </div>
      </div>

      {/* ── Report body ── */}
      <div className="bg-white px-6 py-6 space-y-7">
        {/* Executive Summary */}
        <section>
          <h3 className="text-[10px] font-bold uppercase tracking-[.14em] text-blue-600 mb-3">
            Executive Summary
          </h3>
          <div className="prose-report">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.executive_summary}
            </ReactMarkdown>
          </div>
        </section>

        {result.key_findings.length > 0 && (
          <>
            <hr className="border-slate-100" />
            <section>
              <h3 className="text-[10px] font-bold uppercase tracking-[.14em] text-blue-600 mb-3">
                Key Findings
              </h3>
              <ol className="space-y-3">
                {result.key_findings.map((finding, i) => (
                  <li key={i} className="flex gap-3 items-start">
                    <span className="mt-0.5 w-5 h-5 shrink-0 flex items-center justify-center rounded-full bg-blue-50 text-blue-600 text-[10px] font-bold">
                      {i + 1}
                    </span>
                    <div className="prose-report">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {finding}
                      </ReactMarkdown>
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          </>
        )}

        {result.citations.length > 0 && (
          <>
            <hr className="border-slate-100" />
            <section>
              <h3 className="text-[10px] font-bold uppercase tracking-[.14em] text-blue-600 mb-3">
                Sources
              </h3>
              <ul className="space-y-2">
                {result.citations.map((c, i) => (
                  <li key={i}>
                    <a
                      href={c.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:underline inline-flex items-center gap-1.5"
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
      <div className="no-print bg-slate-50 border-t border-slate-100 px-6 py-3.5 flex items-center gap-2 flex-wrap">
        <button
          onClick={copyReport}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 transition"
        >
          {copied ? <CheckIcon /> : <CopyIcon />}
          {copied ? 'Copied!' : 'Copy Report'}
        </button>

        <button
          onClick={() => window.print()}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 transition"
        >
          <PrintIcon />
          Print
        </button>

        <button
          onClick={onReset}
          className="ml-auto inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-lg px-3 py-1.5 transition"
        >
          ← New Research
        </button>
      </div>
    </div>
  );
}

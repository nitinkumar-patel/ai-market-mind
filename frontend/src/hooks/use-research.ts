'use client';

import { useCallback, useState } from 'react';
import type {
  ResearchRequest,
  ResearchResult,
  ResearchState,
  StepUpdate,
} from '@/lib/types';

const INITIAL: ResearchState = {
  status: 'idle',
  topic: '',
  depth: 'detailed',
  steps: [],
  result: null,
  error: null,
};

export function useResearch() {
  const [state, setState] = useState<ResearchState>(INITIAL);

  const run = useCallback(async (request: ResearchRequest) => {
    setState({
      status: 'streaming',
      topic: request.topic,
      depth: request.depth,
      steps: [],
      result: null,
      error: null,
    });

    try {
      const res = await fetch('/api/v1/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => `Status ${res.status}`);
        throw new Error(text || `Server error ${res.status}`);
      }
      if (!res.body) throw new Error('No response body');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Normalise \r\n → \n, then split on SSE event boundaries (\n\n)
        const parts = buffer.replace(/\r\n/g, '\n').split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          if (!part.trim()) continue;
          const dataLine = part.split('\n').find((l) => l.startsWith('data: '));
          if (!dataLine) continue;

          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(dataLine.slice(6));
          } catch {
            continue;
          }

          if ('step' in payload) {
            setState((prev) => ({
              ...prev,
              steps: [...prev.steps, payload as unknown as StepUpdate],
            }));
          } else if ('executive_summary' in payload) {
            setState((prev) => ({
              ...prev,
              status: 'done',
              result: payload as unknown as ResearchResult,
            }));
          }
        }
      }
    } catch (err) {
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: err instanceof Error ? err.message : 'An unexpected error occurred.',
      }));
    }
  }, []);

  const reset = useCallback(() => setState(INITIAL), []);

  return { state, run, reset };
}

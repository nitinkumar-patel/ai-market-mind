export type ResearchDepth = 'quick' | 'detailed';

export interface ResearchRequest {
  topic: string;
  depth: ResearchDepth;
}

export interface StepUpdate {
  step: string;
  message: string;
}

export interface Citation {
  source_url: string;
  title?: string | null;
}

export interface ResearchResult {
  executive_summary: string;
  key_findings: string[];
  citations: Citation[];
  reused_from_memory: boolean;
}

export type ResearchStatus = 'idle' | 'streaming' | 'done' | 'error';

export interface ResearchState {
  status: ResearchStatus;
  topic: string;
  depth: ResearchDepth;
  steps: StepUpdate[];
  result: ResearchResult | null;
  error: string | null;
}

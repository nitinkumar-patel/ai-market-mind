export function Header() {
  return (
    <header className="sticky top-0 z-20 h-14 bg-bg-surface border-b border-stroke flex items-center px-6 gap-3 shadow-sm">
      <span className="text-lg font-bold font-display text-ink-primary tracking-tight select-none">
        Market<span className="text-accent">Mind</span>
      </span>
      <span className="text-[10px] font-semibold uppercase tracking-widest text-accent bg-accent/10 border border-accent/20 rounded px-1.5 py-0.5 font-mono">
        AI Agent
      </span>
      <span className="ml-auto text-xs text-ink-tertiary hidden sm:block font-mono">
        LangGraph · pgvector · OpenAI
      </span>
    </header>
  );
}

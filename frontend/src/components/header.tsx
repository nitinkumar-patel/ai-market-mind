export function Header() {
  return (
    <header className="sticky top-0 z-20 h-14 bg-navy border-b border-white/5 flex items-center px-6 gap-3 shadow-sm">
      <span className="text-lg font-extrabold text-white tracking-tight select-none">
        Market<span className="text-blue-400">Mind</span>
      </span>
      <span className="text-[10px] font-bold uppercase tracking-widest text-blue-400 bg-blue-400/10 border border-blue-400/20 rounded px-1.5 py-0.5">
        AI Agent
      </span>
      <span className="ml-auto text-xs text-slate-500 hidden sm:block">
        LangGraph · pgvector · OpenAI
      </span>
    </header>
  );
}

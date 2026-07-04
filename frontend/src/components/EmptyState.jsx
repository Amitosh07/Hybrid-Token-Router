import { FiCpu } from 'react-icons/fi';

export default function EmptyState() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col items-center py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-xl border border-ink-800 bg-ink-900 text-accent">
        <FiCpu size={24} />
      </div>
      <h2 className="mt-6 text-3xl font-semibold tracking-normal text-zinc-50">Hybrid Token Router</h2>
      <p className="mt-3 max-w-xl text-base leading-7 text-ink-400">
        Type a prompt below. The router will analyse it, select the best model, and display live routing analytics.
      </p>
    </div>
  );
}

import {
  FiClock,
  FiCpu,
  FiTarget,
  FiZap,
} from 'react-icons/fi';
import AnalyticsCard from './AnalyticsCard.jsx';
import LoadingSpinner from './LoadingSpinner.jsx';
import {
  formatModel,
  formatMs,
  formatPercent,
} from '../services/formatters.js';

export default function AnalyticsSidebar({ analytics, isLoading }) {
  const provider = analytics?.provider;
  const modelTone = provider === 'local' ? 'success' : provider === 'remote' ? 'warning' : 'default';

  const items = [
    {
      icon: FiCpu,
      label: 'Current Model',
      value: formatModel(provider),
      tone: modelTone,
    },
    {
      icon: FiClock,
      label: 'Latency',
      value: formatMs(analytics?.latency),
      tone: 'accent',
    },
    {
      icon: FiTarget,
      label: 'Confidence',
      value: formatPercent(analytics?.confidence),
      tone: 'success',
    },
    {
      icon: FiZap,
      label: 'Routing Decision',
      value: analytics?.decision || 'No route yet',
      tone: provider === 'local' ? 'success' : provider === 'remote' ? 'warning' : 'default',
    },
  ];

  return (
    <aside className="hidden w-80 shrink-0 pt-4 xl:block">
      <section className="sticky top-[88px] rounded-xl border border-ink-800 bg-ink-900 p-4">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">Routing Analytics</h2>
            <p className="mt-1 text-xs text-ink-500">Live response metadata</p>
          </div>
          {isLoading && <LoadingSpinner size="sm" />}
        </div>
        <div className="grid gap-3">
          {items.map((item) => (
            <AnalyticsCard key={item.label} {...item} />
          ))}
        </div>
      </section>
    </aside>
  );
}

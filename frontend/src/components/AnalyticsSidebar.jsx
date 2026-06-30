import { FiActivity, FiClock, FiCpu, FiDollarSign, FiHash, FiTarget, FiZap } from 'react-icons/fi';
import AnalyticsCard from './AnalyticsCard.jsx';
import LoadingSpinner from './LoadingSpinner.jsx';
import { formatCurrency, formatModel, formatMs, formatNumber, formatPercent, formatSeconds } from '../services/formatters.js';

export default function AnalyticsSidebar({ analytics, isLoading }) {
  const model = analytics?.model;
  const modelTone = model === 'fireworks' ? 'warning' : model ? 'success' : 'default';
  const items = [
    { icon: FiCpu, label: 'Current Model', value: formatModel(model), tone: modelTone },
    { icon: FiClock, label: 'Latency', value: formatMs(analytics?.latency), tone: 'accent' },
    { icon: FiHash, label: 'Token Count', value: formatNumber(analytics?.tokens) },
    { icon: FiDollarSign, label: 'Estimated Cost', value: formatCurrency(analytics?.cost), tone: analytics?.cost > 0 ? 'warning' : 'success' },
    { icon: FiZap, label: 'Routing Decision', value: analytics?.decision || 'No route yet' },
    { icon: FiTarget, label: 'Confidence Score', value: formatPercent(analytics?.confidence), tone: 'success' },
    { icon: FiActivity, label: 'Processing Time', value: formatSeconds(analytics?.processingTime), tone: 'accent' }
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

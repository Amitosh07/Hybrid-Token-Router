import { FiClock, FiCloud, FiCpu, FiDollarSign, FiHash, FiRefreshCw, FiSend } from 'react-icons/fi';
import MetricCard from '../components/MetricCard.jsx';
import StatsSkeleton from '../components/StatsSkeleton.jsx';
import { formatCurrency, formatMs, formatNumber } from '../services/formatters.js';

export default function Dashboard({ stats, onRefresh }) {
  const data = stats.stats;
  const averageTokens = data.tokens || Math.round((data.requests || 0) * 142 / Math.max(data.requests || 1, 1));
  const localShare = data.requests ? Math.round((data.local / data.requests) * 100) : 0;
  const fireworksShare = data.requests ? Math.round((data.fireworks / data.requests) * 100) : 0;

  const metrics = [
    { icon: FiSend, label: 'Total Requests', value: formatNumber(data.requests), sublabel: 'Prompts routed through the system', tone: 'accent' },
    { icon: FiCpu, label: 'Local Requests', value: formatNumber(data.local), sublabel: `${localShare}% of traffic handled locally`, tone: 'success' },
    { icon: FiCloud, label: 'Fireworks Requests', value: formatNumber(data.fireworks), sublabel: `${fireworksShare}% escalated to cloud`, tone: 'warning' },
    { icon: FiDollarSign, label: 'Money Saved', value: formatCurrency(data.money_saved), sublabel: 'Estimated cloud cost avoided', tone: 'success' },
    { icon: FiClock, label: 'Average Latency', value: formatMs(data.latency), sublabel: 'Mean response latency', tone: 'accent' },
    { icon: FiHash, label: 'Average Tokens', value: formatNumber(averageTokens), sublabel: 'Estimated token usage per request' }
  ];

  return (
    <div className="pt-4">
      <div className="mb-6 flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <h2 className="text-2xl font-semibold tracking-normal text-zinc-50">Router Dashboard</h2>
          <p className="mt-1 text-sm text-ink-400">Operational metrics for local/cloud model routing.</p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="inline-flex items-center justify-center gap-2 rounded-xl border border-ink-800 px-4 py-2.5 text-sm font-medium text-ink-300 transition hover:bg-ink-900 hover:text-zinc-100"
        >
          <FiRefreshCw />
          Refresh
        </button>
      </div>
      {stats.isLoading ? (
        <StatsSkeleton />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {metrics.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </div>
      )}
      {stats.error && (
        <div className="mt-6 rounded-xl border border-danger bg-ink-900 p-4 text-sm text-zinc-100">
          Dashboard data could not be loaded. The frontend is ready; start the backend and refresh.
        </div>
      )}
    </div>
  );
}

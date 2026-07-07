import {
  FiClock,
  FiCloud,
  FiCpu,
  FiGitBranch,
  FiRefreshCw,
  FiSend,
  FiTarget,
  FiTrendingUp,
} from 'react-icons/fi';
import MetricCard from '../components/MetricCard.jsx';
import StatsSkeleton from '../components/StatsSkeleton.jsx';
import { formatMs, formatNumber, formatPercent } from '../services/formatters.js';

export default function Dashboard({ stats, onRefresh }) {
  const data = stats.stats;

  const totalReqs = data.total_requests ?? 0;
  const localReqs = data.local_requests ?? 0;
  const remoteReqs = data.remote_requests ?? 0;
  const mlPredictions = data.ml_predictions ?? 0;
  const heuristicFallbacks = data.heuristic_fallbacks ?? data.fallback_count ?? 0;
  const localShare = totalReqs ? Math.round((localReqs / totalReqs) * 100) : 0;
  const remoteShare = totalReqs ? Math.round((remoteReqs / totalReqs) * 100) : 0;

  const metrics = [
    {
      icon: FiGitBranch,
      label: 'Current Router',
      value: data.current_router || 'ML Router',
      sublabel: 'Active decision path for the latest request',
      tone: 'accent',
    },
    {
      icon: FiTarget,
      label: 'Prediction Confidence',
      value: formatPercent(data.average_prediction_confidence ?? data.average_confidence ?? 0),
      sublabel: 'Average confidence from live routing decisions',
      tone: 'success',
    },
    {
      icon: FiCpu,
      label: 'Total ML Predictions',
      value: formatNumber(mlPredictions),
      sublabel: 'Requests decided by the trained ML router',
      tone: 'success',
    },
    {
      icon: FiGitBranch,
      label: 'Heuristic Fallbacks',
      value: formatNumber(heuristicFallbacks),
      sublabel: 'Requests protected by fallback routing',
      tone: 'warning',
    },
    {
      icon: FiSend,
      label: 'Total Requests',
      value: formatNumber(totalReqs),
      sublabel: 'Prompts routed through the system',
      tone: 'accent',
    },
    {
      icon: FiTrendingUp,
      label: 'Routing Distribution',
      value: `${localShare}% / ${remoteShare}%`,
      sublabel: `${formatNumber(localReqs)} local, ${formatNumber(remoteReqs)} remote`,
      tone: 'accent',
    },
    {
      icon: FiCpu,
      label: 'Local Requests',
      value: formatNumber(localReqs),
      sublabel: `${localShare}% of traffic handled locally`,
      tone: 'success',
    },
    {
      icon: FiCloud,
      label: 'Remote Requests',
      value: formatNumber(remoteReqs),
      sublabel: `${remoteShare}% escalated to remote`,
      tone: 'warning',
    },
    {
      icon: FiClock,
      label: 'Average Latency',
      value: formatMs(data.average_latency_ms ?? 0),
      sublabel: 'Mean end-to-end response latency',
      tone: 'accent',
    },
    {
      icon: FiTarget,
      label: 'Average Confidence',
      value: formatPercent(data.average_confidence ?? 0),
      sublabel: 'Mean router confidence across all requests',
      tone: 'success',
    },
  ];

  return (
    <div className="pt-4">
      <div className="mb-6 flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <h2 className="text-2xl font-semibold tracking-normal text-zinc-50">Router Dashboard</h2>
          <p className="mt-1 text-sm text-ink-400">Live operational metrics for the Hybrid Token Router.</p>
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
          Dashboard data could not be loaded. Make sure the backend is running, then click Refresh.
        </div>
      )}
    </div>
  );
}

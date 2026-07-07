import { FiActivity, FiClock, FiCpu, FiHash, FiTarget } from 'react-icons/fi';
import { formatModel, formatMs, formatNumber, formatPercent } from '../services/formatters.js';

export default function ResponseFooter({ analytics }) {
  if (!analytics) return null;

  const items = [
    { icon: FiCpu, label: `Selected Model: ${formatModel(analytics.selected_provider || analytics.model)}` },
    { icon: FiTarget, label: `Prediction Confidence: ${formatPercent(analytics.prediction_confidence ?? analytics.confidence)}` },
    { icon: FiActivity, label: `Routing Method: ${analytics.routing_method || 'Heuristic Fallback'}` },
    { icon: FiClock, label: formatMs(analytics.latency) },
    { icon: FiHash, label: `${formatNumber(analytics.tokens)} tokens` }
  ];

  return (
    <div className="mt-4 flex flex-wrap gap-2 border-t border-ink-800 pt-3">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <span key={item.label} className="inline-flex items-center gap-2 rounded-lg border border-ink-800 px-2.5 py-1.5 text-xs text-ink-300">
            <Icon />
            {item.label}
          </span>
        );
      })}
    </div>
  );
}

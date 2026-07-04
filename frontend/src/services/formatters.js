/**
 * formatModel
 * Maps the backend `provider` field ("local" | "remote") to a display label.
 */
export function formatModel(provider) {
  if (!provider || provider === 'none') return 'Waiting';
  if (provider === 'local') return 'Local';
  if (provider === 'remote') return 'Remote';
  // Fallback: capitalise whatever string we receive
  return provider.charAt(0).toUpperCase() + provider.slice(1);
}

/**
 * formatMs
 * Backend returns latency already in milliseconds (chat.py multiplies perf_counter by 1000).
 * We just round and append "ms" — do NOT multiply again.
 */
export function formatMs(ms) {
  if (ms === null || ms === undefined) return '--';
  return `${Math.round(Number(ms))} ms`;
}

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return '--';
  return `${Number(seconds).toFixed(2)}s`;
}

export function formatCurrency(value) {
  if (value === null || value === undefined) return '$0.00';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 4, maximumFractionDigits: 6 }).format(Number(value));
}

export function formatNumber(value) {
  if (value === null || value === undefined) return '--';
  return new Intl.NumberFormat('en-US').format(Number(value));
}

export function formatPercent(value) {
  if (value === null || value === undefined) return '--';
  return `${Math.round(Number(value) * 100)}%`;
}

/** Format a routing score as a plain integer string. */
export function formatScore(value) {
  if (value === null || value === undefined) return '--';
  return String(Math.round(Number(value)));
}

/** Format a list of reason strings into a human-readable summary. */
export function formatReasons(reasons) {
  if (!reasons || !Array.isArray(reasons) || reasons.length === 0) return '--';
  return reasons.join(' · ');
}

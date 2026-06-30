export function formatModel(model) {
  if (!model) return 'Waiting';
  return model.toLowerCase() === 'fireworks' ? 'Fireworks' : 'Local';
}

export function formatMs(seconds) {
  if (seconds === null || seconds === undefined) return '--';
  return `${Math.round(Number(seconds) * 1000)} ms`;
}

export function formatSeconds(seconds) {
  if (seconds === null || seconds === undefined) return '--';
  return `${Number(seconds).toFixed(2)}s`;
}

export function formatCurrency(value) {
  if (value === null || value === undefined) return '$0.00';
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(Number(value));
}

export function formatNumber(value) {
  if (value === null || value === undefined) return '--';
  return new Intl.NumberFormat('en-US').format(Number(value));
}

export function formatPercent(value) {
  if (value === null || value === undefined) return '--';
  return `${Math.round(Number(value) * 100)}%`;
}

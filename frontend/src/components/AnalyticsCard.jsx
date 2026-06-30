export default function AnalyticsCard({ icon: Icon, label, value, tone = 'default' }) {
  const tones = {
    default: 'text-zinc-100',
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
    accent: 'text-accent'
  };

  return (
    <div className="rounded-xl border border-ink-800 bg-ink-950 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-ink-400">{label}</p>
        {Icon && <Icon className="shrink-0 text-ink-500" />}
      </div>
      <p className={`mt-3 truncate text-2xl font-semibold tracking-normal ${tones[tone]}`}>{value}</p>
    </div>
  );
}

import LoadingSpinner from './LoadingSpinner.jsx';

export default function MetricCard({ icon: Icon, label, value, sublabel, tone = 'default', isLoading }) {
  const tones = {
    default: 'text-zinc-100',
    success: 'text-success',
    warning: 'text-warning',
    accent: 'text-accent'
  };

  return (
    <div className="rounded-xl border border-ink-800 bg-ink-900 p-5">
      <div className="flex items-center justify-between gap-4">
        <p className="text-sm text-ink-400">{label}</p>
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-ink-800 bg-ink-950 text-ink-300">
          <Icon />
        </div>
      </div>
      <div className="mt-6">
        {isLoading ? <LoadingSpinner /> : <p className={`text-3xl font-semibold tracking-normal ${tones[tone]}`}>{value}</p>}
        {sublabel && <p className="mt-2 text-sm text-ink-500">{sublabel}</p>}
      </div>
    </div>
  );
}

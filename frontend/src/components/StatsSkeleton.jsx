export default function StatsSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
      {Array.from({ length: 6 }).map((_, index) => (
        <div key={index} className="h-40 animate-pulse rounded-xl border border-ink-800 bg-ink-900 p-5">
          <div className="h-4 w-28 rounded bg-ink-800" />
          <div className="mt-10 h-8 w-24 rounded bg-ink-800" />
          <div className="mt-3 h-3 w-36 rounded bg-ink-800" />
        </div>
      ))}
    </div>
  );
}

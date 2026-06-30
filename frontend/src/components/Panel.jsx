export default function Panel({ children, className = '' }) {
  return (
    <section className={`rounded-xl border border-ink-800 bg-ink-900 ${className}`}>
      {children}
    </section>
  );
}

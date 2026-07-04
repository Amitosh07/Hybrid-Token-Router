export default function Sidebar({ activePage, pages, onPageChange }) {
  return (
    <aside className="hidden w-64 shrink-0 border-r border-ink-800 pr-4 pt-4 lg:block">
      <nav className="space-y-2">
        {Object.entries(pages).map(([key, page]) => {
          const Icon = page.icon;
          const isActive = key === activePage;
          return (
            <button
              key={key}
              type="button"
              onClick={() => onPageChange(key)}
              className={`flex w-full items-center gap-3 rounded-xl border px-4 py-3 text-left text-sm transition ${
                isActive
                  ? 'border-accent bg-ink-900 text-zinc-50'
                  : 'border-transparent text-ink-400 hover:border-ink-800 hover:bg-ink-900 hover:text-zinc-100'
              }`}
            >
              <Icon size={18} />
              <span className="font-medium">{page.label}</span>
            </button>
          );
        })}
      </nav>

    </aside>
  );
}

export default function MobileTabs({ activePage, pages, onPageChange }) {
  return (
    <div className="grid grid-cols-2 gap-2 lg:hidden">
      {Object.entries(pages).map(([key, page]) => {
        const Icon = page.icon;
        const isActive = key === activePage;
        return (
          <button
            key={key}
            type="button"
            onClick={() => onPageChange(key)}
            className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-3 text-sm font-medium ${
              isActive ? 'border-accent bg-ink-900 text-zinc-50' : 'border-ink-800 text-ink-400'
            }`}
          >
            <Icon size={17} />
            {page.label}
          </button>
        );
      })}
    </div>
  );
}

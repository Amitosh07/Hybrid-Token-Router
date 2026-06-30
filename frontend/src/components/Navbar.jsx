import { FiActivity, FiCpu } from 'react-icons/fi';

export default function Navbar({ activePage }) {
  return (
    <header className="sticky top-0 z-30 border-b border-ink-800 bg-ink-950">
      <div className="mx-auto flex h-[72px] w-full max-w-[1720px] items-center justify-between px-4 lg:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-ink-800 bg-ink-900 text-accent">
            <FiCpu size={20} />
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-normal text-zinc-50">Hybrid Token Router</h1>
            <p className="text-sm text-ink-400">{activePage}</p>
          </div>
        </div>
        <div className="hidden items-center gap-2 rounded-xl border border-ink-800 bg-ink-900 px-3 py-2 text-sm text-ink-300 sm:flex">
          <FiActivity className="text-success" />
          Router online
        </div>
      </div>
    </header>
  );
}

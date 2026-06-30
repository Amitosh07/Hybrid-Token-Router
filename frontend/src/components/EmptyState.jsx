import { FiCpu, FiZap } from 'react-icons/fi';

export default function EmptyState({ onPrompt }) {
  const examples = [
    'Explain vector databases using a compact code example.',
    'Summarize token routing tradeoffs for a hackathon judge.',
    'Write a Python helper that estimates request cost.'
  ];

  return (
    <div className="mx-auto flex max-w-3xl flex-col items-center py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-xl border border-ink-800 bg-ink-900 text-accent">
        <FiCpu size={24} />
      </div>
      <h2 className="mt-6 text-3xl font-semibold tracking-normal text-zinc-50">Route prompts with intent.</h2>
      <p className="mt-3 max-w-xl text-base leading-7 text-ink-400">
        Ask once. The router decides whether the Local LLM or Fireworks AI should answer, then reports the routing analytics.
      </p>
      <div className="mt-8 grid w-full gap-3 md:grid-cols-3">
        {examples.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onPrompt(example)}
            className="rounded-xl border border-ink-800 bg-ink-900 p-4 text-left text-sm leading-6 text-ink-300 transition hover:border-accent hover:text-zinc-100"
          >
            <FiZap className="mb-3 text-warning" />
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}

import { useState } from 'react';
import { FiCheck, FiCopy } from 'react-icons/fi';

export default function CodeBlock({ language, children }) {
  const [copied, setCopied] = useState(false);
  const code = String(children).replace(/\n$/, '');

  const copyCode = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="my-4 overflow-hidden rounded-xl border border-ink-800 bg-ink-950">
      <div className="flex items-center justify-between border-b border-ink-800 px-4 py-2">
        <span className="text-xs font-medium uppercase text-ink-500">{language || 'code'}</span>
        <button
          type="button"
          onClick={copyCode}
          className="inline-flex items-center gap-2 rounded-lg px-2 py-1 text-xs text-ink-400 transition hover:bg-ink-900 hover:text-zinc-100"
        >
          {copied ? <FiCheck className="text-success" /> : <FiCopy />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 text-sm leading-6 text-zinc-200">
        <code>{code}</code>
      </pre>
    </div>
  );
}

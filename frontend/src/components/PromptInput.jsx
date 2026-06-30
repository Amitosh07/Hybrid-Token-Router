import { useEffect, useRef, useState } from 'react';
import { FiArrowUp, FiTrash2 } from 'react-icons/fi';
import LoadingSpinner from './LoadingSpinner.jsx';

export default function PromptInput({ onSend, isLoading, onClear, draft = '', onDraftConsumed }) {
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!draft) return;
    setValue(draft);
    onDraftConsumed?.();
    textareaRef.current?.focus();
  }, [draft, onDraftConsumed]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`;
  }, [value]);

  const submit = () => {
    if (!value.trim() || isLoading) return;
    onSend(value);
    setValue('');
  };

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="rounded-xl border border-ink-800 bg-ink-900 p-3 shadow-panel">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={onKeyDown}
        rows={3}
        placeholder="Enter a prompt for the hybrid router..."
        className="max-h-[220px] min-h-[96px] w-full resize-none bg-transparent px-2 py-2 text-base leading-7 text-zinc-100 outline-none placeholder:text-ink-500"
      />
      <div className="flex items-center justify-between gap-3 border-t border-ink-800 pt-3">
        <button
          type="button"
          onClick={onClear}
          className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm text-ink-400 transition hover:bg-ink-950 hover:text-zinc-100"
        >
          <FiTrash2 />
          Clear
        </button>
        <button
          type="button"
          onClick={submit}
          disabled={!value.trim() || isLoading}
          className="inline-flex h-11 min-w-11 items-center justify-center gap-2 rounded-xl bg-accent px-4 text-sm font-semibold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:bg-ink-800 disabled:text-ink-500"
          aria-label="Send prompt"
        >
          {isLoading ? <LoadingSpinner size="sm" /> : <FiArrowUp />}
          <span className="hidden sm:inline">Send</span>
        </button>
      </div>
    </div>
  );
}

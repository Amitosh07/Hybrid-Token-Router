import { useState } from 'react';
import { motion } from 'framer-motion';
import { FiCheck, FiCopy } from 'react-icons/fi';
import MarkdownContent from './MarkdownContent.jsx';
import ResponseFooter from './ResponseFooter.jsx';

export default function MessageBubble({ message }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const copyMessage = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <motion.article
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`group max-w-[min(860px,100%)] rounded-xl border p-5 ${
          isUser
            ? 'border-accent bg-accent text-white'
            : message.isError
              ? 'border-danger bg-ink-900 text-zinc-100'
              : 'border-ink-800 bg-ink-900 text-zinc-100'
        }`}
      >
        <div className={`prose-router ${isUser ? 'text-white' : 'text-zinc-100'}`}>
          {isUser ? <p className="whitespace-pre-wrap">{message.content}</p> : <MarkdownContent content={message.content} />}
        </div>
        {!isUser && (
          <div className="mt-4 flex items-center justify-between gap-3">
            <ResponseFooter analytics={message.analytics} />
            <button
              type="button"
              onClick={copyMessage}
              className="ml-auto inline-flex shrink-0 items-center gap-2 rounded-lg border border-ink-800 px-2.5 py-1.5 text-xs text-ink-300 transition hover:bg-ink-950 hover:text-zinc-100"
            >
              {copied ? <FiCheck className="text-success" /> : <FiCopy />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        )}
      </div>
    </motion.article>
  );
}

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import CodeBlock from './CodeBlock.jsx';

export default function MarkdownContent({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children }) {
          const match = /language-(\w+)/.exec(className || '');
          if (inline) {
            return <code className="rounded-md bg-ink-950 px-1.5 py-0.5 font-mono text-[0.9em] text-zinc-100">{children}</code>;
          }
          return <CodeBlock language={match?.[1]}>{children}</CodeBlock>;
        },
        p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5">{children}</ul>,
        ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5">{children}</ol>,
        a: ({ children, href }) => (
          <a href={href} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-4">
            {children}
          </a>
        ),
        table: ({ children }) => (
          <div className="my-4 overflow-x-auto rounded-xl border border-ink-800">
            <table className="w-full min-w-[520px] text-left text-sm">{children}</table>
          </div>
        ),
        th: ({ children }) => <th className="border-b border-ink-800 px-4 py-3 text-zinc-100">{children}</th>,
        td: ({ children }) => <td className="border-b border-ink-800 px-4 py-3 text-ink-300">{children}</td>
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

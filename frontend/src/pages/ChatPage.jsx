import { useCallback, useState } from 'react';
import { FiCpu, FiRefreshCw } from 'react-icons/fi';
import ChatWindow from '../components/ChatWindow.jsx';
import EmptyState from '../components/EmptyState.jsx';
import PromptInput from '../components/PromptInput.jsx';
import AnalyticsCard from '../components/AnalyticsCard.jsx';
import { formatModel, formatMs, formatNumber, formatCurrency } from '../services/formatters.js';

export default function ChatPage({ chat }) {
  const [draft, setDraft] = useState('');
  const hasConversation = chat.messages.length > 1;

  const usePrompt = useCallback((prompt) => {
    setDraft(prompt);
  }, []);

  const mobileAnalytics = [
    ['Model', formatModel(chat.analytics?.provider)],
    ['Latency', formatMs(chat.analytics?.latency)],
    ['Tokens', formatNumber(chat.analytics?.estimated_input_tokens)],
    ['Cost Saved', formatCurrency(chat.analytics?.estimated_cost_saved)],
    ['Routing Confidence', chat.analytics?.routing_confidence || '--'],
    ['Complexity', chat.analytics?.complexity
      ? chat.analytics.complexity.charAt(0).toUpperCase() + chat.analytics.complexity.slice(1)
      : '--']
  ];

  return (
    <div className="flex h-[calc(100vh-88px)] flex-col gap-4 pt-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold tracking-normal text-zinc-50">AI Routing Console</h2>
        </div>
        <button
          type="button"
          onClick={chat.clearMessages}
          className="hidden items-center gap-2 rounded-xl border border-ink-800 px-3 py-2 text-sm text-ink-400 transition hover:bg-ink-900 hover:text-zinc-100 sm:inline-flex"
        >
          <FiRefreshCw />
          Reset
        </button>
      </div>
      <div className="grid gap-3 xl:hidden">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          {mobileAnalytics.map(([label, value]) => (
            <AnalyticsCard key={label} icon={FiCpu} label={label} value={value} />
          ))}
        </div>
      </div>
      <section className="flex min-h-0 flex-1 flex-col rounded-xl border border-ink-800 bg-ink-950 p-4 md:p-5">
        {!hasConversation ? <EmptyState /> : <ChatWindow messages={chat.messages} isLoading={chat.isLoading} />}
      </section>
      <PromptInput
        onSend={chat.sendMessage}
        isLoading={chat.isLoading}
        onClear={chat.clearMessages}
        draft={draft}
        onDraftConsumed={() => setDraft('')}
      />
    </div>
  );
}

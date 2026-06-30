import { useCallback, useMemo, useState } from 'react';
import { sendChatMessage } from '../services/api.js';

const welcomeMessage = {
  id: 'welcome',
  role: 'assistant',
  content: 'Send a prompt to route it between the Local LLM and Fireworks AI. The response and routing analytics will appear here.',
  createdAt: new Date().toISOString(),
  analytics: null
};

export function useChat({ onToast }) {
  const [messages, setMessages] = useState([welcomeMessage]);
  const [analytics, setAnalytics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (prompt) => {
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || isLoading) return;

    const startedAt = performance.now();
    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: cleanPrompt,
      createdAt: new Date().toISOString()
    };

    setMessages((items) => [...items, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendChatMessage(cleanPrompt);
      const processingTime = (performance.now() - startedAt) / 1000;
      const routingAnalytics = {
        model: response.model,
        latency: response.latency,
        tokens: response.tokens,
        confidence: response.confidence,
        cost: response.cost,
        processingTime,
        decision: response.model === 'fireworks' ? 'Cloud route selected' : 'Local route selected'
      };

      setAnalytics(routingAnalytics);
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: response.answer || 'No response body returned.',
          createdAt: new Date().toISOString(),
          analytics: routingAnalytics
        }
      ]);
    } catch (err) {
      setError(err);
      onToast?.('Routing request failed. Check the backend and try again.', 'error');
      setMessages((items) => [
        ...items,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: 'I could not reach the router API. Please make sure the backend is running, then resend the prompt.',
          createdAt: new Date().toISOString(),
          isError: true
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, onToast]);

  const clearMessages = useCallback(() => {
    setMessages([welcomeMessage]);
    setAnalytics(null);
    setError(null);
  }, []);

  const lastAssistantMessage = useMemo(
    () => [...messages].reverse().find((message) => message.role === 'assistant' && !message.isError),
    [messages]
  );

  return { messages, analytics, isLoading, error, sendMessage, clearMessages, lastAssistantMessage };
}

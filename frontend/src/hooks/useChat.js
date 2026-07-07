import { useCallback, useMemo, useState } from 'react';
import { sendChatMessage } from '../services/api.js';

const welcomeMessage = {
  id: 'welcome',
  role: 'assistant',
  content: 'Send a prompt to get started. The router will decide whether to use the local model or the remote model and show live analytics.',
  createdAt: new Date().toISOString(),
  analytics: null
};

export function useChat({ onToast, onSuccess }) {
  const [messages, setMessages] = useState([welcomeMessage]);
  const [analytics, setAnalytics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (prompt) => {
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || isLoading) return;

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

      // Determine routing decision label from provider
      const provider = response.provider || 'unknown';
      const decisionLabel =
        provider === 'local'
          ? 'Local route selected'
          : provider === 'remote'
          ? 'Remote route selected'
          : `${provider} route selected`;

      const routingAnalytics = {
        // Core identity
        provider,
        selected_provider: response.selected_provider || provider,
        model: provider,
        // Latency — backend already sends milliseconds
        latency: response.latency,
        // Confidence in [0,1]
        confidence: response.prediction_confidence ?? response.confidence,
        prediction_probability: response.prediction_probability,
        prediction_confidence: response.prediction_confidence ?? response.confidence,
        model_version: response.model_version,
        routing_method: response.routing_method,
        // Routing engine output
        routing_score: response.routing_score,
        reason: response.reason,
        // Feature-extractor output
        task_type: response.task_type,
        complexity: response.complexity,
        estimated_input_tokens: response.estimated_input_tokens,
        // Decision label
        decision: decisionLabel,
        // Flags
        fallback_used: response.fallback_used,
        prompt_id: response.prompt_id,
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
      // Notify parent so dashboard stats can be refreshed without polling
      onSuccess?.();
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
  }, [isLoading, onToast, onSuccess]);

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

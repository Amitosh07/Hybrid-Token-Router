import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// Timeout must exceed the backend's worst-case processing time:
// Ollama attempt 1 (20s) + Ollama retry (20s) + remote LLM fallback (~10s) = ~50s max.
// We use 90s to give comfortable headroom without hanging indefinitely.
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 90000,
  headers: {
    'Content-Type': 'application/json'
  }
});

export async function sendChatMessage(prompt) {
  const { data } = await api.post('/chat', { prompt });
  return data;
}

export async function fetchStats() {
  const { data } = await api.get('/stats');
  return data;
}

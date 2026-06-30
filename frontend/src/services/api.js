import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000,
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

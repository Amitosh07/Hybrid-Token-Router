import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': {
        target: 'http://127.0.0.1:8000',
        // Must be > backend worst-case: Ollama attempt 1 (20s) + retry (20s) + remote (~10s) = ~50s
        timeout: 90000,
        proxyTimeout: 90000,
      },
      '/stats': {
        target: 'http://127.0.0.1:8000',
        timeout: 5000,
        proxyTimeout: 5000,
      },
    }
  }
});


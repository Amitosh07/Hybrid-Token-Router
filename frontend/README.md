# Hybrid Token Router Frontend

Production-style React dashboard for routing prompts between a Local LLM and Fireworks AI.

## Run locally

```bash
npm install
npm run dev
```

The Vite dev server runs at `http://127.0.0.1:5173`.

## API

The frontend calls:

- `POST /chat`
- `GET /stats`

During development, Vite proxies those requests to `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` if your backend runs elsewhere.

## Checks

```bash
npm run lint
npm run build
```

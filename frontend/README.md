# PDM Intelligence — Dashboard

React 19 + TypeScript + Tailwind operator dashboard for the PDM Intelligence
platform. Renders live fleet status, machine sensor detail, cost-optimal
decision scenarios and the ISA-18.2 / EU AI Act audit trail.

## Data flow

- REST via `src/api/client.ts` (`VITE_API_BASE`, default `/api/v1`)
- Live updates via the `/ws/live` WebSocket (`useLiveSnapshot` hook,
  exponential-backoff reconnect; the UI falls back to polling when the
  socket is down)

## Development

```bash
npm ci
npm run dev      # proxies /api and /ws to http://localhost:8000
npm run lint
npm run build
```

The production [Dockerfile](Dockerfile) builds the bundle and serves it through
nginx, proxying `/api`, `/docs` and `/ws` to the backend service (see
[nginx.conf](nginx.conf)).

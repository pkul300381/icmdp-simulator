// Drop null/undefined/empty params so the backend never receives the string 'null'.
const qs = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== '')).toString();

// Thin client for the FastAPI backend. Paths are relative; Vite proxies them in dev.
async function req(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'content-type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail ?? data);
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  meta: () => req('/api/meta'),
  intents: () => req('/api/intents'),
  parse: (body) => req('/api/intents/parse', { method: 'POST', body }),
  activate: (id, override = false) =>
    req(`/api/intents/${id}/activate`, { method: 'POST', body: { override } }),
  admissibility: (params) => req('/api/markov/admissibility?' + qs(params)),
  decision: (params) => req('/api/markov/decision?' + qs(params)),
  example51: () => req('/api/markov/example51'),
  run: (body) => req('/api/runs', { method: 'POST', body }),
  runs: () => req('/api/runs'),
  parallel: (body) => req('/api/parallel', { method: 'POST', body }),
  events: () => req('/api/events?limit=200'),
  metrics: () => fetch('/metrics').then((r) => r.text()),
};

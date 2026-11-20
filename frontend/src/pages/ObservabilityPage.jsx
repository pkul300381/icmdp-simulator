import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import EventLog from '../components/EventLog.jsx';

export default function ObservabilityPage() {
  const [events, setEvents] = useState([]);
  const [runs, setRuns] = useState([]);
  const [metrics, setMetrics] = useState('');
  const load = () => {
    api.events().then(setEvents).catch(() => {});
    api.runs().then(setRuns).catch(() => {});
    api.metrics().then(setMetrics).catch(() => {});
  };
  useEffect(() => { load(); const id = setInterval(load, 4000); return () => clearInterval(id); }, []);
  return (
    <>
      <section className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2>Intent lifecycle events</h2>
          <button onClick={load}>Refresh</button>
        </div>
        <div className="note" style={{ marginBottom: 8 }}>Append-only. Every parse, rejection, HOLD, override and activation is recorded with its reason.</div>
        <EventLog events={events} />
      </section>
      <section className="card">
        <h2>Runs</h2>
        <table className="data">
          <thead><tr><th>id</th><th>kind</th><th>status</th><th>created</th></tr></thead>
          <tbody>{runs.map((r) => <tr key={r.id}><td className="mono">{r.id}</td><td>{r.kind}</td><td><span className={`chip ${r.status === 'done' ? 'ok' : 'pending'}`}>{r.status}</span></td><td className="mono">{r.created_at}</td></tr>)}</tbody>
        </table>
      </section>
      <section className="card">
        <h2>Prometheus metrics (GET /metrics)</h2>
        <pre className="pre">{metrics || 'no metrics yet'}</pre>
      </section>
    </>
  );
}

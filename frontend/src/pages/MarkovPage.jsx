import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import DecisionGraph from '../components/DecisionGraph.jsx';
import ExampleGraph from '../components/ExampleGraph.jsx';

export default function MarkovPage({ meta }) {
  const [intents, setIntents] = useState([]);
  const [intentId, setIntentId] = useState(meta.presets.sev1);
  const [st, setSt] = useState({ severity: 'sev1', health: 'degraded', cpu: 85, latency: 300, replicas: 3, deploy_status: 'idle' });
  const [dec, setDec] = useState(null);
  const [ex, setEx] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => { api.intents().then(setIntents).catch(() => {}); api.example51().then(setEx).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    const params = { ...st, ...(intentId ? { intent_id: intentId } : {}) };
    api.decision(params).then(setDec).catch((e) => setError(e.message));
  }, [intentId, st.severity, st.health, st.cpu, st.latency, st.replicas, st.deploy_status]);

  const set = (k) => (e) => setSt((s) => ({ ...s, [k]: ['cpu', 'latency', 'replicas'].includes(k) ? +e.target.value : e.target.value }));

  return (
    <>
      <section className="card">
        <h2>Decision as a Markov graph</h2>
        <div className="note" style={{ marginBottom: 10 }}>
          One decision from state s: the admissible set restricts the choices, each choice leads to successor states
          with probabilities (edge width), and the planner's choice is starred. Switch the intent off to see the baseline MDP.
        </div>
        <div className="row" style={{ marginBottom: 10 }}>
          <label className="field">intent
            <select value={intentId ?? ''} onChange={(e) => setIntentId(e.target.value ? +e.target.value : null)}>
              <option value="">baseline (no intent)</option>
              {intents.filter((i) => ['validated', 'active', 'hold'].includes(i.status)).map((i) => (
                <option key={i.id} value={i.id}>#{i.id} {i.name || i.instruction.slice(0, 40)}</option>
              ))}
            </select>
          </label>
          <label className="field">severity<select value={st.severity} onChange={set('severity')}>{meta.severity.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">health<select value={st.health} onChange={set('health')}>{meta.health.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">cpu %<select value={st.cpu} onChange={set('cpu')}>{meta.cpu_grid.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">latency ms<select value={st.latency} onChange={set('latency')}>{meta.latency_grid.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">replicas<select value={st.replicas} onChange={set('replicas')}>{meta.replicas_grid.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">deploy<select value={st.deploy_status} onChange={set('deploy_status')}>{meta.deploy_status.map((x) => <option key={x}>{x}</option>)}</select></label>
        </div>
        {error && <div className="error-box">{error}</div>}
        <DecisionGraph data={dec} />
      </section>

      <section className="card">
        <h2>Why planning must see the constraint (Paper Example V.1)</h2>
        <div className="note" style={{ marginBottom: 10 }}>
          Same 3-state MDP, two planners. The runtime gate alone cannot repair the upstream choice: it blocks Deploy at s1
          after the planner has already committed to Risky at s0.
        </div>
        <ExampleGraph data={ex} />
      </section>
    </>
  );
}

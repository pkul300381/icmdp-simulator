import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import Stepper from '../components/Stepper.jsx';
import AdmissibilityGrid from '../components/AdmissibilityGrid.jsx';

const STATE_OPTS = (meta) => ({
  severity: meta.severity, health: meta.health, deploy_status: meta.deploy_status,
});

export default function IntentPage({ meta, onChange }) {
  const [instruction, setInstruction] = useState(meta.preset_instructions.sev1);
  const [parser, setParser] = useState('local');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [adm, setAdm] = useState(null);
  const [view, setView] = useState({ intent_id: meta.active_intent?.id ?? meta.presets.sev1, severity: 'sev1', health: 'degraded', deploy_status: 'idle', replicas: 3 });
  const [intents, setIntents] = useState([]);

  const refreshIntents = () => api.intents().then(setIntents).catch(() => {});
  useEffect(() => { refreshIntents(); }, []);

  useEffect(() => {
    api.admissibility(view).then(setAdm).catch((e) => setError(e.message));
  }, [view.intent_id, view.severity, view.health, view.deploy_status, view.replicas]);

  async function parse() {
    setBusy(true); setError(null);
    try {
      const r = await api.parse({ instruction, parser });
      setResult(r);
      refreshIntents();
      setView((v) => ({ ...v, intent_id: r.intent_id }));
    } catch (e) { setError(e.message); }
    setBusy(false);
  }

  async function activate(override = false) {
    if (!result) return;
    setBusy(true); setError(null);
    try {
      const a = await api.activate(result.intent_id, override);
      setResult({
        ...result,
        status: 'active',
        activated: a,
        stages: result.stages.map((st) => (st.name === 'activate'
          ? { ...st, status: 'ok', detail: `active as rule v${a.version}` }
          : st)),
      });
      refreshIntents(); onChange();
    } catch (e) { setError(e.message); }
    setBusy(false);
  }

  const setField = (k) => (e) => setView((v) => ({ ...v, [k]: k === 'replicas' ? +e.target.value : e.target.value }));
  const opts = STATE_OPTS(meta);

  return (
    <>
      <section className="card">
        <h2>1 · Compile a natural-language intent</h2>
        <div className="note" style={{ marginBottom: 10 }}>
          The parser proposes; the deterministic validator decides. A rejected or infeasible rule never reaches planning.
          Activation is atomic and versioned.
        </div>
        <div className="row" style={{ marginBottom: 8 }}>
          {Object.entries(meta.preset_instructions).map(([k, text]) => (
            <button key={k} onClick={() => setInstruction(text)}>{k}</button>
          ))}
        </div>
        <textarea value={instruction} onChange={(e) => setInstruction(e.target.value)} aria-label="instruction" />
        <div className="row" style={{ marginTop: 8 }}>
          <label className="field">parser
            <select value={parser} onChange={(e) => setParser(e.target.value)}>
              <option value="local">local grammar (offline)</option>
              <option value="llm">LLM (needs ANTHROPIC_API_KEY)</option>
            </select>
          </label>
          <button className="primary" onClick={parse} disabled={busy || !instruction.trim()}>Parse → validate → compile</button>
          {result && (result.status === 'validated' || result.status === 'hold') && (
            <button onClick={() => activate(false)} disabled={busy || result.status !== 'validated'}>Activate atomically</button>
          )}
          {result && result.status === 'hold' && (
            <button onClick={() => activate(true)} disabled={busy} title="Explicit override of a HOLD; logged">Force activate (logged override)</button>
          )}
        </div>
        {error && <div className="error-box" style={{ marginTop: 8 }}>{error}</div>}
        {result && (
          <div style={{ marginTop: 14 }}>
            <Stepper stages={result.stages} />
            {result.activated && <div className="note" style={{ marginTop: 10 }}>Activated as rule v{result.activated.version}.</div>}
            {result.spec && <pre className="pre" style={{ marginTop: 10 }}>{JSON.stringify(result.spec, null, 2)}</pre>}
          </div>
        )}
      </section>

      <section className="card">
        <h2>2 · What the rule permits, state by state</h2>
        <div className="row" style={{ marginBottom: 10 }}>
          <label className="field">intent
            <select value={view.intent_id ?? ''} onChange={(e) => setView((v) => ({ ...v, intent_id: e.target.value ? +e.target.value : null }))}>
              <option value="">baseline (no intent)</option>
              {intents.filter((i) => i.status === 'validated' || i.status === 'active' || i.status === 'hold').map((i) => (
                <option key={i.id} value={i.id}>#{i.id} {i.name || i.instruction.slice(0, 40)} [{i.status}]</option>
              ))}
            </select>
          </label>
          <label className="field">severity<select value={view.severity} onChange={setField('severity')}>{opts.severity.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">health<select value={view.health} onChange={setField('health')}>{opts.health.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">deploy<select value={view.deploy_status} onChange={setField('deploy_status')}>{opts.deploy_status.map((x) => <option key={x}>{x}</option>)}</select></label>
          <label className="field">replicas<select value={view.replicas} onChange={setField('replicas')}>{meta.replicas_grid.map((x) => <option key={x}>{x}</option>)}</select></label>
        </div>
        <AdmissibilityGrid data={adm} />
        <div className="note" style={{ marginTop: 10 }}>
          Each cell is a grid state (cpu × latency). Green ✓ actions are in A_I(s) = A₀(s) ∩ I(s); red ✕ actions are available
          but excluded by intent; grey – actions are physically unavailable in that state.
        </div>
      </section>
    </>
  );
}

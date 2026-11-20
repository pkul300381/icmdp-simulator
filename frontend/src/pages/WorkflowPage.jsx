import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import Lanes from '../components/Lanes.jsx';
import { LANE_COLOR } from '../palette.js';

function Pipeline({ title, color, stages, note }) {
  return (
    <div className="card" style={{ borderTop: `3px solid ${color}` }}>
      <h2 style={{ color }}>{title}</h2>
      <div className="note" style={{ marginBottom: 8 }}>{note}</div>
      <table className="data">
        <tbody>
          {stages.map((s, i) => (
            <tr key={s.name}>
              <td className="muted" style={{ width: 28 }}>{i + 1}</td>
              <td><strong>{s.name}</strong></td>
              <td className="secondary">{s.detail}</td>
              <td className="num mono">{s.seconds != null ? `${(s.seconds * 1000).toFixed(1)} ms` : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Parallel workflow: ICMDP (validate, compile, restrict, gate) against CMDP (cost, budget, dual search, policy).
// Two scenarios. 'snap' exposes the grid-snapping leak: raw cpu 75 snaps to 70, so a CMDP policy
// computed on the grid can propose ScaleIn while the raw rule ("cpu > 70") forbids it. 'incident'
// is the Sev1 deploy ban, where both lanes avoid the prohibited action.
const SCENARIOS = {
  snap: { label: 'CPU 75% (snapping case)', intent: 'cpuScaleIn',
          start: { severity: 'none', health: 'healthy', cpu: 75, latency: 300, replicas: 4, deploy_status: 'idle' },
          why: 'Rule: never scale in above 70% CPU. The raw state is 75%, but the grid snaps it to 70%. The ICMDP gate evaluates 75%; the CMDP lane has no gate and acts on the snapped state. Its higher return here comes from breaking the rule, which is the value the constraint gives up.' },
  incident: { label: 'Sev1 incident (deploy ban)', intent: 'sev1',
              start: { severity: 'sev1', health: 'degraded', cpu: 85, latency: 300, replicas: 3, deploy_status: 'idle' },
              why: 'Rule: no deploy during a Sev1 incident. Both planners avoid Deploy here, so the lanes mostly agree; the difference is in the workflow, not the outcome.' },
};

export default function WorkflowPage({ meta }) {
  const [scenario, setScenario] = useState('snap');
  const [seed, setSeed] = useState(3);
  const [horizon, setHorizon] = useState(20);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);


  async function run(sc = scenario) {
    const cfg = SCENARIOS[sc];
    setBusy(true); setError(null);
    try { setData(await api.parallel({ intent_id: meta.presets[cfg.intent], seed, horizon, start: cfg.start })); }
    catch (e) { setError(e.message); }
    setBusy(false);
  }
  const pick = (sc) => { setScenario(sc); run(sc); };
  useEffect(() => { run('snap'); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <section className="card">
        <h2>Two workflows, one episode</h2>
        <div className="note" style={{ marginBottom: 10 }}>
          Both lanes start from the same state and consume the same random draws, so differences come from the workflow, not the luck.
          The ICMDP lane enforces the intent at two points (planning and the runtime gate). The CMDP lane encodes the intent as a cost with a zero
          budget and must find a multiplier λ by dual search; it has no runtime filter.
        </div>
        <div className="row" style={{ marginBottom: 8 }}>
          {Object.entries(SCENARIOS).map(([k, cfg]) => (
            <button key={k} className={scenario === k ? 'primary' : ''} onClick={() => pick(k)} disabled={busy}>{cfg.label}</button>
          ))}
        </div>
        <div className="note" style={{ marginBottom: 10 }}>{SCENARIOS[scenario].why}</div>
        <div className="row">
          <label className="field">seed<input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} style={{ width: 90 }} /></label>
          <label className="field">horizon<input type="number" value={horizon} min={4} max={60} onChange={(e) => setHorizon(+e.target.value)} style={{ width: 90 }} /></label>
          <button className="primary" onClick={() => run()} disabled={busy}>{busy ? 'Running…' : 'Replay with these settings'}</button>
        </div>
        {error && <div className="error-box" style={{ marginTop: 8 }}>{error}</div>}
      </section>

      {data && (
        <>
          <div className="grid2">
            <Pipeline title="ICMDP workflow" color={LANE_COLOR.icmdp} stages={data.icmdp.stages}
                      note="Untrusted parse, then deterministic validation, compile and feasibility, then planning restricted to A_I(s), then a gate on every raw state." />
            <Pipeline title="CMDP workflow" color={LANE_COLOR.cmdp} stages={data.cmdp.stages}
                      note="Same intent as a cost with a zero budget. Cost is only controlled in expectation, and the dual search must converge before a policy exists." />
          </div>
          <section className="card">
            <h2>CMDP dual search (multiplier λ)</h2>
            <table className="data">
              <thead><tr><th>λ</th><th className="num">worst discounted cost</th><th className="num">sweeps</th><th className="num">seconds</th></tr></thead>
              <tbody>
                {data.cmdp.dual_trace.map((d) => (
                  <tr key={d.lambda}><td>{d.lambda}</td><td className="num">{d.worst_cost.toExponential(2)}</td><td className="num">{d.iterations}</td><td className="num">{d.seconds.toFixed(3)}</td></tr>
                ))}
              </tbody>
            </table>
            <div className="note" style={{ marginTop: 8 }}>
              Each λ is a full re-plan. Under a changed intent, this entire search repeats; the ICMDP lane instead activates the new rule and filters at the next decision.
            </div>
          </section>
          <Lanes data={data} />
        </>
      )}
    </>
  );
}

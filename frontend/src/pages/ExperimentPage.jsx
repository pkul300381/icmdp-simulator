import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import { ArmBars, toArmRows, SweepChart } from '../components/ArmCharts.jsx';
import OccupancyChart from '../components/OccupancyChart.jsx';
import { ARM_COLOR } from '../palette.js';

export default function ExperimentPage({ meta }) {
  const [kind, setKind] = useState('six_arm');
  const [intentId, setIntentId] = useState(meta.presets.sev1);
  const [episodes, setEpisodes] = useState(500);
  const [seed, setSeed] = useState(1);
  const [horizon, setHorizon] = useState(20);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [run, setRun] = useState(null);
  const [occArm, setOccArm] = useState('6_combined');

  // Run the default experiment on first view so the page is never empty.
  useEffect(() => { go(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function go() {
    setBusy(true); setError(null);
    try {
      const body = { kind, episodes, seed, horizon, trace_sample: 3 };
      if (kind !== 'freeze') body.intent_id = intentId;
      setRun(await api.run(body));
    } catch (e) { setError(e.message); }
    setBusy(false);
  }

  const s = run?.summary;
  return (
    <>
      <section className="card">
        <h2>Run a reproducible experiment</h2>
        <div className="note" style={{ marginBottom: 10 }}>
          Episodes share random draws across arms (common random numbers). Every arm is scored against the user's instruction,
          so a stale filter is measured against what was asked. Results and decision traces are stored in SQLite.
        </div>
        <div className="row">
          <label className="field">experiment
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="six_arm">six-arm comparison (Paper V.C)</option>
              <option value="penalty_sweep">finite penalty sweep</option>
              <option value="freeze">mid-episode freeze update (Paper IV-D)</option>
            </select>
          </label>
          {kind !== 'freeze' && (
            <label className="field">intent
              <select value={intentId} onChange={(e) => setIntentId(+e.target.value)}>
                {Object.entries(meta.presets).map(([k, id]) => <option key={id} value={id}>{k}</option>)}
              </select>
            </label>
          )}
          <label className="field">episodes
            <select value={episodes} onChange={(e) => setEpisodes(+e.target.value)}>
              <option value={200}>200</option><option value={500}>500</option><option value={2160}>all 2160 grid states</option>
            </select>
          </label>
          <label className="field">seed<input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} style={{ width: 80 }} /></label>
          <label className="field">horizon<input type="number" value={horizon} min={5} max={60} onChange={(e) => setHorizon(+e.target.value)} style={{ width: 80 }} /></label>
          <button className="primary" onClick={go} disabled={busy}>{busy ? 'Running…' : 'Run'}</button>
        </div>
        {error && <div className="error-box" style={{ marginTop: 8 }}>{error}</div>}
      </section>

      {s && run.kind === 'six_arm' && (
        <>
          <div className="note">
            Intent {s.intent} · {s.episodes} episodes · seed {s.seed} · horizon {s.horizon} · zero-budget λ = {s.zero_budget.lambda}
            {' '}(worst violation cost {s.zero_budget.worst_cost.toExponential(1)}; {s.zero_budget.satisfied ? 'satisfied' : 'NOT satisfied'}) ·
            {' '}infeasible states: {s.infeasible_states}
          </div>
          <div className="grid2">
            <ArmBars title="Discounted return (± 95% CI)" unit="return" rows={toArmRows(s.summaries)} valueKey="return" />
            <ArmBars title="Violations per 100 executed steps" rows={toArmRows(s.summaries)} valueKey="violations" />
          </div>
          <section className="card">
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <h2>State occupancy over time</h2>
              <label className="field">arm
                <select value={occArm} onChange={(e) => setOccArm(e.target.value)}>
                  {s.summaries.map((x) => <option key={x.arm} value={x.arm}>{x.arm}</option>)}
                </select>
              </label>
            </div>
            <OccupancyChart title={`Share of episodes in each severity state · ${occArm}`} occupancy={s.occupancy[occArm]} />
          </section>
          <section className="card">
            <h2>CMDP dual search for this intent</h2>
            <table className="data">
              <thead><tr><th>λ</th><th className="num">worst cost</th><th className="num">sweeps</th><th className="num">seconds</th></tr></thead>
              <tbody>{s.cmdp_dual_trace.map((d) => (
                <tr key={d.lambda}><td>{d.lambda}</td><td className="num">{d.worst_cost.toExponential(2)}</td><td className="num">{d.iterations}</td><td className="num">{d.seconds.toFixed(3)}</td></tr>
              ))}</tbody>
            </table>
          </section>
          <section className="card">
            <h2>Arms, full numbers</h2>
            <table className="data">
              <thead><tr><th>arm</th><th>planner</th><th className="num">return ±CI</th><th className="num">viol / 100</th><th className="num">viol. episodes %</th><th className="num">clarify %</th><th className="num">op. errors</th></tr></thead>
              <tbody>{s.summaries.map((x) => (
                <tr key={x.arm}>
                  <td><span style={{ color: ARM_COLOR[x.arm] }}>■</span> {x.arm}</td><td className="secondary">{x.planner}</td>
                  <td className="num">{x.return.mean.toFixed(2)} ±{x.return.ci95.toFixed(2)}</td>
                  <td className="num">{x.violations_per_100_steps.toFixed(2)}</td>
                  <td className="num">{x.violating_episode_pct.toFixed(1)}</td>
                  <td className="num">{x.clarification_pct.toFixed(1)}</td>
                  <td className="num">{x.operational_errors}</td>
                </tr>))}</tbody>
            </table>
          </section>
        </>
      )}

      {s && run.kind === 'penalty_sweep' && <SweepChart sweep={s.sweep} />}

      {s && run.kind === 'freeze' && (
        <>
          <div className="note">Intent changes at step {s.switch_step}. CMDP re-solve took {s.cmdp_resolve_seconds.toFixed(3)} s over {s.cmdp_dual_iterations} λ values; the ICMDP filter activates at the next decision.</div>
          <div className="grid2">
            <ArmBars title="Violations per 100 steps (whole episode)" rows={toArmRows(s.summaries, true)} valueKey="violations" />
            <ArmBars title="Discounted return (± 95% CI)" unit="return" rows={toArmRows(s.summaries, true)} valueKey="return" />
          </div>
        </>
      )}
    </>
  );
}

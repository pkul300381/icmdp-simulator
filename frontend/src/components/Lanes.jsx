import React, { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts';
import { LANE_COLOR, GRID, INK2, INK3, SURFACE, SEV } from '../palette.js';

const GAMMA = 0.9;
const KIND = { ok: ['ok', '✓ executed'], filtered: ['filtered', '▲ gate replaced'], violation: ['violation', '✕ violation'], hold: ['hold', '⚠ hold'] };

function laneSeries(steps) {
  let cum = 0;
  return steps.map((s) => {
    if (s.reward != null) cum += Math.pow(GAMMA, s.t) * s.reward;
    return { t: s.t, cum };
  });
}

function Lane({ name, color, lane, t, onPick, admLabel }) {
  const cur = lane.steps.find((s) => s.t === t);
  const series = useMemo(() => laneSeries(lane.steps), [lane]);
  return (
    <div className="card" style={{ borderLeft: `3px solid ${color}` }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h2 style={{ color }}>{name}</h2>
        <span className="muted" style={{ fontSize: 12 }}>total discounted return {lane.total_return.toFixed(2)} · violations {lane.violations}</span>
      </div>
      <div className="lane-track" aria-label={`${name} severity over time`}>
        {lane.steps.map((s) => (
          <div key={s.t} className={`lane-cell sev-${s.state.severity} ${s.t === t ? 'current' : ''}`}
               title={`t=${s.t} ${s.state.severity}`} onClick={() => onPick(s.t)}>
            {s.t}
          </div>
        ))}
      </div>
      <div className="legend" style={{ marginTop: 4 }}>
        <span style={{ '--c': SEV.none }}>no incident</span>
        <span style={{ '--c': SEV.sev2 }}>sev2</span>
        <span style={{ '--c': SEV.sev1 }}>sev1</span>
      </div>
      {cur ? (
        <table className="data" style={{ marginTop: 8 }}>
          <tbody>
            <tr><th>state at t={cur.t}</th><td className="mono">{cur.state.severity} · {cur.state.health} · {cur.state.deploy_status} · cpu {cur.state.cpu} · r{cur.state.replicas}</td></tr>
            <tr><th>{admLabel}</th><td className="mono">{cur.admissible.length ? cur.admissible.join(', ') : '∅ (HOLD)'}</td></tr>
            <tr><th>proposal → executed</th><td className="mono">{cur.proposal ?? '—'} → {cur.executed ?? '—'}</td></tr>
            <tr><th>outcome</th><td><span className={`chip ${KIND[cur.kind]?.[0] ?? 'ok'}`}>{KIND[cur.kind]?.[1] ?? cur.kind}</span>
              <span className="muted" style={{ marginLeft: 8, fontSize: 12 }}>{cur.event}</span></td></tr>
          </tbody>
        </table>
      ) : <div className="muted" style={{ marginTop: 8 }}>episode ended at t={lane.steps.length}</div>}
      <div style={{ width: '100%', height: 110, marginTop: 6 }}>
        <ResponsiveContainer>
          <LineChart data={series} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke={GRID} />
            <XAxis dataKey="t" tick={{ fill: INK3, fontSize: 10 }} stroke={GRID} />
            <YAxis tick={{ fill: INK3, fontSize: 10 }} stroke={GRID} width={40} />
            <ReferenceLine x={t} stroke={INK2} strokeDasharray="0" />
            <Tooltip contentStyle={{ background: SURFACE, border: `1px solid ${GRID}`, color: INK2, fontSize: 12 }}
                     formatter={(v) => v.toFixed(2)} labelFormatter={(l) => `t=${l}`} />
            <Line type="linear" dataKey="cum" stroke={color} strokeWidth={1.8} dot={false} name="cumulative return" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// Two workflows replay the same start and the same random draws, step by step.
export default function Lanes({ data }) {
  const maxT = Math.max(data.icmdp.steps.length, data.cmdp.steps.length) - 1;
  const [t, setT] = useState(0);
  return (
    <div className="col" style={{ gap: 12 }}>
      <div className="row">
        <label className="field" style={{ flex: 1, minWidth: 260 }}>
          step t = {t} of {maxT}
          <input type="range" min={0} max={Math.max(maxT, 0)} value={t} onChange={(e) => setT(+e.target.value)} />
        </label>
        <button onClick={() => setT((v) => Math.max(0, v - 1))}>◀ prev</button>
        <button onClick={() => setT((v) => Math.min(maxT, v + 1))}>next ▶</button>
      </div>
      <div className="grid2">
        <Lane name="ICMDP lane (intent-constrained)" color={LANE_COLOR.icmdp} lane={data.icmdp} t={t} onPick={setT} admLabel="admissible A_I(s) (gate uses this)" />
        <Lane name="CMDP lane (zero-budget)" color={LANE_COLOR.cmdp} lane={data.cmdp} t={t} onPick={setT} admLabel="available A₀(s) (no filter applied)" />
      </div>
    </div>
  );
}

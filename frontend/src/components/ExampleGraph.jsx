import React from 'react';
import { INK, INK2, INK3, SLOT, SURFACE, STATUS } from '../palette.js';

// Paper Example V.1 as a 3-state Markov graph. Left: planner ignores the intent (picks Risky, gate
// later blocks Deploy). Right: constraint-aware planner sees the downstream exclusion and picks Safe.
const POS = { s0: [90, 150], s1: [360, 60], T: [640, 150] };

function Edge({ from, to, label, chosen, prohibited, offset = 0, color }) {
  const [x1, y1] = POS[from], [x2, y2] = POS[to];
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 + offset;
  const d = `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
  const stroke = prohibited ? STATUS.excluded : color;
  return (
    <g>
      <path d={d} fill="none" stroke={stroke} strokeWidth={chosen ? 2.5 : 1.2}
            strokeDasharray={prohibited ? '6 4' : undefined} opacity={chosen || prohibited ? 1 : 0.5}
            />
      <text x={mx} y={my - 6 + offset * 0.1} textAnchor="middle" fill={prohibited ? STATUS.excluded : INK2} fontSize={11}>
        {label}
      </text>
    </g>
  );
}

function Panel({ title, color, edges, note, subtitle }) {
  return (
    <div className="card" style={{ padding: 10 }}>
      <h3 style={{ color }}>{title}</h3>
      <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>{subtitle}</div>
      <svg viewBox="0 0 740 240" width="100%">
        {edges}
        {Object.entries(POS).map(([k, [x, y]]) => (
          <g key={k}>
            <circle cx={x} cy={y} r={30} fill={SURFACE} stroke={k === 'T' ? INK3 : INK2} strokeWidth={1.2} />
            <text x={x} y={y + 4} textAnchor="middle" fill={INK} fontSize={14} fontWeight={600}>{k === 'T' ? 'T' : k}</text>
          </g>
        ))}
      </svg>
      <div className="note" style={{ marginTop: 6 }}>{note}</div>
    </div>
  );
}

export default function ExampleGraph({ data }) {
  if (!data) return null;
  const u = data.unconstrained, c = data.constraint_aware;
  const Qu = (k) => u.Q[k].toFixed(1), Qc = (k) => c.Q[k].toFixed(1);
  const unc = (
    <>
      <Edge from="s0" to="T" label="Safe · R=5" chosen={false} color={SLOT[0]} offset={-30} />
      <Edge from="s0" to="s1" label={`Risky · R=0 · Q=${Qu('s0:Risky')}`} chosen={u.selected_at_s0 === 'Risky'} color={SLOT[1]} offset={-10} />
      <Edge from="s1" to="T" label="Deploy · R=10 (proposed; gate blocks)" chosen={false} color={SLOT[1]} offset={-24} />
      <Edge from="s1" to="T" label="Recover · R=1" chosen={false} color={SLOT[1]} offset={24} />
    </>
  );
  const con = (
    <>
      <Edge from="s0" to="T" label={`Safe · R=5 · Q=${Qc('s0:Safe')}`} chosen={c.selected_at_s0 === 'Safe'} color={SLOT[0]} offset={-30} />
      <Edge from="s0" to="s1" label={`Risky · Q=${Qc('s0:Risky')}`} chosen={false} color={SLOT[1]} offset={-10} />
      <Edge from="s1" to="T" label="Deploy ✕ prohibited" prohibited color={SLOT[1]} offset={-24} />
      <Edge from="s1" to="T" label={`Recover · R=1 (only option, V_I=${c.V_I_s1.toFixed(1)})`} chosen color={SLOT[1]} offset={24} />
    </>
  );
  return (
    <div className="grid2">
      <Panel title="Unconstrained planner + runtime gate" color={INK} edges={unc}
             subtitle={`selects ${u.selected_at_s0} at s0 · executed return ${u.return_with_gate.toFixed(2)}`}
             note={u.explanation} />
      <Panel title="Constraint-aware planner" color={INK} edges={con}
             subtitle={`selects ${c.selected_at_s0} at s0 · executed return ${c.return.toFixed(2)}`}
             note={c.explanation} />
    </div>
  );
}

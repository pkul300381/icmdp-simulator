import React from 'react';
import { STATUS, SURFACE, GRID, INK, INK2, INK3 } from '../palette.js';

// One decision step as a Markov graph: current state -> actions (status, Q) -> successor states.
// Edge width encodes transition probability (thin marks, max 5px). Excluded actions are dashed and
// labelled; the policy's choice is outlined and starred so the ICMDP effect is visible at a glance.
const W = 1000;
const COL = { state: 10, action: 350, outcome: 630 };
const AW = 200, OW = 330, SW = 300;

function fmtState(st) {
  return `${st.severity} · ${st.health} · ${st.deploy_status} · cpu ${st.cpu} · ${st.latency}ms · r${st.replicas}`;
}

export default function DecisionGraph({ data }) {
  if (!data) return null;
  const blocks = data.actions.map((a) => ({ a, h: Math.max(58, a.outcomes.length * 30 + 14) }));
  const total = blocks.reduce((s, b) => s + b.h, 0) + 20;
  const H = Math.max(total, 260);
  let y = 10;
  const laid = blocks.map((b) => {
    const top = y;
    y += b.h;
    return { ...b, top, mid: top + b.h / 2 };
  });
  const stateY = H / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="decision graph" style={{ maxHeight: 620 }}>
      {/* state node */}
      <g>
        <rect x={COL.state} y={stateY - 34} width={SW} height={68} rx={8} fill={SURFACE} stroke={INK3} />
        <text x={COL.state + 12} y={stateY - 12} fill={INK} fontSize={12} fontWeight={600}>current state s</text>
        <text x={COL.state + 12} y={stateY + 6} fill={INK2} fontSize={11} className="mono">{fmtState(data.state)}</text>
        <text x={COL.state + 12} y={stateY + 22} fill={INK3} fontSize={11}>policy choice: {data.policy_choice ?? '—'}</text>
      </g>

      {laid.map(({ a, top, mid, h }) => {
        const color = STATUS[a.status];
        const dash = a.status === 'excluded' ? '6 4' : a.status === 'unavailable' ? '2 3' : undefined;
        const star = a.is_policy_choice;
        const sx = COL.state + SW, sy = stateY;
        const ax = COL.action, ay = mid;
        return (
          <g key={a.action}>
            {/* edge state -> action */}
            <path d={`M ${sx} ${sy} C ${sx + 40} ${sy}, ${ax - 40} ${ay}, ${ax} ${ay}`} fill="none"
                  stroke={color} strokeWidth={a.status === 'admissible' ? 1.5 : 1} strokeDasharray={dash} opacity={a.status === 'unavailable' ? 0.5 : 1} />
            {/* action node */}
            <rect x={ax} y={mid - 24} width={AW} height={48} rx={8} fill={SURFACE}
                  stroke={star ? INK : color} strokeWidth={star ? 2.5 : 1} strokeDasharray={dash} />
            <text x={ax + 12} y={mid - 4} fill={INK} fontSize={13} fontWeight={600}>
              {star ? '★ ' : ''}{a.action}
            </text>
            <text x={ax + 12} y={mid + 13} fill={color} fontSize={11}>
              {a.status === 'admissible' ? '✓ admissible' : a.status === 'excluded' ? '✕ excluded by intent' : '– unavailable'}
              {a.q != null ? `   Q = ${a.q.toFixed(2)}` : ''}
            </text>

            {/* outcomes */}
            {a.outcomes.map((o, j) => {
              const oy = top + 8 + j * 30;
              const ocy = oy + 11;
              const wpx = 1 + 4 * o.prob;
              return (
                <g key={j}>
                  <path d={`M ${ax + AW} ${ay} C ${ax + AW + 60} ${ay}, ${COL.outcome - 60} ${ocy}, ${COL.outcome} ${ocy}`}
                        fill="none" stroke={color} strokeWidth={wpx} opacity={a.status === 'admissible' ? 0.7 : 0.35} />
                  <rect x={COL.outcome} y={oy} width={OW} height={22} rx={5} fill={SURFACE} stroke={GRID} />
                  <text x={COL.outcome + 8} y={oy + 15} fill={INK2} fontSize={11} className="mono">
                    {`${o.state.severity}/${o.state.health}/${o.state.deploy_status}  cpu ${o.state.cpu} · ${o.state.latency}ms`}
                  </text>
                  <text x={COL.outcome + OW - 8} y={oy + 15} textAnchor="end" fill={INK} fontSize={11}>
                    {`p=${o.prob.toFixed(2)}${o.value != null ? `  V=${o.value.toFixed(1)}` : ''}`}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

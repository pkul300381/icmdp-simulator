import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Rectangle, ResponsiveContainer } from 'recharts';
import { ARM_COLOR, GRID, INK2, INK3, SURFACE } from '../palette.js';

// Bars per arm with direct value labels (relief rule: the light palette fails 3:1 contrast on some slots).
// A table view with the full numbers, CIs and planning cost sits beside every chart.
// Bar shape: recharts' own Rectangle (correct for negative values), plus a value label that is
// always drawn. recharts' LabelList drops labels on zero-height bars, which would hide the zero
// results that matter most here. Negative labels sit below the bar's lower edge.
function TaggedBar({ x, y, width, height, fill, payload }) {
  if (x == null || y == null || height == null) return null;
  const top = Math.min(y, y + height);
  const bottom = Math.max(y, y + height);
  const neg = (payload?.value ?? 0) < 0;
  const labelY = neg ? bottom + 13 : top - 6;
  return (
    <g>
      {Math.abs(height) > 0 && <Rectangle x={x} y={top} width={width} height={bottom - top} radius={3} fill={fill} />}
      <text x={x + width / 2} y={labelY} textAnchor="middle" fill={INK2} fontSize={11}>{payload?.text}</text>
    </g>
  );
}

export function ArmBars({ rows, valueKey, labelKey = 'mean', unit, title }) {
  const [showTable, setShowTable] = useState(false);
  const data = rows.map((r) => ({
    name: r.label,
    arm: r.arm,
    value: r[valueKey],
    text: unit === 'return' ? `${r[valueKey].toFixed(1)} ±${r.ci.toFixed(1)}` : `${r[valueKey].toFixed(2)}`,
  }));
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3>{title}</h3>
        <button onClick={() => setShowTable((v) => !v)}>{showTable ? 'Show chart' : 'Show table'}</button>
      </div>
      {showTable ? (
        <table className="data">
          <thead><tr><th>arm</th><th className="num">return</th><th className="num">±95% CI</th><th className="num">violations / 100 steps</th><th className="num">violating episodes %</th><th className="num">gate interventions / ep</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.arm}>
                <td><span style={{ color: ARM_COLOR[r.arm] }}>■</span> {r.arm}</td>
                <td className="num">{r.return.toFixed(2)}</td>
                <td className="num">{r.ci.toFixed(2)}</td>
                <td className="num">{r.violations.toFixed(2)}</td>
                <td className="num">{r.violatingPct.toFixed(1)}</td>
                <td className="num">{r.interventions.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div style={{ width: '100%', height: 280 }}>
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 22, right: 10, left: 0, bottom: 4 }}>
              <CartesianGrid vertical={false} stroke={GRID} />
              <XAxis dataKey="name" tick={{ fill: INK2, fontSize: 11 }} stroke={GRID} interval={0} />
              <YAxis tick={{ fill: INK3, fontSize: 11 }} stroke={GRID} width={48}
                     domain={[(dataMin) => Math.min(Math.floor(dataMin * 1.1), 0), (dataMax) => Math.max(Math.ceil(dataMax * 1.1), 0)]} />
              <ReferenceLine y={0} stroke={INK3} />
              <Tooltip cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                       contentStyle={{ background: SURFACE, border: `1px solid ${GRID}`, color: INK2, fontSize: 12 }}
                       formatter={(v, n, p) => [p.payload.text, title]} />
              <Bar dataKey="value" maxBarSize={56} isAnimationActive={false}
                   shape={(p) => <TaggedBar {...p} fill={ARM_COLOR[p.payload.arm]} />} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

const SHORT = {
  '1_unconstrained': 'unconstrained', '2_finite_penalty': 'penalty λ=100', '3_zero_budget_cmdp': 'zero-budget CMDP',
  '4_runtime_only': 'gate only', '5_constraint_aware_only': 'A_I, no gate', '6_combined': 'A_I + gate',
  F1_no_update: 'no update', F2_stale_policy_with_new_filter: 'stale + new filter', F3_stale_policy_no_filter_after_switch: 'stale, no filter',
  F4_reoptimised_with_filter: 'reopt. + filter', F5_cmdp_resolve_no_filter: 'CMDP re-solve',
};

export function toArmRows(summaries, freeze = false) {
  return summaries.map((s) => ({
    arm: s.arm,
    label: SHORT[s.arm] ?? s.arm,
    return: s.return.mean,
    ci: s.return.ci95,
    violations: freeze ? s.violations_per_100_steps : s.violations_per_100_steps,
    violatingPct: s.violating_episode_pct,
    interventions: s.interventions_per_episode ?? 0,
  }));
}

export function SweepChart({ sweep }) {
  const data = sweep.map((r) => ({ lambda: String(r.lambda), violations: r.violations_per_100_steps, text: r.violations_per_100_steps.toFixed(2) }));
  return (
    <div className="card">
      <h3>Finite penalty λ vs hard exclusion</h3>
      <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
        A penalty only removes violations once λ is large enough to outweigh the gain from violating.
      </div>
      <div style={{ width: '100%', height: 240 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 22, right: 10, left: 0, bottom: 4 }}>
            <CartesianGrid vertical={false} stroke={GRID} />
            <XAxis dataKey="lambda" tick={{ fill: INK2, fontSize: 11 }} stroke={GRID} label={{ value: 'λ', fill: INK3, position: 'insideBottomRight', offset: -4 }} />
            <YAxis tick={{ fill: INK3, fontSize: 11 }} stroke={GRID} width={48} label={{ value: 'viol / 100 steps', angle: -90, fill: INK3, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: SURFACE, border: `1px solid ${GRID}`, color: INK2, fontSize: 12 }} />
            <Bar dataKey="violations" maxBarSize={48} isAnimationActive={false}
                 shape={(p) => <TaggedBar {...p} fill={ARM_COLOR['2_finite_penalty']} />} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

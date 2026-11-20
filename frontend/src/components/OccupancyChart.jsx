import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { SEV, GRID, INK2, INK3, SURFACE } from '../palette.js';

// Stacked area: share of episodes in each severity state at each step (state occupancy over time).
// Severity is ordinal and carries meaning (good / warning / critical), so the status tokens are used.
export default function OccupancyChart({ occupancy, title }) {
  const data = occupancy.map((o) => ({ t: o.t, none: o.none * 100, sev2: o.sev2 * 100, sev1: o.sev1 * 100 }));
  return (
    <div className="card">
      <h3>{title}</h3>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <AreaChart data={data} margin={{ top: 8, right: 10, left: 0, bottom: 4 }}>
            <CartesianGrid vertical={false} stroke={GRID} />
            <XAxis dataKey="t" tick={{ fill: INK2, fontSize: 11 }} stroke={GRID} label={{ value: 'step', fill: INK3, position: 'insideBottomRight', offset: -4, fontSize: 11 }} />
            <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tick={{ fill: INK3, fontSize: 11 }} stroke={GRID} width={52} tickFormatter={(v) => `${v}%`} />
            <Tooltip contentStyle={{ background: SURFACE, border: `1px solid ${GRID}`, color: INK2, fontSize: 12 }}
                     formatter={(v) => `${v.toFixed(1)}%`} />
            <Legend wrapperStyle={{ fontSize: 12, color: INK2 }} />
            <Area isAnimationActive={false} type="linear" dataKey="none" name="no incident" stackId="1" stroke={SEV.none} fill={SEV.none} fillOpacity={0.55} strokeWidth={1} />
            <Area isAnimationActive={false} type="linear" dataKey="sev2" name="sev2" stackId="1" stroke={SEV.sev2} fill={SEV.sev2} fillOpacity={0.55} strokeWidth={1} />
            <Area isAnimationActive={false} type="linear" dataKey="sev1" name="sev1" stackId="1" stroke={SEV.sev1} fill={SEV.sev1} fillOpacity={0.6} strokeWidth={1} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

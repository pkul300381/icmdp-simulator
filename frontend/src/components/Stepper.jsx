import React from 'react';

const ICON = { ok: '✓', error: '✕', hold: '⚠', pending: '○', validated: '✓', rejected: '✕', active: '●' };

// The ICMDP lifecycle as a visible pipeline: parse (untrusted) -> validate -> compile -> feasibility -> activate.
export default function Stepper({ stages }) {
  return (
    <div className="stepper">
      {stages.map((s) => (
        <div key={s.name} className="step">
          <div className="name">{s.name}</div>
          <div style={{ marginTop: 6 }}>
            <span className={`chip ${s.status}`}>{ICON[s.status] ?? '·'} {s.status}</span>
          </div>
          <div className="detail">{s.detail}</div>
          {s.seconds != null && <div className="detail mono">{(s.seconds * 1000).toFixed(1)} ms</div>}
        </div>
      ))}
    </div>
  );
}

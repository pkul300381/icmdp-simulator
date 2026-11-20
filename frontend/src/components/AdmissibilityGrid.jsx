import React from 'react';

const ICON = { admissible: '✓', excluded: '✕', unavailable: '–' };

// Admissible set over (latency rows x cpu columns) for fixed severity/health/deploy/replicas.
// Each action is a chip showing status with an icon, so the grid reads without color.
export default function AdmissibilityGrid({ data }) {
  if (!data) return null;
  const cpus = [...new Set(data.cells.map((c) => c.cpu))];
  const lats = [...new Set(data.cells.map((c) => c.latency))].sort((a, b) => b - a);
  const byKey = Object.fromEntries(data.cells.map((c) => [`${c.cpu}|${c.latency}`, c]));
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="adm-grid">
        <div className="axis-label">latency ↓ / cpu →</div>
        {cpus.map((c) => <div key={c} className="axis-label" style={{ justifyContent: 'center' }}>cpu {c}%</div>)}
        {lats.map((lat) => (
          <React.Fragment key={lat}>
            <div className="axis-label">{lat} ms</div>
            {cpus.map((c) => {
              const cell = byKey[`${c}|${lat}`];
              return (
                <div key={c} className="adm-cell">
                  <div className="chips">
                    {data.actions.map((a) => {
                      const st = cell.statuses[a];
                      return (
                        <span key={a} className={`chip ${st}`} title={`${a}: ${st}`}>
                          {ICON[st]} {a}
                        </span>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
      <div className="legend">
        <span style={{ '--c': '#0ca30c' }}>✓ admissible</span>
        <span style={{ '--c': '#d03b3b' }}>✕ excluded by intent (struck through)</span>
        <span style={{ '--c': '#8f8e85' }}>– unavailable (physically impossible)</span>
      </div>
    </div>
  );
}

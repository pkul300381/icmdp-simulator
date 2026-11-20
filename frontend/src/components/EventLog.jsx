import React from 'react';

// Append-only lifecycle log (intent parsed, rejected, held, activated; runs done).
export default function EventLog({ events }) {
  return (
    <table className="data">
      <thead><tr><th>time (UTC)</th><th>event</th><th>ref</th><th>payload</th></tr></thead>
      <tbody>
        {events.map((e) => (
          <tr key={e.id}>
            <td className="mono">{e.ts}</td>
            <td><span className={`chip ${e.kind.split('.')[1] ?? ''}`}>{e.kind}</span></td>
            <td className="mono">{e.ref}</td>
            <td className="mono" style={{ color: 'var(--text-secondary)' }}>{JSON.stringify(e.payload)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

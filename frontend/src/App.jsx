import React, { useEffect, useState } from 'react';
import { api } from './api.js';
import IntentPage from './pages/IntentPage.jsx';
import MarkovPage from './pages/MarkovPage.jsx';
import WorkflowPage from './pages/WorkflowPage.jsx';
import ExperimentPage from './pages/ExperimentPage.jsx';
import ObservabilityPage from './pages/ObservabilityPage.jsx';

const TABS = [
  ['intent', 'Intent lifecycle'],
  ['markov', 'Markov views'],
  ['workflow', 'ICMDP vs CMDP workflow'],
  ['experiments', 'Experiments'],
  ['observability', 'Observability'],
];

export default function App() {
  const [tab, setTab] = useState(() => new URLSearchParams(window.location.search).get('tab') || 'intent');
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(e.message));
  }, [refreshKey]);

  const active = meta?.active_intent;
  const refresh = () => setRefreshKey((k) => k + 1);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">ICMDP Agent Harness<small>intent-constrained MDP · Kulkarni, Saini &amp; Das</small></div>
        <nav className="tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={`tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        <div className="active-pill">
          {active ? <>active rule v{active.version}: <span className="mono">{active.name || active.instruction}</span></>
                  : 'no intent active (unconstrained baseline)'}
        </div>
      </header>
      <main className="main">
        {error && <div className="error-box">Backend not reachable ({error}). Start it with <code>uvicorn icmdp.api:app --port 8000</code> from <code>backend/</code>.</div>}
        {meta && tab === 'intent' && <IntentPage meta={meta} onChange={refresh} />}
        {meta && tab === 'markov' && <MarkovPage meta={meta} />}
        {meta && tab === 'workflow' && <WorkflowPage meta={meta} />}
        {meta && tab === 'experiments' && <ExperimentPage meta={meta} />}
        {meta && tab === 'observability' && <ObservabilityPage />}
      </main>
    </div>
  );
}

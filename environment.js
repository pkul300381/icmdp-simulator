/**
 * environment.js
 * MDP Environment — Cloud-Operations Agent (Paper Example 3.1)
 *
 * State space S, action vocabulary A, baseline availability A₀(s),
 * transition model P(s'|s,a), reward function R(s,a), discount γ.
 *
 * Paper reference: Definition 2.1, Definition 3.1, Example 3.1
 */

'use strict';

// ---------------------------------------------------------------------------
// State schema — typed fields for validator/compiler grounding context x
// ---------------------------------------------------------------------------
const STATE_SCHEMA = {
  severity:      { type: 'enum',    values: ['none', 'sev2', 'sev1'] },
  health:        { type: 'enum',    values: ['healthy', 'degraded', 'critical'] },
  cpu:           { type: 'number',  unit: 'percent', min: 0, max: 100 },
  latency:       { type: 'number',  unit: 'ms',      min: 0, max: 5000 },
  replicas:      { type: 'integer', min: 1, max: 10 },
  deploy_status: { type: 'enum',    values: ['idle', 'in_progress'] },
};

// ---------------------------------------------------------------------------
// Action vocabulary A (global)
// ---------------------------------------------------------------------------
const ACTIONS = ['Deploy', 'Rollback', 'Restart', 'ScaleOut', 'ScaleIn', 'DoNothing'];

// ---------------------------------------------------------------------------
// Baseline availability A₀(s) — Paper Definition 3.1
// Actions physically executable in state s, regardless of user intent.
// ---------------------------------------------------------------------------
function baselineAvailable(s) {
  const available = new Set(ACTIONS);
  if (s.deploy_status === 'in_progress') available.delete('Deploy');
  if (s.deploy_status === 'idle')        available.delete('Rollback');
  if (s.replicas >= 10)                  available.delete('ScaleOut');
  if (s.replicas <= 1)                   available.delete('ScaleIn');
  return available;
}

// ---------------------------------------------------------------------------
// Reward function R(s, a) — Paper Definition 2.1
// Optimises service availability, latency, recovery, and infrastructure cost.
// ---------------------------------------------------------------------------
function reward(s, a) {
  const penalties = {
    none:     0,
    sev2:   -10,
    sev1:   -30,
  };
  const healthPenalties = {
    healthy:   0,
    degraded: -5,
    critical: -20,
  };
  const latencyPenalty = Math.min(s.latency / 500, 10);
  const cpuPenalty     = s.cpu > 80 ? (s.cpu - 80) * 0.2 : 0;
  const basePenalty    = penalties[s.severity] + healthPenalties[s.health]
                         - latencyPenalty - cpuPenalty;

  const actionRewards = {
    Deploy:    s.health === 'healthy' ? 8 : -5,
    Rollback:  s.severity !== 'none' ? 12 : -2,
    Restart:   s.health !== 'healthy' ? 6 : -1,
    ScaleOut:  s.cpu > 70 ? 7 : -3,
    ScaleIn:   s.replicas > 3 && s.cpu < 40 ? 3 : -1,
    DoNothing: 0,
  };

  return basePenalty + (actionRewards[a] || 0);
}

// ---------------------------------------------------------------------------
// State enumeration — discrete state space for value iteration
// ---------------------------------------------------------------------------
function enumerateStates() {
  const states = [];
  for (const severity of STATE_SCHEMA.severity.values) {
    for (const health of STATE_SCHEMA.health.values) {
      for (const cpu of [20, 50, 70, 85, 95]) {
        for (const latency of [100, 300, 800, 2000]) {
          for (const replicas of [1, 2, 3, 5, 8, 10]) {
            for (const deploy_status of STATE_SCHEMA.deploy_status.values) {
              states.push({ severity, health, cpu, latency, replicas, deploy_status });
            }
          }
        }
      }
    }
  }
  return states;
}

// ---------------------------------------------------------------------------
// Transition model P(s'|s,a) — stochastic environment response
// Returns array of { state, prob } pairs summing to 1.
// ---------------------------------------------------------------------------
function transition(s, a) {
  // Deterministic next-state logic with stochastic noise
  let next = { ...s };

  if (a === 'Deploy') {
    next.deploy_status = 'in_progress';
    // Deploy may improve or worsen health probabilistically
    return [
      { state: { ...next, health: 'healthy',  deploy_status: 'idle', severity: 'none' }, prob: 0.7 },
      { state: { ...next, health: 'degraded', deploy_status: 'idle', severity: 'sev2' }, prob: 0.2 },
      { state: { ...next, health: 'critical', deploy_status: 'idle', severity: 'sev1' }, prob: 0.1 },
    ];
  }
  if (a === 'Rollback') {
    return [
      { state: { ...s, health: 'healthy', severity: 'none', deploy_status: 'idle' }, prob: 0.85 },
      { state: { ...s, health: 'degraded', severity: 'sev2', deploy_status: 'idle' }, prob: 0.15 },
    ];
  }
  if (a === 'Restart') {
    return [
      { state: { ...s, health: 'healthy', cpu: 30, latency: 150 }, prob: 0.75 },
      { state: { ...s, health: 'degraded', cpu: 60, latency: 400 }, prob: 0.25 },
    ];
  }
  if (a === 'ScaleOut') {
    const r = Math.min(s.replicas + 1, 10);
    return [{ state: { ...s, replicas: r, cpu: Math.max(s.cpu - 15, 10) }, prob: 1.0 }];
  }
  if (a === 'ScaleIn') {
    const r = Math.max(s.replicas - 1, 1);
    return [{ state: { ...s, replicas: r, cpu: Math.min(s.cpu + 20, 100) }, prob: 1.0 }];
  }
  // DoNothing — environment may self-resolve or worsen
  return [
    { state: { ...s }, prob: 0.6 },
    { state: { ...s, cpu: Math.min(s.cpu + 10, 100), latency: Math.min(s.latency + 100, 5000) }, prob: 0.3 },
    { state: { ...s, severity: s.severity === 'none' ? 'sev2' : s.severity, health: s.health === 'healthy' ? 'degraded' : s.health }, prob: 0.1 },
  ];
}

// ---------------------------------------------------------------------------
// State key for lookup tables
// ---------------------------------------------------------------------------
function stateKey(s) {
  return `${s.severity}|${s.health}|${s.cpu}|${s.latency}|${s.replicas}|${s.deploy_status}`;
}

// ---------------------------------------------------------------------------
// Find closest enumerated state to an arbitrary state object
// ---------------------------------------------------------------------------
function snapToGrid(s, states) {
  const cpuGrid     = [20, 50, 70, 85, 95];
  const latGrid     = [100, 300, 800, 2000];
  const repGrid     = [1, 2, 3, 5, 8, 10];
  const snapCpu     = cpuGrid.reduce((a, b) => Math.abs(b - s.cpu) < Math.abs(a - s.cpu) ? b : a);
  const snapLat     = latGrid.reduce((a, b) => Math.abs(b - s.latency) < Math.abs(a - s.latency) ? b : a);
  const snapRep     = repGrid.reduce((a, b) => Math.abs(b - s.replicas) < Math.abs(a - s.replicas) ? b : a);
  return {
    severity:      s.severity,
    health:        s.health,
    cpu:           snapCpu,
    latency:       snapLat,
    replicas:      snapRep,
    deploy_status: s.deploy_status,
  };
}

// Discount factor γ (Paper Definition 2.1)
const GAMMA = 0.9;

if (typeof module !== 'undefined') {
  module.exports = { STATE_SCHEMA, ACTIONS, GAMMA, baselineAvailable, reward, transition, stateKey, snapToGrid, enumerateStates };
}

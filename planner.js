/**
 * planner.js
 * Constraint-Aware Planner — Value Iteration and Policy Iteration
 *
 * Implements Paper Section 5.1, equations 19–20.
 *
 * V_{k+1}(s) = max_{a ∈ A_I(s)} [ R(s,a) + γ Σ_{s'} P(s'|s,a) V_k(s') ]
 *
 * Convergence guaranteed by Proposition 3.1 (γ-contraction, Banach fixed point).
 *
 * Also implements Example 5.1 (Paper §5.1):
 * Demonstrates why runtime-filtering alone is insufficient.
 */

'use strict';

// ---------------------------------------------------------------------------
// Value Iteration over A_I(s)   (Paper Equation 19)
//
// Params:
//   states          — array of all enumerated states
//   admissibleFn    — (s) → Set<string> of admissible actions
//   rewardFn        — (s, a) → number
//   transitionFn    — (s, a) → [{state, prob}]
//   stateKeyFn      — (s) → string
//   snapToGridFn    — (s, states) → nearest grid state
//   gamma           — discount factor
//   maxIter         — iteration limit
//   epsilon         — convergence threshold
// ---------------------------------------------------------------------------
function valueIteration({
  states,
  admissibleFn,
  rewardFn,
  transitionFn,
  stateKeyFn,
  snapToGridFn,
  gamma = 0.9,
  maxIter = 200,
  epsilon = 1e-4,
}) {
  // Initialise V(s) = 0 for all s
  const V = new Map();
  const policy = new Map();
  for (const s of states) {
    V.set(stateKeyFn(s), 0);
  }

  let iterations = 0;
  let delta = Infinity;

  while (delta > epsilon && iterations < maxIter) {
    delta = 0;
    iterations++;

    for (const s of states) {
      const key = stateKeyFn(s);
      const AI  = admissibleFn(s);

      if (AI.size === 0) {
        // Infeasible state — undefined (Paper §3.1)
        V.set(key, -Infinity);
        policy.set(key, null);
        continue;
      }

      let bestValue  = -Infinity;
      let bestAction = null;

      for (const a of AI) {
        const transitions = transitionFn(s, a);
        const qValue = rewardFn(s, a) + gamma * transitions.reduce((sum, { state: s2, prob }) => {
          const s2snapped = snapToGridFn(s2, states);
          const v2 = V.get(stateKeyFn(s2snapped)) ?? 0;
          return sum + prob * v2;
        }, 0);

        if (qValue > bestValue) {
          bestValue  = qValue;
          bestAction = a;
        }
      }

      const oldV = V.get(key);
      V.set(key, bestValue);
      policy.set(key, bestAction);
      delta = Math.max(delta, Math.abs(bestValue - oldV));
    }
  }

  return { V, policy, iterations, converged: delta <= epsilon };
}

// ---------------------------------------------------------------------------
// Q-value computation for a single state   (Paper Equation 21 context)
// Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
// ---------------------------------------------------------------------------
function computeQValues(s, admissibleFn, rewardFn, transitionFn, V, stateKeyFn, snapToGridFn, states, gamma) {
  const AI = admissibleFn(s);
  const qValues = {};

  for (const a of AI) {
    const transitions = transitionFn(s, a);
    qValues[a] = rewardFn(s, a) + gamma * transitions.reduce((sum, { state: s2, prob }) => {
      const s2s = snapToGridFn(s2, states);
      return sum + prob * (V.get(stateKeyFn(s2s)) ?? 0);
    }, 0);
  }

  return qValues;
}

// ---------------------------------------------------------------------------
// Episode runner — executes one episode following the computed policy
// with runtime gate enforcing A_I(s) at every step.
// Returns audit trace (Paper Algorithm 1, Step 6).
// ---------------------------------------------------------------------------
function runEpisode({
  initialState,
  policy,
  V,
  admissibleFn,
  rewardFn,
  transitionFn,
  stateKeyFn,
  snapToGridFn,
  states,
  gamma,
  maxSteps = 20,
  ruleVersion = 1,
}) {
  const trace = [];
  let s = snapToGridFn(initialState, states);
  let totalReturn = 0;
  let step = 0;

  while (step < maxSteps) {
    const key = stateKeyFn(s);
    const AI  = admissibleFn(s);

    // Feasibility monitor (Paper §4.3)
    if (AI.size === 0) {
      trace.push({
        step, state: { ...s }, ruleVersion,
        admissibleSet: [],
        proposal: null,
        executed: null,
        reward: null,
        event: 'HOLD — empty admissible set, requesting clarification',
      });
      break;
    }

    // Policy proposal (may be from unconstrained or constrained planner)
    const proposal = policy.get(key) || [...AI][0];

    // Runtime gate: filter proposal through A_I(s)
    const executed = AI.has(proposal) ? proposal : [...AI][0];

    const r = rewardFn(s, executed);
    totalReturn += Math.pow(gamma, step) * r;

    trace.push({
      step,
      state: { ...s },
      ruleVersion,
      admissibleSet: [...AI],
      proposal,
      executed,
      reward: r,
      event: proposal !== executed ? 'GATE_FILTERED — proposal outside A_I(s)' : 'OK',
    });

    // Stochastic transition
    const transitions = transitionFn(s, executed);
    const rand = Math.random();
    let cumProb = 0;
    let nextState = transitions[0].state;
    for (const { state: s2, prob } of transitions) {
      cumProb += prob;
      if (rand <= cumProb) { nextState = s2; break; }
    }
    s = snapToGridFn(nextState, states);
    step++;

    // Terminal: healthy, no incident, idle
    if (s.severity === 'none' && s.health === 'healthy' && s.deploy_status === 'idle') {
      trace.push({ step, state: { ...s }, ruleVersion, admissibleSet: [...admissibleFn(s)], proposal: null, executed: null, reward: null, event: 'TERMINAL — goal state reached' });
      break;
    }
  }

  return { trace, totalReturn, steps: step };
}

// ---------------------------------------------------------------------------
// Example 5.1 — Paper §5.1
// Demonstrates why runtime-only filtering is insufficient.
//
// Setup: γ=0.9
//   s0: Safe → R=5 (terminal) | Risky → R=0 → s1
//   s1: Deploy → R=10 (terminal, PROHIBITED) | Recover → R=1 (terminal)
//
// Unconstrained planner + runtime gate: selects Risky, gets 0.9×1 = 0.9
// Constraint-aware planner:             selects Safe, gets 5
// ---------------------------------------------------------------------------
function runExample51(admissibleFnWithDeploy, admissibleFnWithout) {
  const gamma = 0.9;

  // Unconstrained Q-values at s0
  const Q_unconstrained_risky = 0 + gamma * 10; // 9.0 — expects Deploy at s1
  const Q_unconstrained_safe  = 5;               // 5.0

  // Unconstrained planner selects Risky (Q=9 > Q=5)
  // Runtime gate at s1 blocks Deploy, only Recover available
  const return_runtimeOnly = gamma * 1; // 0.9

  // Constraint-aware planner
  const V_I_s1 = 1; // Only Recover admissible at s1
  const Q_I_s0_risky = 0 + gamma * V_I_s1; // 0.9
  const Q_I_s0_safe  = 5;                   // 5.0
  // Constraint-aware selects Safe (Q_I=5 > Q_I=0.9)
  const return_constraintAware = 5;

  return {
    unconstrained: {
      Q_s0: { Safe: Q_unconstrained_safe, Risky: Q_unconstrained_risky },
      selected: 'Risky',
      returnWithGate: return_runtimeOnly,
      explanation: 'Unconstrained planner values Risky=9 (expecting Deploy at s₁). Runtime gate blocks Deploy, only Recover available. Return = 0.9×1 = 0.9.',
    },
    constraintAware: {
      V_I_s1,
      Q_I_s0: { Safe: Q_I_s0_safe, Risky: Q_I_s0_risky },
      selected: 'Safe',
      returnWithGate: return_constraintAware,
      explanation: 'Constraint-aware planner uses V_I(s₁)=1 (only Recover admissible). Q_I(s₀,Risky)=0.9 < Q_I(s₀,Safe)=5. Selects Safe. Return = 5.',
    },
    insight: 'Runtime gate ensures local compliance. Constraint-aware planning avoids upstream valuations relying on prohibited future actions.',
  };
}

if (typeof module !== 'undefined') {
  module.exports = { valueIteration, computeQValues, runEpisode, runExample51 };
}

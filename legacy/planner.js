/**
 * planner.js
 * Constraint-Aware Planner, Episode Runner, and Example V.1
 *
 * Implements Paper Section V.A, equations 19-20.
 *
 *   V_{k+1}(s) = max_{a in A_I(s)} [ R(s,a) + gamma * sum_s' P(s'|s,a) V_k(s') ]
 *
 * Value iteration is Jacobi (synchronous): every backup in sweep k reads V_k.
 * Convergence follows from Proposition III.1 (gamma-contraction).
 *
 * Enforcement semantics (Paper Algorithm 1, step 6):
 *   - The runtime gate evaluates the active rule on the RAW observed state.
 *   - Policies are indexed by the grid-snapped state (planning approximation only).
 *   - Replacement actions when the gate filters are chosen by value, not by order.
 */

'use strict';

// ---------------------------------------------------------------------------
// Backup helper: Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(snap(s'))
// ctx = { rewardFn, transitionFn, V, stateKeyFn, snapToGridFn, states, gamma }
// ---------------------------------------------------------------------------
function qFromV(s, a, ctx) {
  const { rewardFn, transitionFn, V, stateKeyFn, snapToGridFn, states, gamma } = ctx;
  let future = 0;
  for (const { state: s2, prob } of transitionFn(s, a)) {
    future += prob * (V.get(stateKeyFn(snapToGridFn(s2, states))) ?? 0);
  }
  return rewardFn(s, a) + gamma * future;
}

// ---------------------------------------------------------------------------
// Value iteration over A_I(s)   (Paper Equation 19, Jacobi form)
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
  let V = new Map();
  const policy = new Map();
  for (const s of states) V.set(stateKeyFn(s), 0);

  let iterations = 0;
  let delta = Infinity;

  while (delta > epsilon && iterations < maxIter) {
    iterations++;
    const Vnext = new Map();
    const ctx = { rewardFn, transitionFn, V, stateKeyFn, snapToGridFn, states, gamma };
    delta = 0;

    for (const s of states) {
      const key = stateKeyFn(s);
      const AI = admissibleFn(s);

      if (AI.size === 0) {
        // Infeasible state: Bellman update undefined (Paper Def. III.2); flagged, not relaxed.
        Vnext.set(key, -Infinity);
        policy.set(key, null);
        continue;
      }

      let bestValue = -Infinity;
      let bestAction = null;
      for (const a of AI) {
        const q = qFromV(s, a, ctx);
        if (bestAction === null || q > bestValue) {
          bestValue = q;
          bestAction = a;
        }
      }

      Vnext.set(key, bestValue);
      policy.set(key, bestAction);

      const oldV = V.get(key);
      const d = (Number.isFinite(bestValue) && Number.isFinite(oldV))
        ? Math.abs(bestValue - oldV)
        : (bestValue === oldV ? 0 : Infinity);
      if (d > delta) delta = d;
    }

    V = Vnext;
  }

  return { V, policy, iterations, converged: delta <= epsilon, residual: delta };
}

// ---------------------------------------------------------------------------
// Replacement selection for the runtime gate: argmax_{a in A_I(s)} Q(s,a) under V.
// Ties and missing V fall back to deterministic alphabetical order.
// ---------------------------------------------------------------------------
function rankAdmissible(s, AI, ctx) {
  const candidates = [...AI].sort();
  if (candidates.length === 0) return null;
  if (!ctx || !ctx.V) return candidates[0];
  let best = candidates[0];
  let bestQ = -Infinity;
  for (const a of candidates) {
    const q = qFromV(s, a, ctx);
    if (q > bestQ) { best = a; bestQ = q; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Episode runner with runtime gate (Paper Algorithm 1, steps 5-7)
//
// Options
//   initialState   raw observed state (never snapped for enforcement)
//   policy         Map: stateKey(snapped s) -> action  (planner output)
//   policyAt       optional (step) -> Map, for time-varying proposals
//   V              value function used only to rank gate replacements
//   baselineFn     raw s -> Set A0(s)
//   ruleAt         optional (step) -> { rule, version } | null   enforcement rule
//   violationRuleAt optional (step) -> { rule, version } | null  user-intent rule used
//                  only to SCORE violations (lets a stale filter be measured against
//                  the instruction the user actually gave). Defaults to ruleAt.
//   gate           boolean or (step) -> boolean. When false the proposal executes raw.
//   isTerminal     optional (s) -> boolean
//   rng            () -> [0,1). Defaults to Math.random; pass a seeded RNG for reproducibility.
//   startStep      offset used for trace numbering and discounting (step-by-step UI)
// ---------------------------------------------------------------------------
function runEpisode({
  initialState,
  policy,
  policyAt = null,
  V = null,
  baselineFn,
  ruleAt = null,
  violationRuleAt = null,
  gate = true,
  rewardFn,
  transitionFn,
  stateKeyFn,
  snapToGridFn,
  states,
  gamma,
  isTerminal = null,
  maxSteps = 20,
  startStep = 0,
  rng = Math.random,
}) {
  const ctx = { V, rewardFn, transitionFn, stateKeyFn, snapToGridFn, states, gamma };
  const trace = [];
  let s = initialState;
  let totalReturn = 0;
  let steps = 0;
  let violations = 0;
  let interventions = 0;
  let holds = 0;
  let operationalErrors = 0;
  let terminated = false;

  for (let t = 0; t < maxSteps; t++) {
    const step = startStep + t;
    const snap = snapToGridFn(s, states);
    const key = stateKeyFn(snap);
    const A0 = baselineFn(s);

    const active = ruleAt ? ruleAt(step) : null;
    const I = active ? active.rule(s) : null;
    const AI = new Set([...A0].filter(a => !I || I.has(a)));
    const ruleVersion = active ? active.version : 0;

    // Feasibility monitor (Paper Def. III.3 / Section IV-C ii)
    if (AI.size === 0) {
      holds++;
      trace.push({
        step, state: { ...s }, ruleVersion,
        admissibleSet: [], proposal: null, executed: null, reward: null, violated: false,
        event: 'HOLD — A_I(s) = ∅, requesting clarification or escalation',
      });
      break;
    }

    const pol = policyAt ? policyAt(step) : policy;
    const proposal = pol.get(key) ?? null;
    const gateOn = typeof gate === 'function' ? gate(step) : gate;

    let executed;
    let event;
    if (gateOn) {
      if (proposal !== null && AI.has(proposal)) {
        executed = proposal;
        event = 'OK';
      } else {
        executed = rankAdmissible(snap, AI, ctx);
        interventions++;
        event = `GATE_FILTERED — "${proposal}" ∉ A_I(s), executing "${executed}"`;
      }
    } else {
      if (proposal === null) {
        holds++;
        trace.push({
          step, state: { ...s }, ruleVersion,
          admissibleSet: [...AI], proposal: null, executed: null, reward: null, violated: false,
          event: 'HOLD — no proposal for state (ungated)',
        });
        break;
      }
      executed = proposal;
      event = 'OK';
    }

    // Scoring against the user's intent (not against what the gate enforced)
    const truth = violationRuleAt ? violationRuleAt(step) : active;
    const Itruth = truth ? truth.rule(s) : null;
    const violated = !!Itruth && A0.has(executed) && !Itruth.has(executed);
    if (violated) {
      violations++;
      event = `VIOLATION — executed "${executed}" excluded by user intent`;
    }
    if (!A0.has(executed)) operationalErrors++;

    const r = rewardFn(s, executed);
    totalReturn += Math.pow(gamma, step) * r;
    steps++;

    trace.push({
      step, state: { ...s }, ruleVersion,
      admissibleSet: [...AI], proposal, executed, reward: r, violated, event,
    });

    // Stochastic transition on the raw state
    const tr = transitionFn(s, executed);
    const u = rng();
    let cum = 0;
    let ns = tr[tr.length - 1].state;
    for (const { state, prob } of tr) {
      cum += prob;
      if (u < cum) { ns = state; break; }
    }
    s = ns;

    if (isTerminal && isTerminal(s)) {
      terminated = true;
      trace.push({
        step: step + 1, state: { ...s }, ruleVersion,
        admissibleSet: [], proposal: null, executed: null, reward: null, violated: false,
        event: 'TERMINAL — goal state reached',
      });
      break;
    }
  }

  return {
    trace, totalReturn, steps, terminated,
    violations, interventions, holds, operationalErrors,
    finalState: s,
  };
}

// ---------------------------------------------------------------------------
// Example V.1 (Paper Section V.A), computed on a toy MDP with the same solver
// and episode runner used for the cloud model. Nothing here is hard-coded.
//
//   s0: Safe   -> R=5, terminal | Risky -> R=0, to s1
//   s1: Deploy -> R=10, terminal (PROHIBITED at s1) | Recover -> R=1, terminal
//   T : absorbing terminal, Noop -> R=0
// ---------------------------------------------------------------------------
const EX51_ACTIONS = { s0: ['Safe', 'Risky'], s1: ['Deploy', 'Recover'], T: ['Noop'] };
const EX51_REWARD  = { 's0|Safe': 5, 's0|Risky': 0, 's1|Deploy': 10, 's1|Recover': 1, 'T|Noop': 0 };
const EX51_NEXT    = { 's0|Safe': 'T', 's0|Risky': 's1', 's1|Deploy': 'T', 's1|Recover': 'T', 'T|Noop': 'T' };

function ex51Reward(s, a) { return EX51_REWARD[`${s.name}|${a}`]; }
function ex51Transition(s, a) { return [{ state: { name: EX51_NEXT[`${s.name}|${a}`] }, prob: 1 }]; }
function ex51A0(s) { return new Set(EX51_ACTIONS[s.name]); }
// Stationary compiled rule: Deploy prohibited at s1
function ex51Rule(s) {
  const I = new Set(['Safe', 'Risky', 'Deploy', 'Recover', 'Noop']);
  if (s.name === 's1') I.delete('Deploy');
  return I;
}

function runExample51() {
  const gamma = 0.9;
  const states = [{ name: 's0' }, { name: 's1' }, { name: 'T' }];
  const key = s => s.name;
  const identity = s => s;
  const isTerminal = s => s.name === 'T';
  const s0 = { name: 's0' };
  const ruleAt = () => ({ rule: ex51Rule, version: 1 });
  const common = {
    states, rewardFn: ex51Reward, transitionFn: ex51Transition,
    stateKeyFn: key, snapToGridFn: identity, gamma, maxIter: 500, epsilon: 1e-10,
  };

  // Unconstrained planner: optimises over A0 only
  const unc = valueIteration({ ...common, admissibleFn: ex51A0 });
  // Constraint-aware planner: optimises over A_I = A0 ∩ I
  const con = valueIteration({
    ...common,
    admissibleFn: s => new Set([...ex51A0(s)].filter(a => ex51Rule(s).has(a))),
  });

  const ctxUnc = { rewardFn: ex51Reward, transitionFn: ex51Transition, V: unc.V, stateKeyFn: key, snapToGridFn: identity, states, gamma };
  const ctxCon = { ...ctxUnc, V: con.V };
  const Q_unc = { Safe: qFromV(s0, 'Safe', ctxUnc), Risky: qFromV(s0, 'Risky', ctxUnc) };
  const Q_con = { Safe: qFromV(s0, 'Safe', ctxCon), Risky: qFromV(s0, 'Risky', ctxCon) };
  const selUnc = unc.policy.get('s0');
  const selCon = con.policy.get('s0');

  // Episodes: unconstrained proposal behind the runtime gate, and constraint-aware policy
  const epUnc = runEpisode({
    initialState: s0, policy: unc.policy, V: unc.V, baselineFn: ex51A0,
    ruleAt, gate: true, rewardFn: ex51Reward, transitionFn: ex51Transition,
    stateKeyFn: key, snapToGridFn: identity, states, gamma, isTerminal, maxSteps: 5,
  });
  const epCon = runEpisode({
    initialState: s0, policy: con.policy, V: con.V, baselineFn: ex51A0,
    ruleAt, gate: true, rewardFn: ex51Reward, transitionFn: ex51Transition,
    stateKeyFn: key, snapToGridFn: identity, states, gamma, isTerminal, maxSteps: 5,
  });

  return {
    unconstrained: {
      Q_s0: Q_unc,
      selected: selUnc,
      returnWithGate: epUnc.totalReturn,
      explanation: `Unconstrained planner values ${selUnc} highest (Q=${Q_unc[selUnc].toFixed(2)}), expecting Deploy at s1. Runtime gate blocks Deploy; replacement chosen by value. Return = ${epUnc.totalReturn.toFixed(2)}.`,
    },
    constraintAware: {
      V_I_s1: con.V.get('s1'),
      Q_I_s0: Q_con,
      selected: selCon,
      returnWithGate: epCon.totalReturn,
      explanation: `Constraint-aware planner uses V_I(s1)=${con.V.get('s1').toFixed(2)}. Q_I(s0,Risky)=${Q_con.Risky.toFixed(2)} < Q_I(s0,Safe)=${Q_con.Safe.toFixed(2)}. Selects ${selCon}. Return = ${epCon.totalReturn.toFixed(2)}.`,
    },
    insight: 'Runtime gate ensures local compliance. Constraint-aware planning avoids upstream valuations relying on prohibited future actions.',
  };
}

if (typeof module !== 'undefined') {
  module.exports = { valueIteration, qFromV, rankAdmissible, runEpisode, runExample51 };
}

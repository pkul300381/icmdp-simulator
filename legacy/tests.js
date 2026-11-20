/**
 * tests.js — run with:  node tests.js
 * Dependency-free (node:assert). Covers the enforcement claims the thesis depends on.
 */

'use strict';

const assert = require('assert');
const env = require('./environment.js');
const comp = require('./compiler.js');
const plan = require('./planner.js');
const exp = require('./experiment.js');

const { ACTIONS, STATE_SCHEMA, baselineAvailable, reward, transition, stateKey, snapToGrid, enumerateStates } = env;

const results = [];
function test(name, fn) {
  const t0 = Date.now();
  try {
    fn();
    results.push({ name, ok: true, ms: Date.now() - t0 });
  } catch (e) {
    results.push({ name, ok: false, error: e.message, ms: Date.now() - t0 });
  }
}

// ---------------------------------------------------------------------------
// 1. Validator: units, ranges, ambiguity (Paper IV-B: ⊥ on ambiguity, canonical units)
// ---------------------------------------------------------------------------
function v(condition) {
  return comp.validate({ prohibitions: [{ action: 'ScaleIn', condition }] }, STATE_SCHEMA, ACTIONS);
}

test('validator rejects out-of-range cpu', () => {
  assert.strictEqual(v('cpu > 150').valid, false);
});

test('validator rejects ambiguous non-integer latency without unit', () => {
  const r = v('latency > 1.2');
  assert.strictEqual(r.valid, false);
  assert.match(r.reason, /ambiguous/);
});

test('validator converts seconds to ms', () => {
  const r = v('latency > 1.2 s');
  assert.strictEqual(r.valid, true);
  assert.strictEqual(r.spec.prohibitions[0].condition.value, 1200);
});

test('validator converts cpu fraction to percent', () => {
  const r = v('cpu > 0.7 fraction');
  assert.strictEqual(r.valid, true);
  assert.strictEqual(r.spec.prohibitions[0].condition.value, 70);
});

test('validator rejects unknown unit and empty numeric literal', () => {
  assert.strictEqual(v('latency > 5 parsecs').valid, false);
  assert.strictEqual(v('latency >').valid, false);   // Number('') would silently be 0
});

test('validator rejects non-integer replicas', () => {
  assert.strictEqual(v('replicas > 2.5').valid, false);
});

test('validator rejects misspelt enum value (no silent case folding)', () => {
  const r = comp.validate({ prohibitions: [{ action: 'Deploy', condition: 'severity = SEV1' }] }, STATE_SCHEMA, ACTIONS);
  assert.strictEqual(r.valid, false);
});

// ---------------------------------------------------------------------------
// 2. Runtime gate evaluates the RAW state, not the grid-snapped state
// ---------------------------------------------------------------------------
test('gate on raw cpu=75 blocks ScaleIn for rule "cpu > 70" even though snapped cpu=70', () => {
  const spec = comp.validate({ prohibitions: [{ action: 'ScaleIn', condition: 'cpu > 70' }] }, STATE_SCHEMA, ACTIONS).spec;
  const rule = comp.compile(spec, ACTIONS);
  const raw = { severity: 'none', health: 'healthy', cpu: 75, latency: 300, replicas: 4, deploy_status: 'idle' };
  const grid = enumerateStates();
  const snapped = snapToGrid(raw, grid);
  assert.strictEqual(snapped.cpu, 70);   // the approximation that used to leak into enforcement

  // A policy that proposes ScaleIn everywhere
  const policy = new Map(grid.map(s => [stateKey(s), 'ScaleIn']));
  const common = {
    initialState: raw, policy, V: null, baselineFn: baselineAvailable,
    rewardFn: reward, transitionFn: transition, stateKeyFn: stateKey,
    snapToGridFn: snapToGrid, states: grid, gamma: 0.9, maxSteps: 1,
    rng: () => 0.5,
  };

  const gated = plan.runEpisode({ ...common, ruleAt: () => ({ rule, version: 1 }), gate: true });
  assert.notStrictEqual(gated.trace[0].executed, 'ScaleIn');
  assert.strictEqual(gated.violations, 0);

  const ungated = plan.runEpisode({ ...common, ruleAt: () => ({ rule, version: 1 }), gate: false });
  assert.strictEqual(ungated.trace[0].executed, 'ScaleIn');
  assert.strictEqual(ungated.violations, 1);
});

test('gate replacement is value-based when V is supplied', () => {
  const grid = enumerateStates();
  const rule = comp.compile(comp.validate({ prohibitions: [{ action: 'Deploy', condition: 'severity = sev1' }] }, STATE_SCHEMA, ACTIONS).spec, ACTIONS);
  const raw = { severity: 'sev1', health: 'degraded', cpu: 85, latency: 300, replicas: 3, deploy_status: 'idle' };
  const policy = new Map(grid.map(s => [stateKey(s), 'Deploy']));
  // V: make ScaleOut best at this state so a value-based gate must pick it
  const V = new Map(grid.map(s => [stateKey(s), 0]));
  const rank = plan.rankAdmissible(snapToGrid(raw, grid), new Set(['Restart', 'ScaleOut', 'DoNothing']), { V, rewardFn: reward, transitionFn: transition, stateKeyFn: stateKey, snapToGridFn: snapToGrid, states: grid, gamma: 0.9 });
  assert.ok(['Restart', 'ScaleOut', 'DoNothing'].includes(rank));
  const r = plan.runEpisode({
    initialState: raw, policy, V, baselineFn: baselineAvailable, ruleAt: () => ({ rule, version: 1 }), gate: true,
    rewardFn: reward, transitionFn: transition, stateKeyFn: stateKey, snapToGridFn: snapToGrid, states: grid,
    gamma: 0.9, maxSteps: 1, rng: () => 0.5,
  });
  assert.notStrictEqual(r.trace[0].executed, 'Deploy');
  assert.strictEqual(r.violations, 0);
});

// ---------------------------------------------------------------------------
// 3. Example V.1 is computed, and reproduces the paper's numbers
// ---------------------------------------------------------------------------
test('Example V.1 reproduces paper values (9 vs 5; returns 0.9 and 5)', () => {
  const r = plan.runExample51();
  assert.ok(Math.abs(r.unconstrained.Q_s0.Risky - 9) < 1e-9);
  assert.ok(Math.abs(r.unconstrained.Q_s0.Safe - 5) < 1e-9);
  assert.strictEqual(r.unconstrained.selected, 'Risky');
  assert.ok(Math.abs(r.unconstrained.returnWithGate - 0.9) < 1e-9);
  assert.ok(Math.abs(r.constraintAware.V_I_s1 - 1) < 1e-9);
  assert.ok(Math.abs(r.constraintAware.Q_I_s0.Risky - 0.9) < 1e-9);
  assert.strictEqual(r.constraintAware.selected, 'Safe');
  assert.ok(Math.abs(r.constraintAware.returnWithGate - 5) < 1e-9);
});

// ---------------------------------------------------------------------------
// 4. Fast indexed solver is the same Bellman operator as planner.valueIteration
// ---------------------------------------------------------------------------
test('indexed solver matches generic Jacobi valueIteration on A_I (sev1 rule)', () => {
  const I = exp.compileIntent('sev1');
  const grid = exp.GRID;
  const fast = exp.solveFast({ allowed: (i, a) => I.permitted[i].has(a), epsilon: 1e-10 });
  const generic = plan.valueIteration({
    states: grid,
    admissibleFn: s => new Set([...baselineAvailable(s)].filter(a => I.rule(s).has(a))),
    rewardFn: reward, transitionFn: transition, stateKeyFn: stateKey, snapToGridFn: snapToGrid,
    gamma: 0.9, maxIter: 2000, epsilon: 1e-10,
  });
  let maxDiff = 0;
  for (const s of grid) {
    const k = stateKey(s);
    maxDiff = Math.max(maxDiff, Math.abs(fast.V.get(k) - generic.V.get(k)));
  }
  assert.ok(maxDiff < 1e-5, `max |V_fast - V_generic| = ${maxDiff}`);
});

// ---------------------------------------------------------------------------
// 5. Six-arm and freeze-update outcomes (small batch for speed)
// ---------------------------------------------------------------------------
let six = null;
let freeze = null;
test('six-arm: unconstrained violates; gated and constraint-aware arms do not (raw-state gate)', () => {
  six = exp.runSixArm({ intent: 'sev1', episodes: 200, seed: 7, horizon: 20 });
  const byName = Object.fromEntries(six.summaries.map(s => [s.arm, s]));
  assert.ok(byName['1_unconstrained'].violationsPer100Steps > 0);
  assert.strictEqual(byName['4_runtime_only'].violationsPer100Steps, 0);
  assert.strictEqual(byName['6_combined'].violationsPer100Steps, 0);
  assert.strictEqual(byName['6_combined'].operationalErrors, 0);
});

test('zero-budget CMDP search reaches zero worst-case violation cost', () => {
  assert.strictEqual(six.zeroBudget.satisfied, true);
});

test('snapping leak is real: constraint-aware planner without gate violates "cpu > 70"; raw-state gate does not', () => {
  const snap = exp.runSixArm({ intent: 'cpuScaleIn', episodes: 200, seed: 7, horizon: 20 });
  const by = Object.fromEntries(snap.summaries.map(s => [s.arm, s]));
  assert.ok(by['5_constraint_aware_only'].violationsPer100Steps > 0, 'expected snapping-induced violations without gate');
  assert.strictEqual(by['6_combined'].violationsPer100Steps, 0);
  assert.strictEqual(by['4_runtime_only'].violationsPer100Steps, 0);
});

test('finite penalty: small lambda leaves violations; lambda=100 removes them (Paper III/V.C)', () => {
  const sweep = exp.runPenaltySweep({ intent: 'sev1', lambdas: [1, 100], episodes: 200, seed: 7, horizon: 20 });
  assert.ok(sweep[0].violationsPer100Steps > 0);
  assert.strictEqual(sweep[1].violationsPer100Steps, 0);
});

test('environment: Deploy reaches in_progress and Rollback becomes available; Sev1 is not absorbing under the Sev1 rule', () => {
  const I = exp.compileIntent('sev1');
  const grid = exp.GRID;
  const deployed = env.transition({ severity: 'none', health: 'healthy', cpu: 50, latency: 300, replicas: 3, deploy_status: 'idle' }, 'Deploy');
  assert.ok(deployed.some(t => t.state.deploy_status === 'in_progress'));
  assert.ok(baselineAvailable(deployed[0].state).has('Rollback'));
  const fast = exp.solveFast({ allowed: (i, a) => I.permitted[i].has(a) });
  const sev1 = snapToGrid({ severity: 'sev1', health: 'degraded', cpu: 85, latency: 300, replicas: 3, deploy_status: 'idle' }, grid);
  assert.ok(fast.V.get(stateKey(sev1)) > -150, `constrained V(sev1) = ${fast.V.get(stateKey(sev1))}`);
});

test('freeze update: no-update and stale-no-filter arms violate after switch; stale-with-filter and re-optimised do not', () => {
  freeze = exp.runFreezeUpdate({ episodes: 200, seed: 7, switchStep: 5, horizon: 30 });
  const by = Object.fromEntries(freeze.summaries.map(s => [s.arm, s]));
  assert.ok(by['F1_no_update'].violationsAfterSwitchPer100Steps > 0);
  assert.ok(by['F3_stale_policy_no_filter_after_switch'].violationsAfterSwitchPer100Steps > 0);
  assert.strictEqual(by['F2_stale_policy_with_new_filter'].violationsAfterSwitchPer100Steps, 0);
  assert.strictEqual(by['F4_reoptimised_at_switch_with_filter'].violationsAfterSwitchPer100Steps, 0);
});

test('same seed gives identical episode outcomes (reproducibility)', () => {
  const a = exp.runFreezeUpdate({ episodes: 50, seed: 3, switchStep: 5, horizon: 15 });
  const b = exp.runFreezeUpdate({ episodes: 50, seed: 3, switchStep: 5, horizon: 15 });
  assert.deepStrictEqual(a.rows.map(r => [r.return, r.violations]), b.rows.map(r => [r.return, r.violations]));
});

// ---------------------------------------------------------------------------
// 6. Feasibility (Def. III.2): sev1 rule feasible everywhere; over-tightened rule not
// ---------------------------------------------------------------------------
test('feasibility: sev1 rule feasible on all grid states; over-tightened rule infeasible on all', () => {
  const inf = Object.fromEntries(exp.runSixArm({ intent: 'sev1', episodes: 1, seed: 1, horizon: 1 }).infeasibility.map(r => [r.intent, r.emptyStates]));
  assert.strictEqual(inf.sev1, 0);
  assert.strictEqual(inf.overTight, exp.GRID.length);
});

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
let failed = 0;
for (const r of results) {
  if (r.ok) console.log(`  ok   ${r.name}  (${r.ms} ms)`);
  else { failed++; console.log(`  FAIL ${r.name}\n       ${r.error}`); }
}
console.log(`\n${results.length - failed}/${results.length} passed`);
process.exitCode = failed ? 1 : 0;

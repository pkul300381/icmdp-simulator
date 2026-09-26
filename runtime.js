/**
 * runtime.js
 * Runtime Gate, Feasibility Monitor, Versioned Rule Store, Audit Trace
 *
 * Implements Paper Algorithm 1, Section 4.3, Section 4.4.
 *
 * Three-timescale separation (Paper §4.1):
 *   SLOW:   parse → validate → compile (this module orchestrates)
 *   MEDIUM: planning (planner.js)
 *   FAST:   runtime gate at every action-selection boundary (this module)
 *
 * Trusted enforcement boundary: validator, compiler, rule store, gate.
 * LLM output is UNTRUSTED until it passes deterministic validation.
 */

'use strict';

// ---------------------------------------------------------------------------
// Versioned Rule Store
// Holds the sequence of activated compiled rules with metadata.
// Paper §4.4: "The previously activated rule remains stored for rollback and audit."
// ---------------------------------------------------------------------------
class RuleStore {
  constructor() {
    this.rules   = [];   // [{ version, rule, spec, instruction, activatedAt }]
    this.active  = null; // current active versioned rule
  }

  // Atomic activation — Paper Algorithm 1, Step 4
  activate(rule, spec, instruction) {
    const version = this.rules.length + 1;
    const entry   = { version, rule, spec, instruction, activatedAt: new Date().toISOString() };
    this.rules.push(entry);
    this.active = entry;
    return version;
  }

  getActive() { return this.active; }
  getHistory() { return [...this.rules]; }
  rollback(version) {
    const entry = this.rules.find(r => r.version === version);
    if (entry) { this.active = entry; return true; }
    return false;
  }
}

// ---------------------------------------------------------------------------
// Audit Trace — Paper Algorithm 1, Step 6; §4.3
// Records for each executed action: state, rule version, A_I(s), proposal, executed.
// "The trace records membership in A_I(st), not a statistical property of the policy."
// ---------------------------------------------------------------------------
class AuditTrace {
  constructor() { this.entries = []; }

  log({ step, state, ruleVersion, admissibleSet, proposal, executed, reward, event }) {
    this.entries.push({
      timestamp: new Date().toISOString(),
      step, state: { ...state }, ruleVersion,
      admissibleSet: [...(admissibleSet || [])],
      proposal, executed, reward,
      event: event || 'OK',
    });
  }

  getEntries() { return [...this.entries]; }
  clear()      { this.entries = []; }

  // Summary for display
  summary() {
    const total    = this.entries.filter(e => e.executed !== null).length;
    const filtered = this.entries.filter(e => e.event && e.event.includes('GATE_FILTERED')).length;
    const holds    = this.entries.filter(e => e.event && e.event.includes('HOLD')).length;
    return { total, filtered, holds };
  }
}

// ---------------------------------------------------------------------------
// Runtime Gate — Paper Algorithm 1, Step 6; §4.3
//
// "At each decision, evaluate the active rule on the current state,
//  filter the proposal, execute only an admissible action."
//
// Returns: { executed, filtered: bool, admissibleSet, event }
// ---------------------------------------------------------------------------
function runtimeGate(proposedAction, currentState, ruleStore, baselineAvailableFn, effectiveAdmissibleSetFn) {
  const activeRule = ruleStore.getActive();

  if (!activeRule) {
    // No compiled rule — use baseline availability only
    const A0 = baselineAvailableFn(currentState);
    if (!A0.has(proposedAction)) {
      const fallback = [...A0][0] || null;
      return {
        executed: fallback,
        filtered: true,
        admissibleSet: [...A0],
        ruleVersion: 0,
        event: 'GATE_FILTERED — action not in A₀(s), no intent rule active',
      };
    }
    return { executed: proposedAction, filtered: false, admissibleSet: [...A0], ruleVersion: 0, event: 'OK (no intent rule)' };
  }

  let AI;
  try {
    AI = effectiveAdmissibleSetFn(currentState, baselineAvailableFn, activeRule.rule);
  } catch (e) {
    // Enforcement failure — field missing from state (Paper §4.3)
    return {
      executed: null,
      filtered: true,
      admissibleSet: [],
      ruleVersion: activeRule.version,
      event: `ENFORCEMENT_FAILURE: ${e.message}`,
    };
  }

  // Feasibility monitor (Paper Definition 3.2)
  if (AI.size === 0) {
    return {
      executed: null,
      filtered: true,
      admissibleSet: [],
      ruleVersion: activeRule.version,
      event: 'HOLD — A_I(s) = ∅, requesting clarification or escalation',
    };
  }

  // Filter proposal
  if (AI.has(proposedAction)) {
    return {
      executed: proposedAction,
      filtered: false,
      admissibleSet: [...AI],
      ruleVersion: activeRule.version,
      event: 'OK',
    };
  }

  // Proposal excluded — select first admissible (fallback)
  const fallback = [...AI][0];
  return {
    executed: fallback,
    filtered: true,
    admissibleSet: [...AI],
    ruleVersion: activeRule.version,
    event: `GATE_FILTERED — "${proposedAction}" ∉ A_I(s), executing "${fallback}"`,
  };
}

// ---------------------------------------------------------------------------
// Full Algorithm 1 orchestrator — Paper Algorithm 1
// Steps 1–7 implemented as an async lifecycle manager.
// ---------------------------------------------------------------------------
class ICMDPLifecycle {
  constructor({ schema, actions, baselineAvailableFn, effectiveAdmissibleSetFn,
                parseFn, validateFn, compileFn, feasibilityCheckFn, states }) {
    this.schema                = schema;
    this.actions               = actions;
    this.baselineAvailableFn   = baselineAvailableFn;
    this.effectiveAdmissibleSetFn = effectiveAdmissibleSetFn;
    this.parseFn               = parseFn;
    this.validateFn            = validateFn;
    this.compileFn             = compileFn;
    this.feasibilityCheckFn    = feasibilityCheckFn;
    this.states                = states;

    this.ruleStore  = new RuleStore();
    this.auditTrace = new AuditTrace();
    this.status     = 'IDLE'; // IDLE | HOLD | ACTIVE
    this.lastParseResult  = null;
    this.lastValidResult  = null;
    this.lastCompileResult = null;
    this.feasibilityResult = null;
  }

  // Step 1–3: Parse, validate, compile
  async processInstruction(instruction, apiKey) {
    this.status = 'HOLD'; // Enter hold while processing (Step 1)
    this.lastParseResult   = null;
    this.lastValidResult   = null;
    this.lastCompileResult = null;
    this.feasibilityResult = null;

    // Step 1: LLM parse (SLOW timescale)
    const parseResult = await this.parseFn(instruction, this.schema, this.actions, apiKey);
    this.lastParseResult = parseResult;
    if (!parseResult.success) {
      return { success: false, stage: 'parse', error: parseResult.error };
    }

    // Step 2: Deterministic validation
    const validResult = this.validateFn(parseResult.raw, this.schema, this.actions);
    this.lastValidResult = validResult;
    if (!validResult.valid) {
      return { success: false, stage: 'validate', error: validResult.reason, field: validResult.field };
    }

    // Step 3: Compile once — reusable function
    const intentRule = this.compileFn(validResult.spec, this.actions);
    this.lastCompileResult = { rule: intentRule, spec: validResult.spec };

    return { success: true, rule: intentRule, spec: validResult.spec };
  }

  // Step 4: Feasibility check + atomic activation
  activate(rule, spec, instruction) {
    // Check current-state feasibility (global check over all enumerated states)
    const feasibility = this.feasibilityCheckFn(this.states, this.baselineAvailableFn, rule);
    this.feasibilityResult = feasibility;

    if (!feasibility.feasible) {
      // Do not silently relax — Paper §3.1
      return {
        success: false,
        error: `Intent infeasible: A_I(s) = ∅ in ${feasibility.violations.length} state(s). Revise instruction.`,
        violations: feasibility.violations.slice(0, 5),
      };
    }

    // Atomic activation (Paper §4.4)
    const version = this.ruleStore.activate(rule, spec, instruction);
    this.status = 'ACTIVE';
    return { success: true, version };
  }

  // Steps 5–7: Runtime gate at each decision step
  gate(proposedAction, currentState) {
    const result = runtimeGate(
      proposedAction, currentState,
      this.ruleStore,
      this.baselineAvailableFn,
      this.effectiveAdmissibleSetFn,
    );

    // Log to audit trace (Step 6)
    this.auditTrace.log({
      step: this.auditTrace.entries.length,
      state: currentState,
      ruleVersion: result.ruleVersion,
      admissibleSet: result.admissibleSet,
      proposal: proposedAction,
      executed: result.executed,
      reward: null, // filled by episode runner
      event: result.event,
    });

    return result;
  }

  getAuditTrace()    { return this.auditTrace.getEntries(); }
  getRuleHistory()   { return this.ruleStore.getHistory(); }
  getActiveRule()    { return this.ruleStore.getActive(); }
  clearTrace()       { this.auditTrace.clear(); }
}

if (typeof module !== 'undefined') {
  module.exports = { RuleStore, AuditTrace, runtimeGate, ICMDPLifecycle };
}

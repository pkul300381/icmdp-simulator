/**
 * compiler.js
 * Deterministic Intent Validator and Compiler
 *
 * Implements the intent grammar G, validator vx, and compiler c from
 * Paper Section 4.2, equations 14–17.
 *
 * Grammar (Paper Equation 16):
 *   g   ::= { r₁, ..., rₘ }
 *   r   ::= prohibit(a, φ)
 *   φ   ::= q | ¬φ | (φ ∧ φ) | (φ ∨ φ)
 *   q   ::= h op u
 *
 * Compiler (Paper Equation 17):
 *   c(g)(s) = A \ { aⱼ : evalx(φⱼ, s) = true }
 *
 * The LLM is NOT called here. This module is purely deterministic.
 * It accepts a parsed JSON specification proposed by the LLM parser (api.js)
 * and either validates/compiles it or returns ⊥.
 */

'use strict';

// Admitted operators per type (filling the gap noted in the paper review)
const ADMITTED_OPS = {
  enum:    ['=', '!='],
  number:  ['=', '!=', '<', '<=', '>', '>='],
  integer: ['=', '!=', '<', '<=', '>', '>='],
};

// ---------------------------------------------------------------------------
// Validator vx : G → G_x^val ∪ {⊥}   (Paper Definition 4.1)
//
// Input:  rawSpec — JSON object proposed by LLM parser
//         schema  — STATE_SCHEMA from environment.js
//         actions — ACTIONS array from environment.js
// Output: { valid: true, spec } | { valid: false, reason, field }
// ---------------------------------------------------------------------------
function validate(rawSpec, schema, actions) {
  if (!rawSpec || !Array.isArray(rawSpec.prohibitions)) {
    return { valid: false, reason: 'Specification must have a "prohibitions" array.', field: 'root' };
  }

  const validated = [];

  for (let i = 0; i < rawSpec.prohibitions.length; i++) {
    const r = rawSpec.prohibitions[i];

    // Check action resolves uniquely
    if (!r.action) {
      return { valid: false, reason: `Rule ${i}: missing "action" field.`, field: 'action' };
    }
    const actionMatch = actions.find(a => a.toLowerCase() === r.action.toLowerCase());
    if (!actionMatch) {
      return { valid: false, reason: `Rule ${i}: unknown action "${r.action}". Known: ${actions.join(', ')}.`, field: 'action' };
    }

    // Validate the condition φ
    if (!r.condition) {
      return { valid: false, reason: `Rule ${i}: missing "condition".`, field: 'condition' };
    }

    const condResult = validateCondition(r.condition, schema, i);
    if (!condResult.valid) return condResult;

    validated.push({ action: actionMatch, condition: condResult.ast });
  }

  return { valid: true, spec: { prohibitions: validated } };
}

// ---------------------------------------------------------------------------
// Recursive condition validator — builds AST
// ---------------------------------------------------------------------------
function validateCondition(cond, schema, ruleIdx) {
  // String form: "field op value"
  if (typeof cond === 'string') {
    return validateAtom(cond.trim(), schema, ruleIdx);
  }

  // Object form: { not: φ } | { and: [φ,φ] } | { or: [φ,φ] } | { field, op, value }
  if (typeof cond === 'object' && cond !== null) {
    if (cond.not !== undefined) {
      const inner = validateCondition(cond.not, schema, ruleIdx);
      if (!inner.valid) return inner;
      return { valid: true, ast: { type: 'not', child: inner.ast } };
    }
    if (cond.and !== undefined) {
      if (!Array.isArray(cond.and) || cond.and.length < 2) {
        return { valid: false, reason: `Rule ${ruleIdx}: "and" requires an array of ≥2 conditions.`, field: 'condition' };
      }
      const children = [];
      for (const sub of cond.and) {
        const r = validateCondition(sub, schema, ruleIdx);
        if (!r.valid) return r;
        children.push(r.ast);
      }
      return { valid: true, ast: { type: 'and', children } };
    }
    if (cond.or !== undefined) {
      if (!Array.isArray(cond.or) || cond.or.length < 2) {
        return { valid: false, reason: `Rule ${ruleIdx}: "or" requires an array of ≥2 conditions.`, field: 'condition' };
      }
      const children = [];
      for (const sub of cond.or) {
        const r = validateCondition(sub, schema, ruleIdx);
        if (!r.valid) return r;
        children.push(r.ast);
      }
      return { valid: true, ast: { type: 'or', children } };
    }
    // Atom in object form
    if (cond.field !== undefined) {
      return validateAtomObj(cond, schema, ruleIdx);
    }
  }

  return { valid: false, reason: `Rule ${ruleIdx}: unrecognised condition format.`, field: 'condition' };
}

function validateAtom(str, schema, ruleIdx) {
  // Parse "field op value"
  const opPatterns = ['<=', '>=', '!=', '<', '>', '='];
  let matched = null;
  for (const op of opPatterns) {
    const idx = str.indexOf(op);
    if (idx > 0) {
      matched = { field: str.slice(0, idx).trim(), op, value: str.slice(idx + op.length).trim() };
      break;
    }
  }
  if (!matched) {
    return { valid: false, reason: `Rule ${ruleIdx}: cannot parse atom "${str}". Expected "field op value".`, field: 'condition' };
  }
  // Optional trailing unit token: "latency > 1.2 s"
  const parts = matched.value.split(/\s+/).filter(Boolean);
  if (parts.length > 2) {
    return { valid: false, reason: `Rule ${ruleIdx}: cannot parse value "${matched.value}" in atom "${str}".`, field: 'condition' };
  }
  matched.value = parts[0] ?? '';
  if (parts[1] !== undefined) matched.unit = parts[1];
  return validateAtomObj(matched, schema, ruleIdx);
}

// Unit conversion to each numeric field's canonical unit (Paper §IV-B: "converts quantities to declared canonical units").
// Multiplier maps the stated unit onto the canonical unit of the schema field.
const UNIT_CONVERSIONS = {
  percent: { percent: 1, '%': 1, fraction: 100 },
  ms:      { ms: 1, s: 1000, seconds: 1000 },
};

const NUMERIC_LITERAL = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

function validateAtomObj({ field, op, value, unit }, schema, ruleIdx) {
  const fieldDef = schema[field];
  if (!fieldDef) {
    return { valid: false, reason: `Rule ${ruleIdx}: unknown field "${field}". Known: ${Object.keys(schema).join(', ')}.`, field };
  }
  const admittedOps = ADMITTED_OPS[fieldDef.type] || [];
  if (!admittedOps.includes(op)) {
    return { valid: false, reason: `Rule ${ruleIdx}: operator "${op}" not admitted for ${fieldDef.type} field "${field}". Admitted: ${admittedOps.join(', ')}.`, field };
  }

  if (fieldDef.type === 'enum') {
    if (unit !== undefined) {
      return { valid: false, reason: `Rule ${ruleIdx}: units do not apply to enum field "${field}".`, field };
    }
    if (!fieldDef.values.includes(value)) {
      return { valid: false, reason: `Rule ${ruleIdx}: value "${value}" not in enum for "${field}". Valid: ${fieldDef.values.join(', ')}.`, field };
    }
    return { valid: true, ast: { type: 'atom', field, op, value } };
  }

  // Numeric field: literal must parse as a number (Number('') === 0, so check the literal first)
  const literal = String(value).trim();
  if (!NUMERIC_LITERAL.test(literal)) {
    return { valid: false, reason: `Rule ${ruleIdx}: value "${value}" is not numeric for field "${field}".`, field };
  }
  let num = Number(literal);

  // Units: fields with a declared unit and more than one accepted unit must be stated explicitly
  // when the literal is non-integer, otherwise the magnitude is ambiguous (e.g. 1.2 seconds vs 1.2 ms).
  if (fieldDef.unit && UNIT_CONVERSIONS[fieldDef.unit]) {
    const table = UNIT_CONVERSIONS[fieldDef.unit];
    if (unit === undefined) {
      if (!Number.isInteger(num)) {
        return { valid: false, reason: `Rule ${ruleIdx}: ambiguous magnitude ${literal} for "${field}". State the unit explicitly (${Object.keys(table).join(', ')}).`, field };
      }
      num = num * table[fieldDef.unit];   // canonical unit name is the schema unit, factor 1
    } else {
      if (!(unit in table)) {
        return { valid: false, reason: `Rule ${ruleIdx}: unknown unit "${unit}" for "${field}". Accepted: ${Object.keys(table).join(', ')}.`, field };
      }
      num = num * table[unit];
    }
  } else if (unit !== undefined) {
    return { valid: false, reason: `Rule ${ruleIdx}: field "${field}" has no unit conversions; "${unit}" rejected.`, field };
  }

  if (fieldDef.type === 'integer' && !Number.isInteger(num)) {
    return { valid: false, reason: `Rule ${ruleIdx}: "${field}" is an integer field; ${num} is not an integer.`, field };
  }
  if (num < fieldDef.min || num > fieldDef.max) {
    return { valid: false, reason: `Rule ${ruleIdx}: value ${num} outside declared range [${fieldDef.min}, ${fieldDef.max}] for "${field}".`, field };
  }
  return { valid: true, ast: { type: 'atom', field, op, value: num } };
}

// ---------------------------------------------------------------------------
// Compiler c : G_x^val → R   (Paper Equation 17)
//
// Returns a function I(s) → Set of admissible actions.
// c(g)(s) = A \ { aⱼ : evalx(φⱼ, s) = true }
// ---------------------------------------------------------------------------
function compile(validatedSpec, allActions) {
  const prohibitions = validatedSpec.prohibitions; // [{ action, condition: AST }]

  // The compiled rule is a reusable function evaluable on any state
  function intentRule(s) {
    const excluded = new Set();
    for (const { action, condition } of prohibitions) {
      if (evalCondition(condition, s)) {
        excluded.add(action);
      }
    }
    // Return permitted set: A \ excluded
    return new Set(allActions.filter(a => !excluded.has(a)));
  }

  return intentRule;
}

// ---------------------------------------------------------------------------
// Condition evaluator evalx(φ, s)
// ---------------------------------------------------------------------------
function evalCondition(ast, s) {
  if (ast.type === 'atom') {
    const fieldVal = s[ast.field];
    if (fieldVal === undefined) {
      // Missing field at evaluation time — enforcement failure
      throw new Error(`ENFORCEMENT_FAILURE: field "${ast.field}" not found in state.`);
    }
    switch (ast.op) {
      case '=':  return fieldVal === ast.value;
      case '!=': return fieldVal !== ast.value;
      case '<':  return fieldVal <  ast.value;
      case '<=': return fieldVal <= ast.value;
      case '>':  return fieldVal >  ast.value;
      case '>=': return fieldVal >= ast.value;
      default:   return false;
    }
  }
  if (ast.type === 'not') {
    return !evalCondition(ast.child, s);
  }
  if (ast.type === 'and') {
    return ast.children.every(c => evalCondition(c, s));
  }
  if (ast.type === 'or') {
    return ast.children.some(c => evalCondition(c, s));
  }
  return false;
}

// ---------------------------------------------------------------------------
// Effective admissible set A_I(s) = A₀(s) ∩ I(s)   (Paper Equation 3)
// ---------------------------------------------------------------------------
function effectiveAdmissibleSet(s, baselineAvailableFn, intentRule) {
  const A0 = baselineAvailableFn(s);
  const I  = intentRule(s);
  return new Set([...A0].filter(a => I.has(a)));
}

// ---------------------------------------------------------------------------
// Feasibility check   (Paper Definition 3.2)
// ---------------------------------------------------------------------------
function checkGlobalFeasibility(states, baselineAvailableFn, intentRule) {
  const violations = [];
  for (const s of states) {
    const AI = effectiveAdmissibleSet(s, baselineAvailableFn, intentRule);
    if (AI.size === 0) {
      violations.push(s);
    }
  }
  return { feasible: violations.length === 0, violations };
}

if (typeof module !== 'undefined') {
  module.exports = { validate, compile, evalCondition, effectiveAdmissibleSet, checkGlobalFeasibility };
}

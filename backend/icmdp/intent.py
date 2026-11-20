"""
Intent language G, deterministic validator v_x, compiler c, and parsers f_phi.

Grammar (Paper Eq. 16):
    g   ::= {prohibit(a, phi), ...}
    phi ::= atom | not phi | (phi and phi ...) | (phi or phi ...)
    atom ::= field op value [unit]

Validator returns either a ValidatedSpec or a rejection (bottom). Ambiguity is a rejection,
never a silent default (Paper Sec. IV-B). Compiler c(g)(s) = A \\ {a_j : phi_j(s) true} (Eq. 17).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .model import ACTIONS, SCHEMA, State

ADMITTED_OPS = {
    "enum": ("=", "!="),
    "number": ("=", "!=", "<", "<=", ">", ">="),
    "integer": ("=", "!=", "<", "<=", ">", ">="),
}
UNIT_CONVERSIONS = {
    "percent": {"percent": 1.0, "%": 1.0, "fraction": 100.0},
    "ms": {"ms": 1.0, "s": 1000.0, "seconds": 1000.0},
}
NUMERIC_LITERAL = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
OP_ORDER = ("<=", ">=", "!=", "<", ">", "=")


class EnforcementFailure(Exception):
    """A field needed by a predicate is missing from the state (Paper Sec. IV-C)."""


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Atom:
    field: str
    op: str
    value: Any  # canonical value (float/int for numbers, str for enums)

    def eval(self, s: State) -> bool:
        if not hasattr(s, self.field):
            raise EnforcementFailure(f'field "{self.field}" missing from state')
        v = getattr(s, self.field)
        return {
            "=": v == self.value, "!=": v != self.value, "<": v < self.value,
            "<=": v <= self.value, ">": v > self.value, ">=": v >= self.value,
        }[self.op]

    def to_dict(self):
        return {"type": "atom", "field": self.field, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class Not:
    child: "Cond"

    def eval(self, s):
        return not self.child.eval(s)

    def to_dict(self):
        return {"type": "not", "child": self.child.to_dict()}


@dataclass(frozen=True)
class And:
    children: tuple

    def eval(self, s):
        return all(c.eval(s) for c in self.children)

    def to_dict(self):
        return {"type": "and", "children": [c.to_dict() for c in self.children]}


@dataclass(frozen=True)
class Or:
    children: tuple

    def eval(self, s):
        return any(c.eval(s) for c in self.children)

    def to_dict(self):
        return {"type": "or", "children": [c.to_dict() for c in self.children]}


Cond = Union[Atom, Not, And, Or]


@dataclass(frozen=True)
class Prohibition:
    action: str
    cond: Cond

    def to_dict(self):
        return {"action": self.action, "condition": self.cond.to_dict()}


@dataclass
class ValidatedSpec:
    prohibitions: List[Prohibition]

    def to_dict(self):
        return {"prohibitions": [p.to_dict() for p in self.prohibitions]}


@dataclass
class ValidationResult:
    valid: bool
    spec: Optional[ValidatedSpec] = None
    reason: str = ""
    field: str = ""

    def to_dict(self):
        return {
            "valid": self.valid,
            "reason": self.reason,
            "field": self.field,
            "spec": self.spec.to_dict() if self.spec else None,
        }


class _Reject(Exception):
    def __init__(self, field_name: str, reason: str):
        super().__init__(reason)
        self.field = field_name
        self.reason = reason


# ---------------------------------------------------------------------------
# Validator v_x
# ---------------------------------------------------------------------------
def _atom_from_string(text: str, idx: int) -> Atom:
    text = text.strip()
    for op in OP_ORDER:
        pos = text.find(op)
        if pos > 0:
            fld = text[:pos].strip()
            rest = text[pos + len(op):].strip()
            parts = rest.split()
            if len(parts) > 2:
                raise _Reject("condition", f'rule {idx}: cannot parse value in "{text}"')
            value = parts[0] if parts else ""
            unit = parts[1] if len(parts) == 2 else None
            return _atom_checked(fld, op, value, unit, idx)
    raise _Reject("condition", f'rule {idx}: cannot parse atom "{text}". Expected "field op value".')


def _atom_checked(fld: str, op: str, value: Any, unit: Optional[str], idx: int) -> Atom:
    if fld not in SCHEMA:
        raise _Reject(fld, f'rule {idx}: unknown field "{fld}". Known: {", ".join(SCHEMA)}.')
    fdef = SCHEMA[fld]
    if op not in ADMITTED_OPS[fdef["type"]]:
        raise _Reject(fld, f'rule {idx}: operator "{op}" not admitted for {fdef["type"]} field "{fld}".')

    if fdef["type"] == "enum":
        if unit is not None:
            raise _Reject(fld, f'rule {idx}: units do not apply to enum field "{fld}".')
        if value not in fdef["values"]:
            raise _Reject(fld, f'rule {idx}: value "{value}" not in enum for "{fld}". Valid: {", ".join(fdef["values"])}.')
        return Atom(fld, op, value)

    literal = str(value).strip()
    if not NUMERIC_LITERAL.match(literal):
        raise _Reject(fld, f'rule {idx}: value "{value}" is not numeric for field "{fld}".')
    num = float(literal)
    table = UNIT_CONVERSIONS.get(fdef.get("unit", ""))
    if table is not None:
        if unit is None:
            if not float(num).is_integer():
                raise _Reject(fld, f'rule {idx}: ambiguous magnitude {literal} for "{fld}". '
                                   f'State the unit explicitly ({", ".join(table)}).')
        else:
            if unit not in table:
                raise _Reject(fld, f'rule {idx}: unknown unit "{unit}" for "{fld}". Accepted: {", ".join(table)}.')
            num = num * table[unit]
    elif unit is not None:
        raise _Reject(fld, f'rule {idx}: field "{fld}" has no unit conversions; "{unit}" rejected.')

    if fdef["type"] == "integer":
        if not float(num).is_integer():
            raise _Reject(fld, f'rule {idx}: "{fld}" is an integer field; {num:g} is not an integer.')
        num = int(num)
    if num < fdef["min"] or num > fdef["max"]:
        raise _Reject(fld, f'rule {idx}: value {num:g} outside declared range [{fdef["min"]}, {fdef["max"]}] for "{fld}".')
    return Atom(fld, op, num)


def _cond(raw: Any, idx: int, depth: int = 0) -> Cond:
    if depth > 8:
        raise _Reject("condition", f"rule {idx}: condition nested too deeply.")
    if isinstance(raw, str):
        return _atom_from_string(raw, idx)
    if isinstance(raw, dict):
        if "not" in raw:
            return Not(_cond(raw["not"], idx, depth + 1))
        for key, ctor in (("and", And), ("or", Or)):
            if key in raw:
                kids = raw[key]
                if not isinstance(kids, list) or len(kids) < 2:
                    raise _Reject("condition", f'rule {idx}: "{key}" requires an array of at least 2 conditions.')
                return ctor(tuple(_cond(k, idx, depth + 1) for k in kids))
        if "field" in raw:
            return _atom_checked(str(raw["field"]), str(raw.get("op", "")), raw.get("value", ""),
                                 raw.get("unit"), idx)
    raise _Reject("condition", f"rule {idx}: unrecognised condition format.")


def validate(raw: Any) -> ValidationResult:
    """v_x : G -> G_val_x  or  bottom. Never raises on bad input; returns valid=False."""
    try:
        if not isinstance(raw, dict) or not isinstance(raw.get("prohibitions"), list):
            raise _Reject("root", 'specification must be an object with a "prohibitions" array.')
        if len(raw["prohibitions"]) == 0:
            raise _Reject("root", "specification has no prohibitions; nothing to compile.")
        out: List[Prohibition] = []
        for i, r in enumerate(raw["prohibitions"]):
            if not isinstance(r, dict) or not r.get("action"):
                raise _Reject("action", f'rule {i}: missing "action".')
            matches = [a for a in ACTIONS if a.lower() == str(r["action"]).lower()]
            if len(matches) != 1:
                raise _Reject("action", f'rule {i}: unknown action "{r["action"]}". Known: {", ".join(ACTIONS)}.')
            if "condition" not in r:
                raise _Reject("condition", f'rule {i}: missing "condition".')
            out.append(Prohibition(matches[0], _cond(r["condition"], i)))
        return ValidationResult(True, ValidatedSpec(out))
    except _Reject as e:
        return ValidationResult(False, None, e.reason, e.field)


# ---------------------------------------------------------------------------
# Compiler c and compiled rule
# ---------------------------------------------------------------------------
class CompiledRule:
    """c(g): state -> permitted action set I(s). Pure function of the state."""

    def __init__(self, spec: ValidatedSpec):
        self.spec = spec

    def excluded(self, s: State) -> frozenset:
        return frozenset(p.action for p in self.spec.prohibitions if p.cond.eval(s))

    def permitted(self, s: State) -> frozenset:
        ex = self.excluded(s)
        return frozenset(a for a in ACTIONS if a not in ex)

    def admissible(self, s: State, baseline_set: frozenset) -> frozenset:
        """A_I(s) = A0(s) ∩ I(s)  (Paper Eq. 3)."""
        return baseline_set & self.permitted(s)


def compile_spec(vr: ValidationResult) -> CompiledRule:
    if not vr.valid or vr.spec is None:
        raise ValueError("cannot compile a rejected specification")
    return CompiledRule(vr.spec)


# ---------------------------------------------------------------------------
# Parsers f_phi (untrusted proposers)
# ---------------------------------------------------------------------------
PRESET_INSTRUCTIONS = {
    "sev1": "Do not deploy while a Severity-1 production incident is active",
    "freeze": "Freeze all production deployments until further notice",
    "cpuScaleIn": "Never scale in when CPU is above 70 percent",
    "overTight": "Prohibit every action whenever CPU is non-negative",
}


def local_parse(instruction: str) -> Dict[str, Any]:
    """Deterministic keyword parser for offline demos. Returns {'raw': ...} or {'error': ...}."""
    norm = instruction.lower().strip()
    if len(norm) < 5 or any(w in norm for w in ("faster", "better", "optimize", "be safe")):
        return {"error": "Instruction is under-specified. Name an action to prohibit and a condition."}
    if ("sev1" in norm or ("severity" in norm and ("1" in norm or "one" in norm))) and "deploy" in norm:
        return {"raw": {"prohibitions": [{"action": "Deploy", "condition": "severity = sev1"}]}}
    if "freeze" in norm and "deploy" in norm:
        return {"raw": {"prohibitions": [{"action": "Deploy", "condition": {"or": [
            "severity = none", "severity = sev2", "severity = sev1"]}}]}}
    if "scale" in norm and ("in" in norm.split() or "down" in norm) and ("cpu" in norm or "latency" in norm):
        m = re.search(r"cpu[^0-9]*(\d+)", norm)
        if m:
            op = "<" if any(w in norm for w in ("below", "under", "less")) else ">"
            return {"raw": {"prohibitions": [{"action": "ScaleIn", "condition": f"cpu {op} {m.group(1)}"}]}}
        return {"raw": {"prohibitions": [{"action": "ScaleIn", "condition": {"or": ["cpu > 70", "latency > 800"]}}]}}
    if any(p in norm for p in ("prohibit every", "prohibit all", "block everything")):
        return {"raw": {"prohibitions": [{"action": a, "condition": "cpu >= 0"} for a in ACTIONS]}}
    for action in ACTIONS:
        if action.lower() in norm:
            m = re.search(r"(cpu|latency|replicas)\D*(\d+)", norm)
            if m:
                op = ">" if any(w in norm for w in ("above", "exceed", "greater", "over")) else "<"
                return {"raw": {"prohibitions": [{"action": action, "condition": f"{m.group(1)} {op} {m.group(2)}"}]}}
    return {"error": "Could not parse locally. Use a preset, or configure the LLM parser."}


LLM_SYSTEM_PROMPT = """You translate an operations instruction into a JSON specification. You only propose;
a deterministic validator decides. Output ONLY JSON.
Actions: {actions}
Fields: severity (none, sev2, sev1); health (healthy, degraded, critical); cpu (percent 0-100);
latency (ms 0-5000); replicas (integer 1-10); deploy_status (idle, in_progress).
Format: {{"prohibitions":[{{"action":"<Action>","condition":<cond>}}]}}
cond is "field op value" (optionally with a unit token, e.g. "latency > 1.2 s"), or
{{"not":cond}}, {{"and":[cond,...]}}, {{"or":[cond,...]}}. Operators: enum = !=; numbers = = != < <= > >=.
A prohibition with a TRUE condition excludes that action in that state.
If the instruction is ambiguous or references unknown fields/actions, output {{"error":"reason"}}."""


def llm_parse(instruction: str, model: str = None, api_key: str = None, timeout: float = 20.0) -> Dict[str, Any]:
    """Anthropic Messages API adapter. The output is untrusted and must pass validate()."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"error": "No ANTHROPIC_API_KEY configured; use the local parser."}
    model = model or os.environ.get("ICMDP_LLM_MODEL", "claude-sonnet-5-5")
    body = json.dumps({
        "model": model,
        "max_tokens": 600,
        "system": LLM_SYSTEM_PROMPT.format(actions=", ".join(ACTIONS)),
        "messages": [{"role": "user", "content": f'Instruction: "{instruction}"'}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return {"error": f"LLM parser unavailable: {e}"}
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return {"error": f"LLM output was not JSON: {text[:160]}"}
    if "error" in parsed:
        return {"error": f"Parser rejected instruction: {parsed['error']}"}
    return {"raw": parsed}

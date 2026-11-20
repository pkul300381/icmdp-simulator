"""
Property-style and adversarial tests. Deterministic (fixed seeds), so they run the same in CI.

  - parser gold set: natural-language instructions -> validated rules (or clean rejection)
  - validator fuzz: malformed specifications are rejected and never raise
  - model invariants: every transition distribution sums to one and stays on the grid
  - runtime invariant: with the gate on, no executed action is ever excluded by the active rule
  - planner invariant: the constraint-aware policy only selects admissible actions
"""
import random

import numpy as np
import pytest

from icmdp import experiments as E
from icmdp import intent as I
from icmdp.model import ACTIONS, GRID, GRID_INDEX, State, baseline, model, snap, transition
from icmdp.runtime import RuleRef, run_episode

SEV1 = "Do not deploy while a Severity-1 production incident is active"


def compiled(text):
    out = I.local_parse(text)
    assert "raw" in out, out
    vr = I.validate(out["raw"])
    assert vr.valid, vr.reason
    return I.compile_spec(vr)


# ---------------------------------------------------------------------------
# Parser gold set
# ---------------------------------------------------------------------------
GOLD_VALID = [
    (SEV1, "Deploy", "severity = sev1"),
    ("Never scale in when CPU is below 40 percent", "ScaleIn", "cpu < 40 percent"),
    ("Never scale in when CPU is above 70 percent", "ScaleIn", "cpu > 70 percent"),
    ("Never scale in when latency is above 800 ms", "ScaleIn", "latency > 800 ms"),
]


@pytest.mark.parametrize("text, action, cond", GOLD_VALID)
def test_gold_instructions_produce_expected_prohibition(text, action, cond):
    out = I.local_parse(text)
    prohibitions = out["raw"]["prohibitions"]
    assert prohibitions[0]["action"] == action
    assert prohibitions[0]["condition"] == cond
    assert I.validate(out["raw"]).valid


@pytest.mark.parametrize("text", [
    "Make it faster and safer",          # under-specified
    "hi",                                # too short
    "Please optimize the pipeline",      # no action, no condition
])
def test_underspecified_instructions_are_held_not_compiled(text):
    out = I.local_parse(text)
    assert "error" in out and "raw" not in out


def test_freeze_instruction_covers_every_severity():
    rule = compiled("Freeze all production deployments until further notice")
    for s in GRID[:50]:
        assert "Deploy" not in rule.permitted(s)


# ---------------------------------------------------------------------------
# Validator fuzz: malformed input must be rejected, never raise
# ---------------------------------------------------------------------------
def _spec(cond, action="Deploy"):
    return {"prohibitions": [{"action": action, "condition": cond}]}


MALFORMED = [
    {"prohibitions": "Deploy"},
    {"prohibitions": []},
    {"prohibitions": [{"action": "Deploy"}]},
    _spec({"and": ["severity = sev1"]}),                  # and needs two or more
    _spec("severity = sev9"),                             # enum value not in declared set
    _spec("severity > sev1"),                             # order operator on enum
    _spec("cpu > 1e9 percent"),                           # outside declared range
    _spec("replicas > 2.5"),                              # integer field, non-integer literal
    _spec("cpu > nan percent"),                           # not a numeric literal
    _spec("cpu > 70 percent extra"),                      # trailing garbage
    _spec("cpu > 70 parsecs"),                            # unknown unit
    _spec(123),                                           # unrecognised condition type
    _spec("latency > 2"),                                 # bare number on a unit-bearing field
    _spec("severity = sev1 ; DROP TABLE intents"),        # injection-shaped value
    {"prohibitions": [{"action": "Deploy; rm -rf /", "condition": "severity = sev1"}]},
    {"prohibitions": [{"action": "Deploy", "condition": "severity = sev1"}, 42]},
    None,
    "prohibit everything",
]


@pytest.mark.parametrize("raw", MALFORMED, ids=lambda r: repr(r)[:60])
def test_malformed_specs_are_rejected_without_raising(raw):
    vr = I.validate(raw)
    assert vr.valid is False
    assert vr.reason


def test_deep_nesting_is_rejected():
    cond = "severity = sev1"
    for _ in range(10):
        cond = {"not": cond}
    assert I.validate(_spec(cond)).valid is False


def test_valid_nested_condition_round_trips():
    cond = {"and": ["severity = sev1", {"not": "health = healthy"}]}
    vr = I.validate(_spec(cond))
    assert vr.valid
    rule = I.compile_spec(vr)
    hit = State("sev1", "degraded", 50.0, 300.0, 3, "idle")
    miss = State("none", "degraded", 50.0, 300.0, 3, "idle")
    assert "Deploy" not in rule.permitted(hit)
    assert "Deploy" in rule.permitted(miss)


# ---------------------------------------------------------------------------
# Model invariants
# ---------------------------------------------------------------------------
def test_transition_distributions_sum_to_one_and_stay_on_grid():
    for s in GRID:
        for a in baseline(s):
            outcomes = transition(s, a)
            total = sum(p for _, p in outcomes)
            assert abs(total - 1.0) < 1e-9, (s, a, total)
            for ns, p in outcomes:
                assert p >= 0
                assert snap(ns) in GRID_INDEX


# ---------------------------------------------------------------------------
# Runtime invariant: with the gate on, nothing excluded by the rule is executed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gate_never_executes_an_excluded_action(seed):
    rule = compiled(SEV1)
    ref = RuleRef(rule, 1)
    rng = np.random.default_rng(seed)
    m = model()
    policy = rng.integers(-1, len(ACTIONS), size=len(GRID))   # arbitrary, often inadmissible proposals
    starts = random.Random(seed).sample(GRID, 25)
    for i, st in enumerate(starts):
        res = run_episode(st, m, policy_at=lambda t: policy, V=None,
                          enforce_at=lambda t: ref, truth_at=lambda t: ref,
                          gate_at=lambda t: True, horizon=12, seed=seed * 100 + i)
        assert res.violations == 0
        assert res.operational_errors == 0
        for step in res.steps:
            if step.executed is None:
                continue
            raw = State(**{k: v for k, v in step.state.items()})
            assert step.executed in rule.admissible(raw, baseline(raw))


# ---------------------------------------------------------------------------
# Planner invariant: the constraint-aware policy stays inside A_I(s)
# ---------------------------------------------------------------------------
def test_constraint_aware_policy_selects_only_admissible_actions():
    plan = E.build_plan("sev1", compiled(SEV1))
    sol = plan.solve("constraint_aware")
    allowed = plan.avail & plan.permitted
    for i in range(len(GRID)):
        k = int(sol.policy[i])
        if allowed[i].any():
            assert k >= 0 and allowed[i, k], GRID[i]
        else:
            assert k == -1

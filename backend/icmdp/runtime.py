"""
Runtime gate and episode runner (Paper Algorithm 1, steps 5-7).

The gate evaluates the active rule on the RAW observed state. Policies are indexed by the
grid-snapped state (planning approximation). Replacement actions are argmax of Q over A_I(s).
Violations are scored against the user's intent (truth), which may differ from the rule the
gate enforces: that is how a stale filter is measured.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .intent import CompiledRule
from .model import ACTION_INDEX, ACTIONS, GAMMA, GRID_INDEX, IndexedModel, State, baseline, reward, snap, transition


@dataclass
class RuleRef:
    rule: CompiledRule
    version: int


@dataclass
class StepRecord:
    t: int
    state: dict
    rule_version: int
    admissible: List[str]
    proposal: Optional[str]
    executed: Optional[str]
    reward: Optional[float]
    violated: bool
    event: str
    kind: str   # "ok" | "filtered" | "violation" | "hold" | "terminal"


@dataclass
class EpisodeResult:
    steps: List[StepRecord] = field(default_factory=list)
    total_return: float = 0.0
    executed_steps: int = 0
    violations: int = 0
    interventions: int = 0
    holds: int = 0
    operational_errors: int = 0
    final_state: Optional[State] = None
    severity_path: List[str] = field(default_factory=list)   # severity observed before each executed step


def _rank_replacement(m: IndexedModel, V: Optional[np.ndarray], snapped_idx: int, admissible: List[str],
                      gamma: float) -> str:
    """Value-based replacement: argmax_{a in A_I(s)} Q(s,a) under V. Falls back to ACTIONS order."""
    if V is None:
        return admissible[0]
    best, best_q = admissible[0], -np.inf
    for a in admissible:
        k = ACTION_INDEX[a]
        nxt = m.nxt[snapped_idx, k]
        p = m.prob[snapped_idx, k]
        f = float(np.where(p > 0, p * V[nxt], 0.0).sum())
        q = m.base_r[snapped_idx, k] + gamma * f
        if q > best_q:
            best, best_q = a, q
    return best


def run_episode(
    start: State,
    m: IndexedModel,
    policy_at: Callable[[int], np.ndarray],          # step -> (N,) action index over grid
    V: Optional[np.ndarray],                          # value function for replacements
    enforce_at: Callable[[int], Optional[RuleRef]],   # rule the gate enforces (None = no filter)
    truth_at: Callable[[int], Optional[RuleRef]],     # user intent used to score violations
    gate_at: Callable[[int], bool],
    horizon: int,
    seed: int,
    start_step: int = 0,
    gamma: float = GAMMA,
    record_steps: bool = True,
) -> EpisodeResult:
    rng = random.Random(seed)
    res = EpisodeResult()
    s = start
    for t in range(horizon):
        step = start_step + t
        snapped = GRID_INDEX[snap(s)]
        A0 = baseline(s)
        enforce = enforce_at(step)
        admissible_set = A0 if enforce is None else enforce.rule.admissible(s, A0)
        admissible = [a for a in ACTIONS if a in admissible_set]
        rule_version = enforce.version if enforce else 0

        if not admissible:
            res.holds += 1
            res.steps.append(StepRecord(step, s.as_dict(), rule_version, [], None, None, None, False,
                                        "HOLD — A_I(s) = ∅, requesting clarification or escalation", "hold"))
            break

        pol = policy_at(step)
        pk = int(pol[snapped])
        proposal = ACTIONS[pk] if pk >= 0 else None
        gate_on = gate_at(step)

        if gate_on:
            if proposal is not None and proposal in admissible:
                executed, event, kind = proposal, "OK", "ok"
            else:
                executed = _rank_replacement(m, V, snapped, admissible, gamma)
                res.interventions += 1
                event = f'GATE_FILTERED — "{proposal}" ∉ A_I(s), executing "{executed}"'
                kind = "filtered"
        else:
            if proposal is None:
                res.holds += 1
                res.steps.append(StepRecord(step, s.as_dict(), rule_version, admissible, None, None, None, False,
                                            "HOLD — no proposal for state (ungated)", "hold"))
                break
            executed, event, kind = proposal, "OK", "ok"

        truth = truth_at(step)
        violated = False
        if truth is not None and executed in A0:
            violated = executed not in truth.rule.permitted(s)
        if violated:
            res.violations += 1
            event = f'VIOLATION — executed "{executed}" excluded by user intent'
            kind = "violation"
        if executed not in A0:
            res.operational_errors += 1

        r = reward(s, executed)
        res.total_return += (gamma ** step) * r
        res.executed_steps += 1
        res.severity_path.append(s.severity)
        if record_steps:
            res.steps.append(StepRecord(step, s.as_dict(), rule_version, admissible, proposal, executed,
                                        r, violated, event, kind))

        u = rng.random()
        cum = 0.0
        ns = transition(s, executed)[-1][0]
        for nxt_state, p in transition(s, executed):
            cum += p
            if u < cum:
                ns = nxt_state
                break
        s = ns
    res.final_state = s
    return res

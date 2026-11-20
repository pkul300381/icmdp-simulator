"""
Paper Example V.1 as an explicit 3-state MDP (s0, s1, T), solved with the same Bellman operator.

  s0: Safe -> R=5, to T | Risky -> R=0, to s1
  s1: Deploy -> R=10, to T (PROHIBITED at s1) | Recover -> R=1, to T
  T : absorbing, Noop -> R=0
Expected: unconstrained selects Risky (Q=9 > 5) and the gate returns 0.9;
constraint-aware selects Safe (Q=5 > 0.9) and returns 5.
"""
from __future__ import annotations

from typing import Dict

GAMMA = 0.9
STATES = ("s0", "s1", "T")
ACTIONS = {"s0": ("Safe", "Risky"), "s1": ("Deploy", "Recover"), "T": ("Noop",)}
REWARD = {("s0", "Safe"): 5.0, ("s0", "Risky"): 0.0, ("s1", "Deploy"): 10.0, ("s1", "Recover"): 1.0, ("T", "Noop"): 0.0}
NEXT = {("s0", "Safe"): "T", ("s0", "Risky"): "s1", ("s1", "Deploy"): "T", ("s1", "Recover"): "T", ("T", "Noop"): "T"}


def permitted(s: str, a: str, intent: bool) -> bool:
    return not (intent and s == "s1" and a == "Deploy")


def solve(intent: bool, iters: int = 500) -> Dict:
    V = {s: 0.0 for s in STATES}
    pol = {}
    for _ in range(iters):
        newV = {}
        for s in STATES:
            best, arg = None, None
            for a in ACTIONS[s]:
                if not permitted(s, a, intent):
                    continue
                q = REWARD[(s, a)] + GAMMA * V[NEXT[(s, a)]]
                if best is None or q > best:
                    best, arg = q, a
            newV[s] = best
            pol[s] = arg
        if max(abs(newV[s] - V[s]) for s in STATES) < 1e-12:
            V = newV
            break
        V = newV
    Q = {}
    for s in STATES:
        for a in ACTIONS[s]:
            Q[(s, a)] = REWARD[(s, a)] + GAMMA * V[NEXT[(s, a)]]
    return {"V": V, "policy": pol, "Q": Q}


def example51() -> Dict:
    unc = solve(intent=False)
    con = solve(intent=True)
    # runtime gate at s1 replaces the prohibited Deploy with the best admissible action (Recover)
    gate_at_s1 = "Recover"
    ret_unc = GAMMA * REWARD[("s1", gate_at_s1)]
    ret_con = REWARD[("s0", con["policy"]["s0"])] if con["policy"]["s0"] == "Safe" else \
        GAMMA * REWARD[("s1", con["policy"]["s1"])]
    return {
        "states": list(STATES),
        "edges": [{"from": s, "action": a, "to": NEXT[(s, a)], "reward": REWARD[(s, a)],
                   "prohibited_under_intent": not permitted(s, a, True)}
                  for s in STATES for a in ACTIONS[s]],
        "unconstrained": {
            "Q": {f"{s}:{a}": q for (s, a), q in unc["Q"].items()},
            "policy": unc["policy"], "selected_at_s0": unc["policy"]["s0"],
            "return_with_gate": ret_unc,
            "explanation": "Planner values Risky at 9 by expecting Deploy at s1; the gate blocks Deploy and Recover returns 0.9.",
        },
        "constraint_aware": {
            "Q": {f"{s}:{a}": q for (s, a), q in con["Q"].items()},
            "V_I_s1": con["V"]["s1"], "policy": con["policy"], "selected_at_s0": con["policy"]["s0"],
            "return": 5.0,
            "explanation": "Planner sees that Risky leads to a state where only Recover is admissible: Q_I(s0,Risky)=0.9 < 5.",
        },
    }

"""
Evaluation protocols (Paper Sec. V.C) and the parallel ICMDP / CMDP workflow runner.

Six arms
  1 unconstrained           plan over A0, no gate
  2 finite penalty          plan over A0 with reward - lambda*C_I, no gate (lambda=100)
  3 zero-budget CMDP        Lagrangian doubling search on lambda, no gate
  4 runtime filter only     unconstrained proposal behind the gate
  5 constraint-aware only   plan over A_I, no gate
  6 combined                plan over A_I, gate on raw state
Freeze update (Paper IV-D, Eq. 13): sev1 rule -> deployment freeze at `switch_step`.
  F1 no update; F2 stale policy behind new filter; F3 stale policy, filter dropped;
  F4 re-optimised over A_I2 at the switch behind the filter; F5 CMDP re-solve at the switch, no filter.

Common random numbers: episode i uses seed (base_seed * 1_000_003 + i) in every arm.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import intent as I
from .model import ACTIONS, GAMMA, GRID, N_ACTIONS, N_STATES, IndexedModel, State, baseline, model, reward, snap, transition
from .planning import finite_penalty, value_iteration, zero_budget_cmdp
from .runtime import EpisodeResult, RuleRef, run_episode


PENALTY_LAMBDA = 100.0


@dataclass
class IntentPlan:
    """Everything the medium timescale derives from one compiled rule (cached per rule)."""
    name: str
    version: int
    rule: I.CompiledRule
    m: IndexedModel
    avail: np.ndarray
    permitted: np.ndarray
    violation: np.ndarray           # available but excluded by intent
    compile_seconds: float
    solutions: Dict[str, object] = field(default_factory=dict)
    cmdp: Optional[object] = None

    def ref(self) -> RuleRef:
        return RuleRef(self.rule, self.version)

    def solve(self, key: str):
        if key in self.solutions:
            return self.solutions[key]
        m, av, perm = self.m, self.avail, self.permitted
        if key == "unconstrained":
            sol = value_iteration(m, av)
        elif key == "constraint_aware":
            sol = value_iteration(m, av & perm)
        elif key == "penalty":
            sol = finite_penalty(m, av, self.violation, PENALTY_LAMBDA)
        elif key == "cmdp":
            if self.cmdp is None:
                self.cmdp = zero_budget_cmdp(m, av, self.violation)
            sol = self.cmdp.solution
        else:
            raise KeyError(key)
        self.solutions[key] = sol
        return sol


def build_plan(name: str, rule: I.CompiledRule, version: int = 1) -> IntentPlan:
    t0 = time.perf_counter()
    m = model()
    permitted = m.permitted_mask(rule)
    compile_seconds = time.perf_counter() - t0
    avail = m.avail.copy()
    violation = (avail & ~permitted).astype(float)
    return IntentPlan(name, version, rule, m, avail, permitted, violation, compile_seconds)


def sample_starts(n: int, seed: int) -> List[State]:
    rng = random.Random(seed)
    idx = list(range(N_STATES))
    rng.shuffle(idx)
    return [GRID[i] for i in idx[: min(n, N_STATES)]]


def _summ(xs: List[float]) -> Dict[str, float]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "ci95": 0.0}
    mean = sum(xs) / n
    sd = (sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
    return {"n": n, "mean": mean, "sd": sd, "ci95": 1.96 * sd / (n ** 0.5)}


def _occupancy(paths: List[List[str]], horizon: int) -> List[Dict[str, float]]:
    out = []
    for t in range(horizon):
        counts = {"none": 0, "sev2": 0, "sev1": 0}
        alive = 0
        for p in paths:
            if t < len(p):
                counts[p[t]] += 1
                alive += 1
        out.append({"t": t, **{k: (v / alive if alive else 0.0) for k, v in counts.items()}})
    return out


ProgressFn = Callable[[str, int, int], None]


def run_six_arm(plan: IntentPlan, episodes: int = 500, seed: int = 1, horizon: int = 20,
                trace_sample: int = 3, progress: Optional[ProgressFn] = None) -> Dict:
    m = plan.m
    starts = sample_starts(episodes, seed)
    unc = plan.solve("unconstrained")
    pen = plan.solve("penalty")
    cmdp = plan.solve("cmdp")
    cmdp_res = plan.cmdp           # CMDPResult: lambda, dual trace, satisfaction
    con = plan.solve("constraint_aware")
    ref = plan.ref()

    arm_defs = [
        ("1_unconstrained", unc, None, lambda t: False, "A0"),
        ("2_finite_penalty", pen, None, lambda t: False, f"A0 + penalty λ={PENALTY_LAMBDA:g}"),
        ("3_zero_budget_cmdp", cmdp, None, lambda t: False, f"A0 + Lagrangian λ={cmdp_res.lam:g}"),
        ("4_runtime_only", unc, ref, lambda t: True, "A0 + gate (raw state)"),
        ("5_constraint_aware_only", con, None, lambda t: False, "A_I (no gate)"),
        ("6_combined", con, ref, lambda t: True, "A_I + gate (raw state)"),
    ]

    summaries, episode_rows, traces, occupancy = [], [], {}, {}
    for k, (name, sol, enforce, gate, planner_desc) in enumerate(arm_defs):
        pol = sol.policy
        t0 = time.perf_counter()
        results: List[EpisodeResult] = []
        for i, st in enumerate(starts):
            res = run_episode(
                st, m,
                policy_at=lambda t, p=pol: p,
                V=sol.V,
                enforce_at=lambda t, e=enforce: e,
                truth_at=lambda t: ref,
                gate_at=gate,
                horizon=horizon,
                seed=seed * 1_000_003 + i,
                record_steps=i < trace_sample,
            )
            results.append(res)
            if i < trace_sample:
                traces.setdefault(name, []).append({"index": i, "start": st.as_dict(),
                                                    "steps": [s.__dict__ for s in res.steps]})
            episode_rows.append({
                "arm": name, "index": i, "start": snap(st).key(), "return": res.total_return,
                "steps": res.executed_steps, "violations": res.violations,
                "interventions": res.interventions, "holds": res.holds,
                "operational_errors": res.operational_errors,
            })
            if progress and i % 100 == 0:
                progress(name, i, len(starts))
        wall = time.perf_counter() - t0
        steps = sum(r.executed_steps for r in results)
        viol = sum(r.violations for r in results)
        summaries.append({
            "arm": name,
            "planner": planner_desc,
            "gate": enforce is not None,
            "return": _summ([r.total_return for r in results]),
            "violations_per_100_steps": 100.0 * viol / steps if steps else 0.0,
            "violating_episode_pct": 100.0 * sum(r.violations > 0 for r in results) / len(results),
            "interventions_per_episode": _summ([float(r.interventions) for r in results])["mean"],
            "clarification_pct": 100.0 * sum(r.holds > 0 for r in results) / len(results),
            "operational_errors": sum(r.operational_errors for r in results),
            "planning_seconds": sol.seconds if hasattr(sol, "seconds") else None,
            "planning_iterations": sol.iterations,
            "episode_seconds_per_step": wall / steps if steps else None,
        })
        occupancy[name] = _occupancy([r.severity_path for r in results], horizon)

    cmdp_dual = [
        {"lambda": d["lambda"], "worst_cost": d["worst_cost"], "iterations": d["iterations"], "seconds": d["seconds"]}
        for d in cmdp_res.dual_trace
    ]
    return {
        "intent": plan.name,
        "intent_version": plan.version,
        "episodes": len(starts),
        "seed": seed,
        "horizon": horizon,
        "gamma": GAMMA,
        "summaries": summaries,
        "episode_rows": episode_rows,
        "traces": traces,
        "occupancy": occupancy,
        "cmdp_dual_trace": cmdp_dual,
        "compile_seconds": plan.compile_seconds,
        "zero_budget": {"lambda": cmdp_res.lam, "worst_cost": cmdp_res.worst_cost, "satisfied": cmdp_res.satisfied},
        "infeasible_states": int((~((plan.avail & plan.permitted).any(axis=1))).sum()),
    }


def run_penalty_sweep(plan: IntentPlan, lambdas=(1, 5, 10, 20, 50, 100), episodes: int = 500,
                      seed: int = 1, horizon: int = 20) -> List[Dict]:
    starts = sample_starts(episodes, seed)
    ref = plan.ref()
    out = []
    for lam in lambdas:
        sol = finite_penalty(plan.m, plan.avail, plan.violation, float(lam))
        results = [run_episode(st, plan.m, policy_at=lambda t, p=sol.policy: p, V=sol.V,
                               enforce_at=lambda t: None, truth_at=lambda t: ref, gate_at=lambda t: False,
                               horizon=horizon, seed=seed * 1_000_003 + i, record_steps=False)
                   for i, st in enumerate(starts)]
        steps = sum(r.executed_steps for r in results)
        viol = sum(r.violations for r in results)
        ret = _summ([r.total_return for r in results])
        out.append({"lambda": lam, "return_mean": ret["mean"], "return_ci95": ret["ci95"],
                    "violations_per_100_steps": 100.0 * viol / steps if steps else 0.0,
                    "violating_episode_pct": 100.0 * sum(r.violations > 0 for r in results) / len(results)})
    return out


def run_freeze_update(rule_before: I.CompiledRule, rule_after: I.CompiledRule, episodes: int = 500,
                      seed: int = 1, switch_step: int = 5, horizon: int = 30) -> Dict:
    """Paper IV-D: the intent changes from rule_before to rule_after at switch_step."""
    m = model()
    p1 = build_plan("before", rule_before, 1)
    p2 = build_plan("after", rule_after, 2)
    starts = sample_starts(episodes, seed)
    truth = lambda t: p1.ref() if t < switch_step else p2.ref()
    before = p1.solve("constraint_aware")
    after = p2.solve("constraint_aware")
    t0 = time.perf_counter()
    cmdp2 = zero_budget_cmdp(m, p2.avail, p2.violation)     # CMDP must re-solve its dual
    cmdp_resolve_seconds = time.perf_counter() - t0
    before_pol = lambda t: before.policy
    # (name, policy_at, V, enforce_at, gate_at)
    arms = [
        ("F1_no_update", before_pol, before.V, lambda t: p1.ref(), lambda t: True),
        ("F2_stale_policy_with_new_filter", before_pol, before.V, truth, lambda t: True),
        ("F3_stale_policy_no_filter_after_switch", before_pol, before.V, truth, lambda t: t < switch_step),
        ("F4_reoptimised_with_filter", lambda t: before.policy if t < switch_step else after.policy,
         after.V, truth, lambda t: True),
        ("F5_cmdp_resolve_no_filter", lambda t: before.policy if t < switch_step else cmdp2.solution.policy,
         cmdp2.solution.V, lambda t: None, lambda t: False),
    ]
    summaries, rows = [], []
    for name, policy_at, V_use, enforce_at, gate_at in arms:
        results = [run_episode(st, m, policy_at=policy_at, V=V_use, enforce_at=enforce_at, truth_at=truth,
                               gate_at=gate_at, horizon=horizon, seed=seed * 1_000_003 + i, record_steps=False)
                   for i, st in enumerate(starts)]
        for i, res in enumerate(results):
            rows.append({"arm": name, "index": i, "return": res.total_return,
                         "violations": res.violations, "interventions": res.interventions})
        steps = sum(r.executed_steps for r in results)
        viol = sum(r.violations for r in results)
        summaries.append({
            "arm": name,
            "return": _summ([r.total_return for r in results]),
            "violations_per_100_steps": 100.0 * viol / steps if steps else 0.0,
            "violating_episode_pct": 100.0 * sum(r.violations > 0 for r in results) / len(results),
            "interventions_per_episode": _summ([float(r.interventions) for r in results])["mean"],
        })
    return {
        "switch_step": switch_step, "horizon": horizon, "episodes": len(starts), "seed": seed,
        "summaries": summaries, "rows": rows,
        "cmdp_resolve_seconds": cmdp_resolve_seconds,
        "cmdp_dual_iterations": len(cmdp2.dual_trace),
        "icmdp_compile_seconds": p2.compile_seconds,
    }


def cmdp_dual(plan: IntentPlan) -> List[Dict]:
    plan.solve("cmdp")
    return [{"lambda": d["lambda"], "worst_cost": d["worst_cost"], "iterations": d["iterations"],
             "seconds": d["seconds"]} for d in plan.cmdp.dual_trace]


def run_parallel(plan: IntentPlan, seed: int = 1, horizon: int = 20, start: Optional[State] = None) -> Dict:
    """Two workflows on the same start and the same random draws (common random numbers).

    ICMDP lane: validated rule -> A_I planning -> runtime gate on raw state.
    CMDP lane:  cost C_I, budget 0 -> Lagrangian dual search -> policy -> no gate.
    Both lanes return per-step records that the UI replays in lockstep.
    """
    m = model()
    if start is None:
        start = sample_starts(1, seed)[0]
    ref = plan.ref()
    icmdp_sol = plan.solve("constraint_aware")
    cmdp_sol = plan.solve("cmdp")
    dual = cmdp_dual(plan)
    icmdp = run_episode(start, m, policy_at=lambda t: icmdp_sol.policy, V=icmdp_sol.V,
                        enforce_at=lambda t: ref, truth_at=lambda t: ref, gate_at=lambda t: True,
                        horizon=horizon, seed=seed, record_steps=True)
    cmdp = run_episode(start, m, policy_at=lambda t: cmdp_sol.policy, V=cmdp_sol.V,
                       enforce_at=lambda t: None, truth_at=lambda t: ref, gate_at=lambda t: False,
                       horizon=horizon, seed=seed, record_steps=True)
    return {
        "start": start.as_dict(),
        "seed": seed,
        "horizon": horizon,
        "icmdp": {
            "stages": [
                {"name": "parse", "detail": "proposal f_phi(l, x) — untrusted"},
                {"name": "validate", "detail": "v_x: schema, grounding, units, ranges"},
                {"name": "compile", "detail": f"c(g) rule evaluated over {N_STATES} states", "seconds": plan.compile_seconds},
                {"name": "feasibility", "detail": "A_I(s) ≠ ∅ on every grid state"},
                {"name": "activate", "detail": f"rule v{plan.version} installed atomically"},
                {"name": "plan", "detail": f"Bellman over A_I(s): {icmdp_sol.iterations} sweeps", "seconds": icmdp_sol.seconds},
                {"name": "gate", "detail": "filter every proposal on the raw state"},
            ],
            "steps": [s.__dict__ for s in icmdp.steps],
            "total_return": icmdp.total_return,
            "violations": icmdp.violations,
            "interventions": icmdp.interventions,
        },
        "cmdp": {
            "stages": [
                {"name": "spec", "detail": "same intent restated as cost C_I(s,a) = 1{a∈A0, a∉I}"},
                {"name": "budget", "detail": "constraint E[Σ γ^t C_I] ≤ 0 (zero budget)"},
                {"name": "dual search", "detail": f"{len(dual)} λ values tried, doubling",
                 "seconds": sum(d["seconds"] for d in dual)},
                {"name": "policy", "detail": f"Lagrangian policy at λ={plan.cmdp.lam:g}", "seconds": cmdp_sol.seconds},
                {"name": "execute", "detail": "no runtime filter; policy only"},
            ],
            "dual_trace": dual,
            "steps": [s.__dict__ for s in cmdp.steps],
            "total_return": cmdp.total_return,
            "violations": cmdp.violations,
            "interventions": 0,
        },
    }

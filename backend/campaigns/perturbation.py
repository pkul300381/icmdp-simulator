"""Campaign 4: observation noise and transition-model mismatch.

Rule under test: ScaleIn is forbidden when CPU is above 70 percent. This rule depends on a
continuous observation (cpu), so noise on cpu can move an observation across the threshold.
(The severity rule does not depend on cpu or latency, so noise on those fields cannot affect it.)

Arms:
  combined        constraint-aware planner + gate on the noisy observation (the paper's Algorithm 1)
  runtime_only    unconstrained planner + gate on the noisy observation
  constraint_only constraint-aware planner, no gate
Violations are scored against the true state.
"""
import json
import pathlib
import time

from icmdp import experiments as E

from . import common as C

OUT = pathlib.Path(__file__).parent / "results" / "campaign4_perturbation.json"
SIGMA_CPU = (0, 2, 5, 10)
SIGMA_LAT = (0, 100, 300)
EPS = (0.0, 0.1, 0.2, 0.3)


def run(episodes=150, horizon=20, seed=11):
    rule = C.compile_text(C.RULE_CPU)
    plan = E.build_plan("cpu70", rule)
    con = plan.solve("constraint_aware")
    unc = plan.solve("unconstrained")
    starts = C.sample_states(episodes, seed)
    arms = [C.Arm("combined", con.policy, con.V, rule),
            C.Arm("runtime_only", unc.policy, unc.V, rule),
            C.Arm("constraint_only", con.policy, con.V, None)]
    rows, t0 = [], time.time()
    for sc in SIGMA_CPU:
        for sl in SIGMA_LAT:
            for eps in EPS:
                for arm in arms:
                    st = C.simulate(arm, rule, starts, horizon, seed=seed,
                                    sig_cpu=sc, sig_lat=sl, eps=eps, salt=str(seed))
                    rows.append(dict(arm=arm.name, sigma_cpu=sc, sigma_latency=sl, eps=eps, **st))
    return dict(campaign="4_perturbation", rule=C.RULE_CPU, episodes=episodes, horizon=horizon,
                seed=seed, seconds=round(time.time() - t0, 1), rows=rows)


def summarise(res):
    print(f"{'arm':18s} {'eps':>4s}  {'viol/100 steps (min..max over noise cells)':>46s}  {'episodes with violation (max)':>30s}")
    for arm in ("combined", "runtime_only", "constraint_only"):
        for eps in EPS:
            cells = [r for r in res["rows"] if r["arm"] == arm and r["eps"] == eps]
            v = [c["violations_per_100_steps"] for c in cells]
            e = [c["episode_violation_rate"] for c in cells]
            print(f"{arm:18s} {eps:4.1f}  {min(v):>20.3f} .. {max(v):<20.3f}  {max(e):>30.3f}")


if __name__ == "__main__":
    res = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"done in {res['seconds']} s\n")
    summarise(res)

"""Campaign 5: where the snapping leak appears, and how it depends on grid resolution.

Part A (leakmap, deterministic): for each threshold t, the rule "ScaleIn is forbidden when
cpu > t" is evaluated on raw cpu values 0..100 (step 0.05). A raw value is a leak when its
admissible set differs from the admissible set of its nearest grid point (the planner's view).

Part B (episodes): the constraint-aware planner with no gate (leaks) against the same planner
with the gate on the raw state (should not leak), at the default threshold 70 percent, for each
CPU grid resolution. The resolution is set with ICMDP_CPU_GRID in a fresh process.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

from icmdp import experiments as E
from icmdp import intent as I
from icmdp.model import CPU_GRID, GRID, State, baseline, snap

from . import common as C

OUT = pathlib.Path(__file__).parent / "results" / "campaign5_grid_boundary.json"
GRIDS = {
    "5 points (paper)": "20,50,70,85,95",
    "11 points (step 10)": ",".join(str(x) for x in range(0, 101, 10)),
    "21 points (step 5)": ",".join(str(x) for x in range(0, 101, 5)),
    "41 points (step 2.5)": ",".join(f"{2.5 * x:g}" for x in range(41)),
}


def leakmap(step=1.0, x_step=0.05):
    xs = np.arange(0.0, 100.0 + 1e-9, x_step)
    rows = []
    for t in np.arange(0.0, 100.0 + 1e-9, step):
        rule = I.compile_spec(I.validate({"prohibitions": [
            {"action": "ScaleIn", "condition": f"cpu > {t:g} percent"}]}))
        leaks = 0
        for x in xs:
            s = State("none", "healthy", float(x), 300.0, 4, "idle")
            sn = snap(s)
            if rule.admissible(s, baseline(s)) != rule.admissible(sn, baseline(sn)):
                leaks += 1
        rows.append(dict(threshold=float(t), leak_fraction=round(leaks / len(xs), 4)))
    return rows


def episodes(episodes=300, horizon=20, seed=7):
    rule = C.compile_text(C.RULE_CPU)
    plan = E.build_plan("cpu70", rule)
    con = plan.solve("constraint_aware")
    starts = C.sample_states(episodes, seed)
    res = {"cpu_grid": [float(x) for x in CPU_GRID], "n_states": len(GRID)}
    for arm in (C.Arm("ungated_planner", con.policy, con.V, None),
                C.Arm("gated_on_raw_state", con.policy, con.V, rule)):
        res[arm.name] = C.simulate(arm, rule, starts, horizon, seed=seed)
    return res


def sweep():
    out = {}
    for label, spec in GRIDS.items():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        env = dict(os.environ, ICMDP_CPU_GRID=spec, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-m", "campaigns.grid_boundary", "--mode", "episodes", "--out", tmp],
                       check=True, env=env)
        out[label] = json.loads(pathlib.Path(tmp).read_text())
        os.unlink(tmp)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["leakmap", "episodes", "sweep"], default="sweep")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.mode == "episodes":
        pathlib.Path(args.out).write_text(json.dumps(episodes()), encoding="utf-8")
    elif args.mode == "leakmap":
        print(json.dumps(leakmap(), indent=1))
    else:
        result = dict(campaign="5_grid_boundary", leakmap=leakmap(), resolution_sweep=sweep())
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("resolution sweep (ungated planner violations per 100 steps vs gated):")
        for label, r in result["resolution_sweep"].items():
            u, g = r["ungated_planner"], r["gated_on_raw_state"]
            print(f"  {label:22s} states={r['n_states']:6d}  ungated={u['violations_per_100_steps']:.3f}"
                  f" (episodes {u['episode_violation_rate']:.3f})  gated={g['violations_per_100_steps']:.3f}")
        lm = result["leakmap"]
        worst = max(lm, key=lambda r: r["leak_fraction"])
        nz = [r["threshold"] for r in lm if r["leak_fraction"] > 0]
        print(f"\nleakmap (paper grid): max leak fraction {worst['leak_fraction']:.4f} at threshold {worst['threshold']:g};"
              f" thresholds with any leak: {len(nz)} of {len(lm)}")

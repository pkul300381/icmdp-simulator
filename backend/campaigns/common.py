"""Shared simulator for the campaigns.

Mirrors icmdp.runtime.run_episode with two additions that the paper's assumptions exclude:
  * the planner and the gate act on a noisy observation of the true state (sig_cpu, sig_lat);
  * executed transitions come from a perturbed model (relative perturbation eps per outcome),
    while the planner keeps the nominal model.
Violations are scored against the true state and the true intent.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from icmdp import intent as I
from icmdp.model import (ACTION_INDEX, ACTIONS, DEPLOY, GAMMA, GRID_INDEX, HEALTH, SEVERITY,
                         State, baseline, model, reward, snap, transition)

RULE_CPU = "Never scale in when CPU is above 70 percent"
RULE_SEV1 = "Do not deploy while a Severity-1 production incident is active"


def compile_text(text):
    out = I.local_parse(text)
    if "raw" not in out:
        raise ValueError(out.get("error"))
    vr = I.validate(out["raw"])
    if not vr.valid:
        raise ValueError(vr.reason)
    return I.compile_spec(vr)


@dataclass
class Arm:
    name: str
    policy: np.ndarray
    V: Optional[np.ndarray]
    gate_rule: Optional[object]     # CompiledRule enforced by the gate, None = no gate


def perturbed_transition(s, a, eps, salt):
    outs = transition(s, a)
    if eps <= 0 or len(outs) < 2:
        return outs
    r = random.Random(zlib.crc32(f"{s.key()}|{a}|{salt}".encode()))
    w = [max(1e-9, p * (1.0 + eps * (2.0 * r.random() - 1.0))) for _, p in outs]
    tot = sum(w)
    return [(ns, wi / tot) for (ns, _), wi in zip(outs, w)]


def observe(s, rng, sig_cpu, sig_lat):
    if sig_cpu <= 0 and sig_lat <= 0:
        return s
    cpu = s.cpu + (rng.normal(0.0, sig_cpu) if sig_cpu > 0 else 0.0)
    lat = s.latency + (rng.normal(0.0, sig_lat) if sig_lat > 0 else 0.0)
    return s._replace(cpu=min(100.0, max(0.0, cpu)), latency=min(5000.0, max(0.0, lat)))


def best_admissible(m, V, idx, names):
    if V is None:
        return names[0]
    best, bq = names[0], -np.inf
    for a in names:
        k = ACTION_INDEX[a]
        p = m.prob[idx, k]
        q = m.base_r[idx, k] + GAMMA * float(np.where(p > 0, p * V[m.nxt[idx, k]], 0.0).sum())
        if q > bq:
            best, bq = a, q
    return best


def simulate(arm, truth, starts, horizon, seed, sig_cpu=0.0, sig_lat=0.0, eps=0.0, salt="0"):
    m = model()
    rng = np.random.default_rng(seed)
    steps = viol = ops_err = holds = interv = 0
    ep_flags, rets = [], []
    for s0 in starts:
        s, ret, flag = s0, 0.0, 0
        for t in range(horizon):
            o = observe(s, rng, sig_cpu, sig_lat)
            idx = GRID_INDEX[snap(o)]
            pk = int(arm.policy[idx])
            if arm.gate_rule is not None:
                adm = arm.gate_rule.admissible(o, baseline(o))
                names = [a for a in ACTIONS if a in adm]
                if not names:
                    holds += 1
                    break
                if pk >= 0 and ACTIONS[pk] in adm:
                    executed = ACTIONS[pk]
                else:
                    executed = best_admissible(m, arm.V, idx, names)
                    interv += 1
            else:
                if pk < 0:
                    holds += 1
                    break
                executed = ACTIONS[pk]
            steps += 1
            if executed not in baseline(s):
                ops_err += 1
            elif executed not in truth.permitted(s):
                viol += 1
                flag = 1
            ret += (GAMMA ** t) * reward(s, executed)
            outs = perturbed_transition(s, executed, eps, salt)
            u, cum, ns = rng.random(), 0.0, outs[-1][0]
            for nxt, p in outs:
                cum += p
                if u < cum:
                    ns = nxt
                    break
            s = ns
        ep_flags.append(flag)
        rets.append(ret)
    return dict(steps=steps, violations=viol,
                violations_per_100_steps=round(100.0 * viol / max(steps, 1), 4),
                episodes=len(starts), episodes_with_violation=sum(ep_flags),
                episode_violation_rate=round(sum(ep_flags) / max(len(starts), 1), 4),
                operational_errors=ops_err, holds=holds, interventions=interv,
                mean_return=round(float(np.mean(rets)), 4))


def sample_states(n, seed):
    """Raw states drawn uniformly over the continuous ranges (not restricted to the grid)."""
    r = random.Random(seed)
    return [State(r.choice(SEVERITY), r.choice(HEALTH), round(r.uniform(0, 100), 2),
                  round(r.uniform(100, 2000), 1), r.randint(1, 10), r.choice(DEPLOY))
            for _ in range(n)]

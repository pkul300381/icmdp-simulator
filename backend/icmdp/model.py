"""
Cloud-operations MDP (Paper Example III.1) and its indexed planning model.

Dynamics match the revised legacy prototype (see INSTRUCTIONS.md):
  - Deploy enters in_progress; in-progress deployments complete w.p. 0.5 under any
    action other than Deploy/Rollback.
  - Restart clears an active incident w.p. 0.3.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

import numpy as np

SEVERITY = ("none", "sev2", "sev1")
HEALTH = ("healthy", "degraded", "critical")
DEPLOY = ("idle", "in_progress")
ACTIONS = ("Deploy", "Rollback", "Restart", "ScaleOut", "ScaleIn", "DoNothing")
CPU_GRID = (20, 50, 70, 85, 95)
LAT_GRID = (100, 300, 800, 2000)
REP_GRID = (1, 2, 3, 5, 8, 10)
GAMMA = 0.9
MAX_OUTCOMES = 6

SCHEMA = {
    "severity": {"type": "enum", "values": SEVERITY},
    "health": {"type": "enum", "values": HEALTH},
    "cpu": {"type": "number", "unit": "percent", "min": 0, "max": 100},
    "latency": {"type": "number", "unit": "ms", "min": 0, "max": 5000},
    "replicas": {"type": "integer", "min": 1, "max": 10},
    "deploy_status": {"type": "enum", "values": DEPLOY},
}


class State(NamedTuple):
    severity: str
    health: str
    cpu: float
    latency: float
    replicas: int
    deploy_status: str

    def key(self) -> str:
        return f"{self.severity}|{self.health}|{self.cpu:g}|{self.latency:g}|{self.replicas}|{self.deploy_status}"

    def as_dict(self) -> Dict:
        return {
            "severity": self.severity, "health": self.health, "cpu": self.cpu,
            "latency": self.latency, "replicas": self.replicas, "deploy_status": self.deploy_status,
        }

    @staticmethod
    def from_dict(d: Dict) -> "State":
        return State(
            severity=str(d["severity"]), health=str(d["health"]), cpu=float(d["cpu"]),
            latency=float(d["latency"]), replicas=int(d["replicas"]),
            deploy_status=str(d["deploy_status"]),
        )


def baseline(s: State) -> frozenset:
    """A0(s): operationally available actions (Paper Def. III.1)."""
    avail = set(ACTIONS)
    if s.deploy_status == "in_progress":
        avail.discard("Deploy")
    if s.deploy_status == "idle":
        avail.discard("Rollback")
    if s.replicas >= 10:
        avail.discard("ScaleOut")
    if s.replicas <= 1:
        avail.discard("ScaleIn")
    return frozenset(avail)


def reward(s: State, a: str) -> float:
    penalties = {"none": 0, "sev2": -10, "sev1": -30}
    health_pen = {"healthy": 0, "degraded": -5, "critical": -20}
    latency_pen = min(s.latency / 500.0, 10.0)
    cpu_pen = (s.cpu - 80) * 0.2 if s.cpu > 80 else 0.0
    base = penalties[s.severity] + health_pen[s.health] - latency_pen - cpu_pen
    action_r = {
        "Deploy": 8 if s.health == "healthy" else -5,
        "Rollback": 12 if s.severity != "none" else -2,
        "Restart": 6 if s.health != "healthy" else -1,
        "ScaleOut": 7 if s.cpu > 70 else -3,
        "ScaleIn": 3 if (s.replicas > 3 and s.cpu < 40) else -1,
        "DoNothing": 0,
    }
    return float(base + action_r.get(a, 0))


def _transition_base(s: State, a: str) -> List[Tuple[State, float]]:
    if a == "Deploy":
        return [
            (s._replace(health="healthy", deploy_status="in_progress", severity="none"), 0.7),
            (s._replace(health="degraded", deploy_status="in_progress", severity="sev2"), 0.2),
            (s._replace(health="critical", deploy_status="in_progress", severity="sev1"), 0.1),
        ]
    if a == "Rollback":
        return [
            (s._replace(health="healthy", severity="none", deploy_status="idle"), 0.85),
            (s._replace(health="degraded", severity="sev2", deploy_status="idle"), 0.15),
        ]
    if a == "Restart":
        if s.severity == "none":
            return [
                (s._replace(health="healthy", cpu=30, latency=150), 0.75),
                (s._replace(health="degraded", cpu=60, latency=400), 0.25),
            ]
        return [
            (s._replace(severity="none", health="healthy", cpu=30, latency=150), 0.3),
            (s._replace(health="healthy", cpu=30, latency=150), 0.45),
            (s._replace(health="degraded", cpu=60, latency=400), 0.25),
        ]
    if a == "ScaleOut":
        return [(s._replace(replicas=min(s.replicas + 1, 10), cpu=max(s.cpu - 15, 10)), 1.0)]
    if a == "ScaleIn":
        return [(s._replace(replicas=max(s.replicas - 1, 1), cpu=min(s.cpu + 20, 100)), 1.0)]
    # DoNothing
    worse_sev = "sev2" if s.severity == "none" else s.severity
    worse_health = "degraded" if s.health == "healthy" else s.health
    return [
        (s, 0.6),
        (s._replace(cpu=min(s.cpu + 10, 100), latency=min(s.latency + 100, 5000)), 0.3),
        (s._replace(severity=worse_sev, health=worse_health), 0.1),
    ]


def transition(s: State, a: str) -> List[Tuple[State, float]]:
    """P(s'|s,a) on the raw state. Probabilities sum to 1."""
    base = _transition_base(s, a)
    if s.deploy_status != "in_progress" or a in ("Deploy", "Rollback"):
        return base
    out: List[Tuple[State, float]] = []
    for ns, p in base:
        out.append((ns._replace(deploy_status="idle"), p * 0.5))
        out.append((ns._replace(deploy_status="in_progress"), p * 0.5))
    return out


def snap(s: State) -> State:
    """Nearest grid state. Planning approximation only; enforcement uses the raw state."""
    def near(v, grid):
        return min(grid, key=lambda g: (abs(g - v), g))
    return State(
        severity=s.severity, health=s.health,
        cpu=near(s.cpu, CPU_GRID), latency=near(s.latency, LAT_GRID),
        replicas=near(s.replicas, REP_GRID), deploy_status=s.deploy_status,
    )


def enumerate_grid() -> List[State]:
    out = []
    for sev in SEVERITY:
        for h in HEALTH:
            for c in CPU_GRID:
                for lat in LAT_GRID:
                    for r in REP_GRID:
                        for d in DEPLOY:
                            out.append(State(sev, h, float(c), float(lat), r, d))
    return out


GRID: List[State] = enumerate_grid()
GRID_INDEX: Dict[State, int] = {s: i for i, s in enumerate(GRID)}
N_STATES = len(GRID)
N_ACTIONS = len(ACTIONS)
ACTION_INDEX = {a: k for k, a in enumerate(ACTIONS)}


class IndexedModel:
    """Dense arrays over the grid for vectorised Bellman backups.

    avail[i,k]     action k is in A0 of grid state i
    base_r[i,k]    R(s_i, a_k)
    nxt[i,k,m]     index of m-th snapped successor (padded with index 0, prob 0)
    prob[i,k,m]    successor probabilities (0 for padding)
    """

    def __init__(self):
        N, A, M = N_STATES, N_ACTIONS, MAX_OUTCOMES
        self.avail = np.zeros((N, A), dtype=bool)
        self.base_r = np.zeros((N, A))
        self.nxt = np.zeros((N, A, M), dtype=np.int64)
        self.prob = np.zeros((N, A, M))
        for i, s in enumerate(GRID):
            av = baseline(s)
            for a in ACTIONS:
                k = ACTION_INDEX[a]
                if a not in av:
                    continue
                self.avail[i, k] = True
                self.base_r[i, k] = reward(s, a)
                for m, (ns, p) in enumerate(transition(s, a)):
                    self.nxt[i, k, m] = GRID_INDEX[snap(ns)]
                    self.prob[i, k, m] = p

    def permitted_mask(self, rule) -> np.ndarray:
        """Boolean (N, A): action k permitted by the compiled rule at grid state i."""
        mask = np.zeros((N_STATES, N_ACTIONS), dtype=bool)
        for i, s in enumerate(GRID):
            allowed = rule.permitted(s)
            for a in ACTIONS:
                mask[i, ACTION_INDEX[a]] = a in allowed
        return mask


_MODEL: IndexedModel | None = None


def model() -> IndexedModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = IndexedModel()
    return _MODEL

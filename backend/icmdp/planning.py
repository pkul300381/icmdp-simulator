"""
Planning over admissible sets (Paper Eq. 19), zero-budget CMDP via Lagrangian search, and
policy cost evaluation (Prop. III.2). All backups are vectorised over the indexed grid model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import GAMMA, IndexedModel


@dataclass
class Solution:
    V: np.ndarray            # (N,) value per grid state (-inf where A(s) empty)
    policy: np.ndarray       # (N,) action index, -1 where no admissible action
    iterations: int
    residual: float
    converged: bool
    seconds: float


def _q_table(m: IndexedModel, V: np.ndarray, reward_adj: Optional[np.ndarray], gamma: float) -> np.ndarray:
    base = m.base_r if reward_adj is None else reward_adj
    nv = V[m.nxt]                                         # (N, A, M)
    contrib = np.where(m.prob > 0, m.prob * nv, 0.0)      # avoid 0 * -inf = nan
    return base + gamma * contrib.sum(axis=2)


def value_iteration(m: IndexedModel, allowed: np.ndarray, reward_adj: Optional[np.ndarray] = None,
                    gamma: float = GAMMA, tol: float = 1e-8, max_iter: int = 3000) -> Solution:
    """Jacobi value iteration: V_{k+1}(s) = max_{a in A_I(s)} Q_k(s,a)."""
    t0 = time.perf_counter()
    valid = allowed & m.avail
    V = np.zeros(valid.shape[0])
    policy = np.full(valid.shape[0], -1, dtype=np.int64)
    empty = ~valid.any(axis=1)
    residual = np.inf
    it = 0
    while it < max_iter:
        it += 1
        Q = _q_table(m, V, reward_adj, gamma)
        Qm = np.where(valid, Q, -np.inf)
        best = Qm.max(axis=1)
        arg = Qm.argmax(axis=1)
        best[empty] = -np.inf
        arg = np.where(empty, -1, arg)
        finite = np.isfinite(best) & np.isfinite(V)
        diff = np.where(finite, np.abs(best - V), np.where(best == V, 0.0, np.inf))
        residual = float(diff.max())
        V, policy = best, arg
        if residual <= tol:
            break
    return Solution(V=V, policy=policy, iterations=it, residual=residual,
                    converged=residual <= tol, seconds=time.perf_counter() - t0)


def policy_cost_value(m: IndexedModel, policy: np.ndarray, cost: np.ndarray,
                      gamma: float = GAMMA, tol: float = 1e-11, max_iter: int = 20000) -> np.ndarray:
    """Discounted cost value of a fixed policy: Vc = c(s,pi(s)) + gamma * P_pi Vc."""
    N = policy.shape[0]
    rows = np.arange(N)
    act = np.where(policy >= 0, policy, 0)
    c = np.where(policy >= 0, cost[rows, act], 0.0)
    nxt = m.nxt[rows, act]        # (N, M)
    prob = np.where(policy[:, None] >= 0, m.prob[rows, act], 0.0)
    Vc = np.zeros(N)
    for _ in range(max_iter):
        new = c + gamma * np.where(prob > 0, prob * Vc[nxt], 0.0).sum(axis=1)
        if np.max(np.abs(new - Vc)) < tol:
            Vc = new
            break
        Vc = new
    return Vc


@dataclass
class CMDPResult:
    solution: Solution
    lam: float
    worst_cost: float
    satisfied: bool
    dual_trace: list   # [(lambda, worst_cost, iterations, seconds)] — the CMDP workflow's search log


def zero_budget_cmdp(m: IndexedModel, allowed: np.ndarray, violation: np.ndarray,
                     gamma: float = GAMMA, lam_max: float = 2 ** 24) -> CMDPResult:
    """Zero-budget CMDP via Lagrangian doubling search on lambda (approximates E[sum g^t C] <= 0).

    violation[i,k] = 1 if action k is available but excluded by the intent at grid state i.
    """
    dual = []
    lam = 1.0
    last = None
    while lam <= lam_max:
        penalty = m.base_r - lam * violation
        sol = value_iteration(m, allowed, reward_adj=penalty, gamma=gamma)
        Vc = policy_cost_value(m, sol.policy, violation.astype(float), gamma=gamma)
        worst = float(Vc.max())
        dual.append({"lambda": lam, "worst_cost": worst, "iterations": sol.iterations, "seconds": sol.seconds})
        last = CMDPResult(sol, lam, worst, worst <= 1e-9, dual)
        if last.satisfied:
            break
        lam *= 2
    return last


def finite_penalty(m: IndexedModel, allowed: np.ndarray, violation: np.ndarray, lam: float,
                   gamma: float = GAMMA) -> Solution:
    return value_iteration(m, allowed, reward_adj=m.base_r - lam * violation, gamma=gamma)

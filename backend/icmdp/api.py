"""
FastAPI service: intent lifecycle, Markov queries, experiment runs, parallel ICMDP/CMDP workflow,
event log, and Prometheus metrics. Run with:  uvicorn icmdp.api:app --reload --port 8000
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import experiments as E
from . import intent as I
from . import toy
from .model import ACTIONS, CPU_GRID, DEPLOY, GAMMA, GRID_INDEX, HEALTH, LAT_GRID, REP_GRID, SCHEMA, SEVERITY, State, baseline, model, snap
from .observability import METRICS, get_logger
from .planning import value_iteration
from .store import Store

log = get_logger("icmdp.api")
DB_PATH = os.environ.get("ICMDP_DB", os.path.join(os.path.dirname(__file__), "..", "data", "icmdp.db"))
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

store = Store(DB_PATH)
PLANS: Dict[int, E.IntentPlan] = {}
BASELINE_PLAN: Optional[E.IntentPlan] = None

app = FastAPI(title="ICMDP Agent Harness", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def baseline_plan() -> E.IntentPlan:
    global BASELINE_PLAN
    if BASELINE_PLAN is None:
        rule = I.CompiledRule(I.ValidatedSpec([]))     # permits everything: plain MDP
        BASELINE_PLAN = E.build_plan("baseline", rule, 0)
    return BASELINE_PLAN


def plan_for_intent(intent_id: int) -> E.IntentPlan:
    if intent_id in PLANS:
        return PLANS[intent_id]
    row = store.intent(intent_id)
    if row is None:
        raise HTTPException(404, f"intent {intent_id} not found")
    vr = I.validate(row["raw"])
    if not vr.valid:
        raise HTTPException(409, f"intent {intent_id} no longer validates: {vr.reason}")
    plan = E.build_plan(row.get("name") or f"intent{intent_id}", I.CompiledRule(vr.spec), row.get("version") or 1)
    PLANS[intent_id] = plan
    return plan


def seed_presets() -> Dict[str, int]:
    ids = {}
    existing = {r["name"]: r["id"] for r in store.intents() if r.get("name")}
    for name, text in I.PRESET_INSTRUCTIONS.items():
        if name in existing:
            ids[name] = existing[name]
            continue
        raw = I.local_parse(text).get("raw")
        vr = I.validate(raw) if raw else I.ValidationResult(False, None, "no parse")
        ids[name] = store.create_intent(name, text, "preset", raw, vr.to_dict()["spec"],
                                        "validated" if vr.valid else "rejected", vr.reason)
    return ids


PRESETS = seed_presets()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ParseBody(BaseModel):
    instruction: str
    parser: str = "local"            # local | llm
    name: Optional[str] = None


class RunBody(BaseModel):
    kind: str = "six_arm"            # six_arm | penalty_sweep | freeze
    intent_id: Optional[int] = None
    before_intent_id: Optional[int] = None   # freeze only
    after_intent_id: Optional[int] = None    # freeze only
    episodes: int = 500
    seed: int = 1
    horizon: int = 20
    trace_sample: int = 3


class ParallelBody(BaseModel):
    intent_id: Optional[int] = None
    seed: int = 1
    horizon: int = 20
    start: Optional[Dict[str, Any]] = None


class ActivateBody(BaseModel):
    override: bool = False


# ---------------------------------------------------------------------------
# Meta and health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "db": DB_PATH}


@app.get("/api/meta")
def meta():
    return {
        "schema": SCHEMA,
        "actions": list(ACTIONS),
        "severity": list(SEVERITY), "health": list(HEALTH), "deploy_status": list(DEPLOY),
        "cpu_grid": list(CPU_GRID), "latency_grid": list(LAT_GRID), "replicas_grid": list(REP_GRID),
        "gamma": GAMMA,
        "grid_states": len(GRID_INDEX),
        "presets": PRESETS,
        "preset_instructions": I.PRESET_INSTRUCTIONS,
        "active_intent": store.active_intent(),
    }


# ---------------------------------------------------------------------------
# Intent lifecycle (slow timescale)
# ---------------------------------------------------------------------------
@app.get("/api/intents")
def list_intents():
    return store.intents()


@app.get("/api/intents/active")
def get_active():
    return store.active_intent()


@app.get("/api/intents/{intent_id}")
def get_intent(intent_id: int):
    row = store.intent(intent_id)
    if row is None:
        raise HTTPException(404, "not found")
    return row


@app.post("/api/intents/parse")
def parse_intent(body: ParseBody):
    """Parse -> validate -> compile -> feasibility. Each stage is reported for the UI pipeline."""
    stages: List[Dict[str, Any]] = []
    if body.parser == "llm":
        out = I.llm_parse(body.instruction)
    else:
        out = I.local_parse(body.instruction)
    if "error" in out:
        stages.append({"name": "parse", "status": "error", "detail": out["error"]})
        iid = store.create_intent(body.name, body.instruction, body.parser, None, None, "hold", out["error"])
        METRICS.inc("icmdp_intent_events_total", outcome="hold")
        store.event("intent.hold", str(iid), {"stage": "parse", "reason": out["error"]})
        return {"intent_id": iid, "status": "hold", "stages": stages}
    raw = out["raw"]
    stages.append({"name": "parse", "status": "ok", "detail": f"proposal from {body.parser} parser (untrusted)", "raw": raw})

    vr = I.validate(raw)
    if not vr.valid:
        stages.append({"name": "validate", "status": "error", "detail": vr.reason, "field": vr.field})
        iid = store.create_intent(body.name, body.instruction, body.parser, raw, None, "rejected", vr.reason)
        METRICS.inc("icmdp_intent_events_total", outcome="rejected")
        store.event("intent.rejected", str(iid), {"field": vr.field, "reason": vr.reason})
        return {"intent_id": iid, "status": "rejected", "stages": stages}
    stages.append({"name": "validate", "status": "ok", "detail": "schema, grounding, units and ranges checked",
                   "spec": vr.spec.to_dict()})

    rule = I.compile_spec(vr)
    plan = E.build_plan(body.name or "draft", rule, 0)
    stages.append({"name": "compile", "status": "ok", "detail": f"c(g) evaluated on {len(GRID_INDEX)} states",
                   "seconds": plan.compile_seconds})

    avail_perm = plan.avail & plan.permitted
    empty = int((~avail_perm.any(axis=1)).sum())
    total = len(GRID_INDEX)
    if empty:
        stages.append({"name": "feasibility", "status": "hold",
                       "detail": f"A_I(s) = ∅ on {empty} of {total} states (Def. III.2): HOLD, no silent relaxation",
                       "empty_states": empty})
        status = "hold"
    else:
        stages.append({"name": "feasibility", "status": "ok", "detail": f"A_I(s) ≠ ∅ on all {total} states",
                       "empty_states": 0})
        status = "validated"
    stages.append({"name": "activate", "status": "pending",
                   "detail": "awaiting atomic activation" if status == "validated" else "blocked by HOLD"})

    iid = store.create_intent(body.name, body.instruction, body.parser, raw, vr.spec.to_dict(), status,
                              "" if status == "validated" else "infeasible", empty_states=empty)
    PLANS[iid] = plan
    METRICS.inc("icmdp_intent_events_total", outcome=status)
    store.event(f"intent.{status}", str(iid), {"empty_states": empty, "instruction": body.instruction})
    log.info("intent parsed", extra={"fields": {"intent_id": iid, "status": status, "empty_states": empty}})
    return {"intent_id": iid, "status": status, "stages": stages, "spec": vr.spec.to_dict(), "empty_states": empty}


@app.post("/api/intents/{intent_id}/activate")
def activate(intent_id: int, body: ActivateBody = ActivateBody()):
    row = store.intent(intent_id)
    if row is None:
        raise HTTPException(404, "not found")
    if row["status"] == "active":
        # Idempotent: a repeated activation (e.g. a double click) returns the live version unchanged.
        return {"intent_id": intent_id, "version": row["version"], "status": "active",
                "override": False, "already_active": True}
    if row["status"] == "validated" or (row["status"] == "hold" and body.override and row["spec"]):
        version = store.activate_intent(intent_id)
        if row["status"] == "hold":
            store.event("intent.override", str(intent_id), {"note": "explicit override of HOLD; logged"})
        PLANS.pop(intent_id, None)
        plan_for_intent(intent_id)
        METRICS.inc("icmdp_intent_events_total", outcome="activated")
        METRICS.set("icmdp_active_intent_version", version)
        store.event("intent.activated", str(intent_id), {"version": version, "override": body.override})
        return {"intent_id": intent_id, "version": version, "status": "active", "override": body.override}
    raise HTTPException(409, f"intent is '{row['status']}'; cannot activate" +
                        ("" if row["status"] != "hold" else " (set override=true to force, logged)"))


# ---------------------------------------------------------------------------
# Markov views
# ---------------------------------------------------------------------------
def _state_from_query(severity, health, cpu, latency, replicas, deploy_status) -> State:
    return State(severity, health, float(cpu), float(latency), int(replicas), deploy_status)


@app.get("/api/markov/admissibility")
def admissibility(intent_id: Optional[int] = None, severity: str = "none", health: str = "healthy",
                  deploy_status: str = "idle", replicas: int = 3):
    """Heatmap data: for fixed (severity, health, deploy, replicas), the admissible set over cpu x latency."""
    plan = plan_for_intent(intent_id) if intent_id else baseline_plan()
    cells = []
    for c in CPU_GRID:
        for lat in LAT_GRID:
            s = State(severity, health, float(c), float(lat), int(replicas), deploy_status)
            av = baseline(s)
            statuses = {}
            for a in ACTIONS:
                if a not in av:
                    statuses[a] = "unavailable"
                elif a in plan.rule.permitted(s):
                    statuses[a] = "admissible"
                else:
                    statuses[a] = "excluded"
            cells.append({"cpu": c, "latency": lat, "statuses": statuses})
    return {"intent_id": intent_id, "fixed": {"severity": severity, "health": health,
                                               "deploy_status": deploy_status, "replicas": replicas},
            "actions": list(ACTIONS), "cells": cells}


@app.get("/api/markov/decision")
def decision(intent_id: Optional[int] = None, severity: str = "sev1", health: str = "degraded",
             cpu: float = 85, latency: float = 300, replicas: int = 3, deploy_status: str = "idle"):
    """One decision step as a graph: state -> actions (status, Q) -> successor states (probabilities)."""
    plan = plan_for_intent(intent_id) if intent_id else baseline_plan()
    m = model()
    s = State(severity, health, float(cpu), float(latency), int(replicas), deploy_status)
    sol = plan.solve("constraint_aware" if intent_id else "unconstrained")
    k_snap = GRID_INDEX[snap(s)]
    av = baseline(s)
    actions = []
    from .model import reward, transition  # local to keep module top clean
    for a in ACTIONS:
        k = ACTIONS.index(a)
        if a not in av:
            status = "unavailable"
        elif a in plan.rule.permitted(s):
            status = "admissible"
        else:
            status = "excluded"
        q = None
        if status == "admissible":
            nxt = m.nxt[k_snap, k]
            p = m.prob[k_snap, k]
            q = float(m.base_r[k_snap, k] + GAMMA * np.where(p > 0, p * sol.V[nxt], 0.0).sum())
        outcomes = []
        if status != "unavailable":
            for ns, p in transition(s, a):
                ks = GRID_INDEX[snap(ns)]
                outcomes.append({"state": ns.as_dict(), "prob": p, "value": float(sol.V[ks]) if np.isfinite(sol.V[ks]) else None})
        actions.append({"action": a, "status": status, "q": q, "reward": reward(s, a), "outcomes": outcomes,
                        "is_policy_choice": bool(sol.policy[k_snap] == k)})
    return {"intent_id": intent_id, "state": s.as_dict(), "actions": actions,
            "policy_choice": ACTIONS[sol.policy[k_snap]] if sol.policy[k_snap] >= 0 else None}


@app.get("/api/markov/example51")
def example51():
    return toy.example51()


# ---------------------------------------------------------------------------
# Experiment runs (medium + fast timescales, persisted)
# ---------------------------------------------------------------------------
def _clean_summary(out: Dict) -> Dict:
    keep = {k: v for k, v in out.items() if k not in ("episode_rows", "traces")}
    return keep


@app.post("/api/runs")
def create_run(body: RunBody):
    run_id = uuid.uuid4().hex[:12]
    METRICS.inc("icmdp_runs_total", kind=body.kind)
    if body.kind == "six_arm":
        iid = body.intent_id or PRESETS["sev1"]
        plan = plan_for_intent(iid)
        store.create_run(run_id, body.kind, iid, body.model_dump())
        out = E.run_six_arm(plan, episodes=body.episodes, seed=body.seed, horizon=body.horizon,
                            trace_sample=body.trace_sample)
        store.save_episodes(run_id, out["episode_rows"])
        store.save_traces(run_id, out["traces"])
        for s in out["summaries"]:
            METRICS.inc("icmdp_violations_total", value=s["violations_per_100_steps"] * out["episodes"] / 100.0,
                        arm=s["arm"])
            METRICS.inc("icmdp_gate_interventions_total", value=s["interventions_per_episode"] * out["episodes"],
                        arm=s["arm"])
            if s["planning_seconds"] is not None:
                METRICS.observe("icmdp_planning_seconds", s["planning_seconds"], planner=s["arm"])
        summary = _clean_summary(out)
    elif body.kind == "penalty_sweep":
        iid = body.intent_id or PRESETS["sev1"]
        plan = plan_for_intent(iid)
        store.create_run(run_id, body.kind, iid, body.model_dump())
        summary = {"sweep": E.run_penalty_sweep(plan, episodes=body.episodes, seed=body.seed, horizon=body.horizon)}
    elif body.kind == "freeze":
        before = plan_for_intent(body.before_intent_id or PRESETS["sev1"]).rule
        after = plan_for_intent(body.after_intent_id or PRESETS["freeze"]).rule
        store.create_run(run_id, body.kind, body.after_intent_id or PRESETS["freeze"], body.model_dump())
        out = E.run_freeze_update(before, after, episodes=body.episodes, seed=body.seed, horizon=max(body.horizon, 10))
        store.save_episodes(run_id, [{"arm": r["arm"], "index": r["index"], "start": None, "return": r["return"],
                                      "steps": 0, "violations": r["violations"], "interventions": r["interventions"],
                                      "holds": 0, "operational_errors": 0} for r in out["rows"]])
        summary = {k: v for k, v in out.items() if k != "rows"}
    else:
        raise HTTPException(400, f"unknown kind {body.kind}")
    store.finish_run(run_id, summary)
    store.event("run.done", run_id, {"kind": body.kind})
    log.info("run done", extra={"fields": {"run_id": run_id, "kind": body.kind}})
    return {"run_id": run_id, "kind": body.kind, "summary": summary}


@app.get("/api/runs")
def list_runs():
    return store.runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    r = store.run(run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    return r


@app.get("/api/runs/{run_id}/episodes/{arm}")
def run_episodes(run_id: str, arm: str, limit: int = 200, offset: int = 0):
    return store.episodes(run_id, arm, limit, offset)


@app.get("/api/runs/{run_id}/trace/{arm}/{idx}")
def run_trace(run_id: str, arm: str, idx: int):
    steps = store.trace(run_id, arm, idx)
    if not steps:
        raise HTTPException(404, "no trace stored for this episode (see trace_sample)")
    return {"run_id": run_id, "arm": arm, "index": idx, "steps": steps}


# ---------------------------------------------------------------------------
# Parallel workflows: ICMDP lane vs CMDP lane on the same start and draws
# ---------------------------------------------------------------------------
@app.post("/api/parallel")
def parallel(body: ParallelBody):
    iid = body.intent_id or PRESETS["sev1"]
    plan = plan_for_intent(iid)
    start = State.from_dict(body.start) if body.start else None
    out = E.run_parallel(plan, seed=body.seed, horizon=body.horizon, start=start)
    METRICS.inc("icmdp_runs_total", kind="parallel")
    return {"intent_id": iid, **out}


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
@app.get("/api/events")
def events(limit: int = Query(100, ge=1, le=1000)):
    return store.events(limit)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return METRICS.render()

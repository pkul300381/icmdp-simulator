from fastapi.testclient import TestClient
from icmdp.api import app, PRESETS

client = TestClient(app)


def test_meta_has_presets():
    m = client.get("/api/meta").json()
    assert set(PRESETS) >= {"sev1", "freeze", "cpuScaleIn", "overTight"}
    assert m["grid_states"] == 2160


def test_parse_validate_activate_lifecycle():
    r = client.post("/api/intents/parse", json={"instruction": "Do not deploy while a Severity-1 production incident is active", "parser": "local"}).json()
    assert r["status"] == "validated"
    assert [s["status"] for s in r["stages"]][:4] == ["ok", "ok", "ok", "ok"]
    a = client.post(f"/api/intents/{r['intent_id']}/activate")
    assert a.status_code == 200 and a.json()["version"] >= 1


def test_activate_is_idempotent_and_retired_is_refused():
    first = client.post("/api/intents/parse", json={"instruction": "Do not deploy while a Severity-1 production incident is active", "parser": "local"}).json()
    a1 = client.post(f"/api/intents/{first['intent_id']}/activate").json()
    a2 = client.post(f"/api/intents/{first['intent_id']}/activate")
    assert a2.status_code == 200 and a2.json()["version"] == a1["version"] and a2.json()["already_active"]
    second = client.post("/api/intents/parse", json={"instruction": "Freeze all production deployments until further notice", "parser": "local"}).json()
    assert client.post(f"/api/intents/{second['intent_id']}/activate").status_code == 200
    assert client.post(f"/api/intents/{first['intent_id']}/activate").status_code == 409


def test_rejected_and_hold_paths():
    bad = client.post("/api/intents/parse", json={"instruction": "Never scale in when CPU is above 150 percent", "parser": "local"}).json()
    assert bad["status"] == "rejected"
    hold = client.post("/api/intents/parse", json={"instruction": "Prohibit every action whenever CPU is non-negative", "parser": "local"}).json()
    assert hold["status"] == "hold" and hold["empty_states"] == 2160
    assert client.post(f"/api/intents/{hold['intent_id']}/activate").status_code == 409
    forced = client.post(f"/api/intents/{hold['intent_id']}/activate", json={"override": True})
    assert forced.status_code == 200


def test_underspecified_instruction_holds():
    r = client.post("/api/intents/parse", json={"instruction": "Make it faster and safer", "parser": "local"}).json()
    assert r["status"] == "hold"


def test_run_and_trace_persisted():
    run = client.post("/api/runs", json={"kind": "six_arm", "intent_id": PRESETS["sev1"], "episodes": 60, "seed": 3, "horizon": 10, "trace_sample": 2}).json()
    rid = run["run_id"]
    assert client.get(f"/api/runs/{rid}").json()["status"] == "done"
    assert len(client.get(f"/api/runs/{rid}/episodes/6_combined").json()) == 60
    assert len(client.get(f"/api/runs/{rid}/trace/6_combined/0").json()["steps"]) > 0


def test_markov_views_and_example():
    dec = client.get("/api/markov/decision", params={"severity": "sev1", "health": "degraded", "cpu": 85, "intent_id": PRESETS["sev1"]}).json()
    statuses = {a["action"]: a["status"] for a in dec["actions"]}
    assert statuses["Deploy"] == "excluded" and statuses["Rollback"] == "unavailable"
    adm = client.get("/api/markov/admissibility", params={"intent_id": PRESETS["sev1"], "severity": "sev1"}).json()
    assert len(adm["cells"]) == 20
    assert client.get("/api/markov/example51").json()["unconstrained"]["selected_at_s0"] == "Risky"


def test_parallel_lanes_and_metrics():
    p = client.post("/api/parallel", json={"seed": 3, "horizon": 6}).json()
    assert p["icmdp"]["violations"] == 0 and p["cmdp"]["violations"] == 0
    assert len(p["cmdp"]["dual_trace"]) >= 1
    text = client.get("/metrics").text
    assert "icmdp_runs_total" in text

import numpy as np
from icmdp import intent as I, experiments as E, toy
from icmdp.model import ACTIONS, GRID, GRID_INDEX, State, baseline, model, reward, transition, snap
from icmdp.runtime import RuleRef, run_episode
from icmdp.planning import value_iteration, zero_budget_cmdp


def rule_for(text):
    vr = I.validate(I.local_parse(text)["raw"])
    assert vr.valid, vr.reason
    return I.compile_spec(vr)


def test_validator_range_units_ambiguity():
    assert not I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "cpu > 150"}]}).valid
    assert not I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "latency > 1.2"}]}).valid
    assert not I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "latency > 2"}]}).valid
    assert not I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "cpu > 70"}]}).valid
    assert I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "cpu > 70 percent"}]}).valid
    r = I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "latency > 1.2 s"}]})
    assert r.valid and r.spec.prohibitions[0].cond.value == 1200
    r = I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "cpu > 0.7 fraction"}]})
    assert r.valid and r.spec.prohibitions[0].cond.value == 70
    assert not I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "latency >"}]}).valid
    assert not I.validate({"prohibitions": [{"action": "ScaleIn", "condition": "severity = SEV1"}]}).valid
    assert not I.validate({"prohibitions": [{"action": "Scale", "condition": "cpu > 5"}]}).valid


def test_local_parser_operator_direction():
    raw = I.local_parse("Never scale in when CPU is below 40 percent")["raw"]
    assert raw["prohibitions"][0]["condition"] == "cpu < 40 percent"


def test_gate_uses_raw_state():
    rule = rule_for("Never scale in when CPU is above 70 percent")
    raw = State("none", "healthy", 75.0, 300.0, 4, "idle")
    assert "ScaleIn" not in rule.admissible(raw, baseline(raw))
    assert "ScaleIn" in rule.admissible(snap(raw), baseline(snap(raw)))   # the leak the gate avoids


def test_example_v1_reproduces_paper():
    r = toy.example51()
    assert abs(r["unconstrained"]["Q"]["s0:Risky"] - 9) < 1e-9
    assert r["unconstrained"]["selected_at_s0"] == "Risky"
    assert abs(r["unconstrained"]["return_with_gate"] - 0.9) < 1e-9
    assert abs(r["constraint_aware"]["V_I_s1"] - 1) < 1e-9
    assert abs(r["constraint_aware"]["return"] - 5.0) < 1e-9
    assert r["constraint_aware"]["selected_at_s0"] == "Safe"


def test_vectorised_vi_matches_scalar_backup():
    plan = E.build_plan("sev1", rule_for("Do not deploy while a Severity-1 production incident is active"))
    sol = plan.solve("constraint_aware")
    m = model()
    i = GRID_INDEX[State("sev1", "degraded", 85.0, 300.0, 3, "idle")]
    best = max(
        m.base_r[i, k] + 0.9 * sum(p * sol.V[n] for n, p in zip(m.nxt[i, k], m.prob[i, k]))
        for k in range(len(ACTIONS)) if (m.avail[i, k] and plan.permitted[i, k]))
    assert abs(best - sol.V[i]) < 1e-6


def test_sev1_not_absorbing_under_rule():
    plan = E.build_plan("sev1", rule_for("Do not deploy while a Severity-1 production incident is active"))
    sol = plan.solve("constraint_aware")
    assert sol.V[GRID_INDEX[State("sev1", "degraded", 85.0, 300.0, 3, "idle")]] > -150


def test_zero_budget_cmdp_satisfies_and_matches_icmdp_policy_value():
    plan = E.build_plan("sev1", rule_for("Do not deploy while a Severity-1 production incident is active"))
    res = zero_budget_cmdp(plan.m, plan.avail, plan.violation)
    assert res.satisfied and res.worst_cost <= 1e-9
    assert np.all(~((plan.violation > 0) & (res.solution.policy[:, None] == np.arange(len(ACTIONS))[None, :])).any(axis=1))


def test_six_arm_constrained_arms_have_zero_violations():
    plan = E.build_plan("sev1", rule_for("Do not deploy while a Severity-1 production incident is active"))
    out = E.run_six_arm(plan, episodes=150, seed=7, horizon=15, trace_sample=1)
    by = {s["arm"]: s for s in out["summaries"]}
    assert by["1_unconstrained"]["violations_per_100_steps"] > 0
    for arm in ("4_runtime_only", "6_combined"):
        assert by[arm]["violations_per_100_steps"] == 0
        assert by[arm]["operational_errors"] == 0
    assert by["4_runtime_only"]["interventions_per_episode"] > 0
    assert out["zero_budget"]["satisfied"]


def test_cpu_rule_snapping_leak_shows_without_gate():
    plan = E.build_plan("cpu", rule_for("Never scale in when CPU is above 70 percent"))
    out = E.run_six_arm(plan, episodes=200, seed=7, horizon=15, trace_sample=0)
    by = {s["arm"]: s for s in out["summaries"]}
    assert by["5_constraint_aware_only"]["violations_per_100_steps"] > 0
    assert by["6_combined"]["violations_per_100_steps"] == 0


def test_episode_deterministic_under_seed():
    plan = E.build_plan("sev1", rule_for("Do not deploy while a Severity-1 production incident is active"))
    sol = plan.solve("unconstrained")
    st = GRID[10]
    args = dict(policy_at=lambda t: sol.policy, V=sol.V, enforce_at=lambda t: None,
                truth_at=lambda t: plan.ref(), gate_at=lambda t: False, horizon=12, seed=5)
    a = run_episode(st, plan.m, **args)
    b = run_episode(st, plan.m, **args)
    assert a.total_return == b.total_return and a.severity_path == b.severity_path


def test_freeze_update_stale_filter_fails_fix_holds():
    before = rule_for("Do not deploy while a Severity-1 production incident is active")
    after = rule_for("Freeze all production deployments until further notice")
    out = E.run_freeze_update(before, after, episodes=120, seed=7, switch_step=5, horizon=20)
    by = {s["arm"]: s for s in out["summaries"]}
    assert by["F1_no_update"]["violations_per_100_steps"] > 0
    assert by["F3_stale_policy_no_filter_after_switch"]["violations_per_100_steps"] > 0
    for arm in ("F2_stale_policy_with_new_filter", "F4_reoptimised_with_filter", "F5_cmdp_resolve_no_filter"):
        assert by[arm]["violations_per_100_steps"] == 0, arm


def test_health_rule_parses_and_excludes_deploy_only_when_critical():
    raw = I.local_parse("Do not deploy while health is critical")["raw"]
    rule = I.compile_spec(I.validate(raw))
    crit = State("none", "critical", 50.0, 300.0, 3, "idle")
    healthy = State("none", "healthy", 50.0, 300.0, 3, "idle")
    assert "Deploy" not in rule.permitted(crit)
    assert "Deploy" in rule.permitted(healthy)
    assert I.PRESET_INSTRUCTIONS["healthCritical"] == "Do not deploy while health is critical"

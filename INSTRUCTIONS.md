# ICMDP Agent Harness — Instructions

Harness for testing the Intent-Constrained MDP framework (Kulkarni, Saini & Das, 2026). It compiles a
natural-language intent into a validated, versioned rule; plans over the admissible sets; enforces the rule
at a runtime gate; and compares the result with a zero-budget CMDP run as a parallel workflow.

## Layout

```
backend/            Python 3.9+ · FastAPI · numpy · SQLite (stdlib)
  icmdp/model.py       cloud-operations MDP (Paper Ex. III.1), indexed planning model
  icmdp/intent.py      grammar G, validator v_x (units, ranges, ambiguity), compiler c, parsers f_phi
  icmdp/planning.py    Jacobi value iteration over A_I(s) (Eq. 19), zero-budget CMDP (Lagrangian)
  icmdp/runtime.py     runtime gate on the RAW state, value-based replacement, episode runner
  icmdp/experiments.py six-arm comparison, penalty sweep, freeze update, parallel ICMDP/CMDP lanes
  icmdp/toy.py         Example V.1 as an explicit 3-state MDP
  icmdp/store.py       SQLite: intents (versions), runs, episodes, trace steps, event log
  icmdp/observability.py  JSON logs, Prometheus metrics
  icmdp/api.py         HTTP API (see below)
  tests/               pytest suite (18 tests): validator, gate, planning, experiments, API
frontend/           Vite · React · recharts (five pages; see below)
legacy/             the retired single-file JS prototype, kept for reference only
prometheus.yml      scrape config for /metrics
```

## Run it

```bash
# 1. backend
cd backend
python3 -m pip install -r requirements.txt
python3 -m uvicorn icmdp.api:app --port 8000        # SQLite file: backend/data/icmdp.db
python3 -m pytest -q tests                           # 18 tests

# 2. frontend (separate terminal)
cd frontend
npm install
npm run dev                                          # http://localhost:5173 (proxies /api to :8000)
```

Optional: `ANTHROPIC_API_KEY` enables the LLM parser (`parser: "llm"`, model `ICMDP_LLM_MODEL`,
default `claude-sonnet-5-5`). The LLM output is untrusted and must pass the validator; the local grammar parser
works offline. The LLM adapter has not been exercised against the live API in this environment.

## The lifecycle (slow → medium → fast timescales)

1. **Parse** the instruction (local grammar or LLM). The output is a proposal.
2. **Validate** against the schema: action and field names, operators, enum values, numeric ranges,
   declared units (`latency > 1.2 s`), and ambiguity: a unit-bearing field (`cpu`, `latency`) without an explicit unit is rejected, integer or not. Rejections are logged.
3. **Compile** the spec into a rule `I(s)` and evaluate it on all 2,160 grid states.
4. **Feasibility**: if `A_I(s) = ∅` anywhere, the intent goes to HOLD. Activation requires an explicit,
   logged override.
5. **Activate** atomically as a new version. The previous active version is retired.
6. **Plan** over `A_I(s) = A₀(s) ∩ I(s)` (Eq. 19).
7. **Gate** every proposal on the raw observed state. A replacement is chosen by argmax of Q over A_I(s).
8. **Audit**: every decision is stored with state, rule version, admissible set, proposal, executed action and event.

## Environment dynamics

Revised from the legacy prototype (the original made Sev1 absorbing under the Sev1 rule):
- `Deploy` enters `in_progress`; a deployment completes w.p. 0.5 under any action except Deploy/Rollback.
- `Restart` clears an active incident w.p. 0.3.

These are modelling choices, not from the paper. State them in the thesis.

## Evaluation protocols (Paper §V.C and §IV-D)

**Six arms** (`POST /api/runs {kind: "six_arm"}`):

| arm | planner | gate |
|---|---|---|
| 1 unconstrained | A₀ | no |
| 2 finite penalty | A₀ with reward − λ·C_I, λ = 100 | no |
| 3 zero-budget CMDP | A₀, Lagrangian doubling search on λ | no |
| 4 runtime filter only | A₀ (unconstrained proposal) | yes |
| 5 constraint-aware only | A_I | no |
| 6 combined | A_I | yes |

The zero-budget CMDP is a Lagrangian approximation, exact only as λ → ∞. The search returns the smallest
doubling λ that gives zero worst-case discounted violation cost on every grid state. Results are reported on
the true environment return. Violations are scored against the user's instruction.

**Penalty sweep** (`kind: "penalty_sweep"`): λ ∈ {1, 5, 10, 20, 50, 100} against hard exclusion.

**Freeze update** (`kind: "freeze"`, Eq. 13): the sev1 rule switches to a deployment freeze at step 5. Five arms:
F1 no update; F2 stale policy behind the new filter; F3 stale policy with the filter dropped;
F4 re-optimised over A_I2 behind the filter; F5 CMDP re-solve at the switch with no filter. F5 is timed,
because the CMDP must re-run its dual search before it has a policy.

Common random numbers: episode *i* uses the same seed in every arm.

## API (selection)

| method | path | purpose |
|---|---|---|
| POST | `/api/intents/parse` | parse → validate → compile → feasibility; returns each stage |
| POST | `/api/intents/{id}/activate` | atomic activation (`override` for a logged HOLD override) |
| GET | `/api/intents`, `/api/intents/active` | intent history and active version |
| GET | `/api/markov/admissibility` | cpu × latency admissible-set grid |
| GET | `/api/markov/decision` | one decision as a graph: actions, Q, successor probabilities |
| GET | `/api/markov/example51` | Example V.1, computed |
| POST | `/api/runs` | six_arm · penalty_sweep · freeze (persisted) |
| GET | `/api/runs/{id}`, `/api/runs/{id}/episodes/{arm}`, `/api/runs/{id}/trace/{arm}/{idx}` | results and audit traces |
| POST | `/api/parallel` | ICMDP and CMDP lanes on the same start and random draws |
| GET | `/api/events` | append-only lifecycle log |
| GET | `/metrics` | Prometheus text format |

## Frontend pages

1. **Intent lifecycle**: the parse → validate → compile → feasibility → activate pipeline and the admissible-set grid.
2. **Markov views**: one decision drawn as a graph (edge width = transition probability; the policy's choice starred).
   The Example V.1 graph shows why the planner must see the constraint.
3. **ICMDP vs CMDP workflow**: two pipelines (stages and timings) and the CMDP dual search, with lockstep lanes
   that replay the same episode step by step. Scenarios: the snapping case (raw cpu 75 vs a grid cpu 70) and
   the Sev1 deploy ban.
4. **Experiments**: six-arm bars with CIs (table view available), penalty sweep, state-occupancy stacked area,
   freeze update.
5. **Observability**: lifecycle events, runs, and the metrics text.

Colors come from the validated dark categorical palette (dataviz skill). Status colors are used only for
good / warning / critical. Every status also carries an icon and a label.

## Findings the harness exposes (seed 1, 500 episodes, Sev1 rule)

- Unconstrained: highest return (−12.8) but 0.86 violations per 100 steps, in 17% of episodes.
- Every constrained arm: zero violations. Gate-only loses about 1.6 return against combined, so planning
  must see the constraint.
- Snapping leak (`cpu > 70`, raw 75, seed 1, 500 episodes): the constraint-aware planner without a gate, and the
  zero-budget CMDP, both violate in 66% of episodes (4.2 per 100 steps). Gating on the raw state brings this to zero,
  at a return cost of about 1.7 against the ungated planner.
- The penalty needs λ ≈ 100 to reach zero violations for this rule.
- Freeze update: stale policy without a filter fails; re-optimising behind the filter, and the CMDP
  re-solve, both hold, but the CMDP re-solve must run its dual search first (about 0.6 s here).

## Not yet implemented

- Parser accuracy and perturbed or ambiguous instruction sets (§V.C). The local parser is keyword-based and is
  not evaluated against a gold set.
- Latency and runtime-overhead instrumentation beyond per-arm wall-clock. OpenTelemetry tracing is a planned
  drop-in; the metric names are stable for it.
- Q-learning and policy-gradient variants (Eqs. 21–22).
- The mid-episode update is driven by the harness, not by a live operator instruction in the UI.
- Browser checks used headless Chrome screenshots of the five pages. No automated UI tests exist yet.

## Paper correspondence

| Component | Paper |
|---|---|
| `A0`, `I`, `A_I = A0 ∩ I` | Def. III.1, Eq. 3 |
| Compiler `c(g)(s)` | Eq. 17 |
| Grammar G | Eq. 16 |
| Value iteration over `A_I` | Eq. 19 |
| Zero-budget CMDP encoding | Prop. III.2 |
| Feasibility `A_I(s) ≠ ∅` | Def. III.2 |
| Dynamic intent and activation | §IV-D, Eq. 13 |
| Gate and audit trace | Algorithm 1, steps 4–7 |
| Example V.1 | §V.A |

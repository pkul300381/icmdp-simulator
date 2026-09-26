# ICMDP Simulator — Instructions

## Overview

This simulator implements the **Intent-Constrained Markov Decision Process (ICMDP)**
framework from:

> Kulkarni & Saini (2026). *Intent-Constrained MDPs: Compiling User-Specified Intent
> into Planning-Aware Action Admissibility.*

It provides a browser-based runtime environment in which:

1. A user types a **natural-language instruction** (e.g., "Do not deploy during a Sev1 incident")
2. An **LLM parser** (Claude via Anthropic API) proposes a structured typed specification
3. A **deterministic validator** checks the specification against the declared schema
4. A **deterministic compiler** produces a versioned executable rule `I(s)`
5. A **constraint-aware planner** runs value iteration over `A_I(s) = A₀(s) ∩ I(s)`
6. A **runtime gate** enforces the active rule at every action-selection step
7. An **audit trace** logs every decision with state, admissible set, proposal, and executed action

---

## Architecture — Three Timescales (Paper Section 4.1)

```
SLOW  (parse once, compile once)
  User instruction → LLM parser fφ → raw spec g_raw
  → Deterministic validator vx → validated spec g_l,x
  → Deterministic compiler c → versioned rule I^(v)(·)

MEDIUM (planning)
  I^(v)(s) evaluated on hypothetical states
  → Value iteration over A_I(s) = A₀(s) ∩ I(s)
  → Optimal policy π*_I

FAST (enforcement, every step)
  Current state s_t
  → Evaluate I^(v)(s_t) → A_I(s_t)
  → Filter proposal → execute admissible action
  → Log to audit trace
```

---

## File Structure

```
icmdp-simulator/
├── INSTRUCTIONS.md          ← this file
├── environment.js           ← MDP environment definition (states, actions, transitions, rewards)
├── compiler.js              ← Intent grammar, validator, and compiler (Paper §4.2, eq. 16–17)
├── planner.js               ← Value iteration and policy iteration (Paper §5.1, eq. 19–20)
├── runtime.js               ← Runtime gate, feasibility monitor, audit trace (Paper §4.3)
├── api.js                   ← Anthropic API call — LLM parser fφ (Paper §4.1)
└── index.html               ← Full simulator UI (single-file application)
```

---

## The MDP Environment (Cloud-Operations, Paper Example 3.1)

### State Space S
Each state is a tuple:
```
s = (severity, health, cpu, latency, replicas, deploy_status)
```

| Field          | Type    | Values                          |
|----------------|---------|---------------------------------|
| severity       | enum    | none, sev2, sev1                |
| health         | enum    | healthy, degraded, critical     |
| cpu            | number  | 0–100 (%)                       |
| latency        | number  | ms                              |
| replicas       | integer | 1–10                            |
| deploy_status  | enum    | idle, in_progress               |

### Action Vocabulary A
```
A = { Deploy, Rollback, Restart, ScaleOut, ScaleIn, DoNothing }
```

### Baseline Availability A₀(s)
Actions physically available given current state (regardless of intent):
- `Deploy` unavailable if `deploy_status = in_progress`
- `Rollback` unavailable if `deploy_status = idle`
- `ScaleOut` unavailable if `replicas = 10`
- `ScaleIn` unavailable if `replicas = 1`
- `Restart`, `DoNothing` always available

### Reward Function R(s, a)
Optimises service availability, latency, and infrastructure cost among admissible actions.

### Discount Factor γ = 0.9

---

## Intent Grammar G (Paper §4.2, Equation 16)

```
g    ::= { r₁, ..., rₘ }
r    ::= prohibit(a, φ)
φ    ::= q | ¬φ | (φ ∧ φ) | (φ ∨ φ)
q    ::= h op u
```

Where:
- `a ∈ A` — action name
- `h` — typed state field name
- `u` — value of same type as h
- `op` — operator admitted for that type:
  - enum fields: `=`, `!=`
  - numeric fields: `=`, `!=`, `<`, `<=`, `>`, `>=`

### Compiler (Equation 17)
```
c(g)(s) = A \ { aⱼ : evalx(φⱼ, s) = true }
```
Intersection with A₀(s) gives A_I(s).

### Example
Instruction: *"Do not deploy while a Severity-1 production incident is active"*

Parser produces:
```json
{
  "prohibitions": [
    { "action": "Deploy", "condition": "severity = sev1" }
  ]
}
```

Compiler produces rule `I_sev1(s)`:
- `Deploy ∈ I(s)` when `severity ≠ sev1`
- `Deploy ∉ I(s)` when `severity = sev1`

---

## Key Algorithms

### Value Iteration (Paper Equation 19)
```
V_{k+1}(s) = max_{a ∈ A_I(s)} [ R(s,a) + γ Σ_{s'} P(s'|s,a) V_k(s') ]
```
Converges by Proposition 3.1 (γ-contraction under sup norm).

### Intent Feasibility Check (Paper Definition 3.2)
```
A_I(s) ≠ ∅  ∀ s ∈ S
```
If violated: system enters HOLD, requests clarification. No silent relaxation.

### Runtime Gate (Paper Algorithm 1, Step 6)
At each step:
1. Evaluate `I^(v)(s_t)` → `A_I(s_t)`
2. Check `A_I(s_t) ≠ ∅` (feasibility monitor)
3. Filter proposal policy output
4. Execute only admissible action
5. Log: `{ t, s_t, rule_version, A_I(s_t), proposal, executed, reward }`

---

## Example 5.1 — Why Runtime Filtering Alone Is Insufficient (Paper §5.1)

The simulator demonstrates this with γ = 0.9:

| Approach | At s₀ selects | Return |
|---|---|---|
| Unconstrained + runtime gate | Risky → gate → Recover | 0.9 × 1 = **0.9** |
| Constraint-aware planner | Safe | **5** |

Because `Q_unconstrained(s₀, Risky) = 0.9×10 = 9 > 5`, an unconstrained planner chooses
Risky expecting Deploy at s₁ — but Deploy is prohibited at s₁. The constraint-aware planner
uses `V_I(s₁) = 1`, computes `Q_I(s₀, Risky) = 0.9 < 5`, and correctly selects Safe.

---

## How to Run

### Option A — Open index.html directly
No server required. Open `index.html` in any modern browser.

You will need an Anthropic API key. Enter it in the Settings panel in the UI.
The key is used only for the LLM parser step (slow timescale). It never touches
the validator, compiler, planner, or runtime gate.

### Option B — Serve locally
```bash
cd icmdp-simulator
python3 -m http.server 8080
# then open http://localhost:8080
```

---

## Using the Simulator

1. **Set the environment state** using the state panel sliders and dropdowns
2. **Enter a natural-language instruction** in the intent input box
3. **Click Parse Intent** — the LLM proposes a structured specification
4. **Review the parsed spec** — the validator shows accept/reject with reason
5. **Click Compile & Activate** — the rule is versioned and installed
6. **Click Run Episode** — the planner runs value iteration, then executes one episode
7. **Inspect the audit trace** — every step logged with admissible set and executed action
8. **Toggle constraint-aware vs runtime-only** — see Example 5.1 in practice

---

## Failure Modes Demonstrated

| Failure | How triggered | System response |
|---|---|---|
| Over-tightening | Instruction excludes too many actions | Feasibility monitor fires, HOLD state |
| Under-specification | Instruction too vague | Validator returns ⊥, requests clarification |
| Empty admissible set | Constraints + state leave A_I(s) = ∅ | HOLD, escalation message shown |
| Dynamic intent update | New instruction entered mid-episode | Atomic activation, old rule archived |

---

## Paper Correspondence

| Simulator component | Paper reference |
|---|---|
| State/action/transition/reward | Definition 2.1, Example 3.1 |
| A₀(s) baseline availability | Definition 3.1 |
| A_I(s) = A₀(s) ∩ I(s) | Equation 3 |
| Hard admissibility π(a\|s)=0 | Equation 4 |
| Intent grammar G | Equation 16 |
| Compiler c(g)(s) | Equation 17 |
| Value iteration | Equation 19 |
| Q-learning update | Equation 21 |
| Feasibility check | Definition 3.2 |
| Three-timescale architecture | Figure 1, Algorithm 1 |
| Runtime gate + audit trace | Section 4.3 |
| Example 5.1 demo | Section 5.1 |

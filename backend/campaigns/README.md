# Stress campaigns

Campaigns that try to break the ICMDP framework's assumptions. They are separate from the unit
tests (`tests/`), which check that the code does what it was written to do.

Run from `backend/`:

```bash
python3 -m campaigns.parser_suite                 # Campaign 1: parser on 50 labelled instructions (<1 s)
python3 -m campaigns.perturbation                 # Campaign 4: observation noise and model mismatch (~15 s)
python3 -m campaigns.grid_boundary --mode sweep   # Campaign 5: snapping leak and grid resolution (~2 min)
```

Outputs are written to `campaigns/results/` as JSON. Every run is deterministic (fixed seeds).

## Campaign 1: parser (`parser_suite.py`, `gold_parser_set.py`)

Labels are hand-written by one person and have not been double-checked. Inter-rater agreement must
be measured before any number is cited. Only the keyword parser (`local_parse`) was run; the LLM
parser needs an API key and has not been run.

## Campaign 4: perturbation (`perturbation.py`)

Rule: forbid ScaleIn when CPU is above 70 percent. Noise is added to the cpu and latency readings
the planner and gate see; transitions are perturbed by a relative factor eps per outcome while the
planner keeps the nominal model. Violations are scored on the true state.

## Campaign 5: grid boundary (`grid_boundary.py`)

`ICMDP_CPU_GRID` sets the CPU grid for a fresh process. The default is the paper's 5-point grid.

## Shared simulator (`common.py`)

Mirrors `icmdp.runtime.run_episode`, with observation noise and a perturbed model added.

# ICMDP Agent Harness

A harness for testing the Intent-Constrained MDP framework: natural-language intent → validated rule →
constraint-aware planning → runtime gate, compared with a zero-budget CMDP workflow.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for the architecture, protocols, API, and findings.

Quick start:

cd ~/ai-workspace/icmdp-simulator/backend

pip3 install -r requirements.txt

python3 -m pytest -q tests

python3 -m pytest -q tests --html=report.html --self-contained-html

```bash
cd backend && python3 -m pip install -r requirements.txt && python3 -m uvicorn icmdp.api:app --port 8000
cd frontend && npm install && npm run dev      # open http://localhost:5173
```

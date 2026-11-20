"""Campaign 1: parser behaviour on hand-labelled, perturbed instructions.

Scoring compares what the two compiled rules EXCLUDE over a dense probe set of raw states:
  cpu 0..100 in steps of 0.5, latency 0..5000 ms in steps of 50, and every value of the
  other fields. Only fields that either rule mentions are varied; the rest are held fixed.
  (The 2,160-point planning grid is too coarse for threshold comparisons: its smallest CPU
  value is 20, so "cpu below 20" would match no grid state.)
Two phrasings count as equal when they exclude the same (state, action) pairs.

Verdicts:
  exact          same exclusions as the gold spec
  over_only      excludes extra (state, action) pairs (safe, but too restrictive)
  under_only     misses excluded pairs (UNSAFE: an operator's prohibition is not enforced)
  mixed          both (UNSAFE)
  held           no rule produced (safe: the operator must clarify)
  correct_reject the instruction was expected to be refused, and was
  silent_accept  the instruction was expected to be refused, but a rule was produced (CRITICAL)
"""
import itertools
import json
import pathlib
from collections import Counter, defaultdict

from icmdp import intent as I
from icmdp.model import DEPLOY, HEALTH, SEVERITY, State, baseline

from .gold_parser_set import GOLD

OUT = pathlib.Path(__file__).parent / "results" / "campaign1_parser.json"
ALL_FIELDS = ("severity", "health", "cpu", "latency", "replicas", "deploy_status")
DOMAINS = {
    "severity": list(SEVERITY),
    "health": list(HEALTH),
    "cpu": [round(0.5 * i, 1) for i in range(201)],
    "latency": [float(x) for x in range(0, 5001, 50)],
    "replicas": list(range(1, 11)),
    "deploy_status": list(DEPLOY),
}
FIXED = dict(severity="none", health="healthy", cpu=50.0, latency=300.0, replicas=4, deploy_status="idle")


def fields_of(cond, acc):
    if hasattr(cond, "field"):
        acc.add(cond.field)
    elif hasattr(cond, "child"):
        fields_of(cond.child, acc)
    elif hasattr(cond, "children"):
        for c in cond.children:
            fields_of(c, acc)
    return acc


def rule_fields(rule):
    acc = set()
    for p in rule.spec.prohibitions:
        fields_of(p.cond, acc)
    return acc


def excl_set(rule, dims):
    vals = [DOMAINS[f] if f in dims else [FIXED[f]] for f in ALL_FIELDS]
    out = set()
    for combo in itertools.product(*vals):
        s = State(*combo)
        av = baseline(s)
        for a in rule.excluded(s):
            if a in av:
                out.add((combo, a))
    return out


def compile_spec(spec):
    vr = I.validate(spec)
    assert vr.valid, (spec, vr.reason)
    return I.compile_spec(vr)


def evaluate(it):
    out = I.local_parse(it["text"])
    pred, status = None, "held"
    if "raw" in out:
        vr = I.validate(out["raw"])
        if vr.valid:
            pred, status = I.compile_spec(vr), "rule"
        else:
            status = "rejected: " + vr.reason
    else:
        status = "held: " + out.get("error", "")

    row = dict(id=it["id"], category=it["category"], text=it["text"], expect=it["expect"], status=status)
    if it["expect"] == "reject":
        row["verdict"] = "correct_reject" if pred is None else "silent_accept"
        return row

    gold = compile_spec(it["gold"])
    dims = rule_fields(gold) | (rule_fields(pred) if pred is not None else set())
    G = excl_set(gold, dims)
    if pred is None:
        row.update(verdict="held", over=0, under=len(G), gold_pairs=len(G))
        return row
    P = excl_set(pred, dims)
    over, under = len(P - G), len(G - P)
    verdict = ("exact" if over == 0 and under == 0 else
               "over_only" if under == 0 else
               "under_only" if over == 0 else "mixed")
    row.update(verdict=verdict, over=over, under=under, gold_pairs=len(G), predicted_pairs=len(P),
               dims=sorted(dims))
    return row


def main():
    rows = [evaluate(it) for it in GOLD]
    total = Counter(r["verdict"] for r in rows)
    by_cat = defaultdict(Counter)
    for r in rows:
        by_cat[r["category"]][r["verdict"]] += 1
    rule_items = [r for r in rows if r["expect"] == "rule"]
    unsafe = [r["id"] for r in rows if r["verdict"] in ("under_only", "mixed", "silent_accept")]
    summary = dict(n=len(rows), n_rule_items=len(rule_items), n_reject_items=len(rows) - len(rule_items),
                   verdicts=dict(total), unsafe_or_critical=unsafe,
                   under_restriction_rate_on_rule_items=round(
                       sum(1 for r in rule_items if r["verdict"] in ("under_only", "mixed")) / max(1, len(rule_items)), 3),
                   silent_accept_rate_on_reject_items=round(
                       sum(1 for r in rows if r["verdict"] == "silent_accept")
                       / max(1, len(rows) - len(rule_items)), 3))
    result = dict(campaign="1_parser", labeller="single labeller, not double-checked",
                  scoring="dense probe set over referenced fields (see parser_suite.py)",
                  summary=summary, by_category={k: dict(v) for k, v in by_cat.items()}, rows=rows)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nper category:")
    for k, v in by_cat.items():
        print(f"  {k:12s} {dict(v)}")
    print("\nnon-exact outcomes:")
    for r in rows:
        if r["verdict"] not in ("exact", "correct_reject", "held"):
            print(f"  {r['id']:4s} {r['verdict']:14s} {r['text'][:78]}")


if __name__ == "__main__":
    main()

"""
Observability: structured JSON logs and a Prometheus text-format metrics registry.

Metrics are exposed at GET /metrics (scrape with the provided prometheus.yml). Traces of
individual decisions live in the SQLite audit tables; the event log mirrors intent lifecycle
events. OpenTelemetry export is a drop-in next step; the metric names are stable for it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, Tuple


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str = "icmdp") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class Metrics:
    """Minimal counters, gauges and summaries with labels, rendered in Prometheus text format."""

    def __init__(self):
        self.lock = threading.Lock()
        self.counters: Dict[Tuple[str, Tuple], float] = defaultdict(float)
        self.gauges: Dict[Tuple[str, Tuple], float] = {}
        self.summaries: Dict[Tuple[str, Tuple], list] = defaultdict(lambda: [0.0, 0])  # [sum, count]
        self.help: Dict[str, str] = {}

    @staticmethod
    def _labels(labels: Dict[str, str]) -> Tuple:
        return tuple(sorted((k, str(v)) for k, v in labels.items()))

    def describe(self, name: str, text: str) -> None:
        self.help[name] = text

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        with self.lock:
            self.counters[(name, self._labels(labels))] += value

    def set(self, name: str, value: float, **labels) -> None:
        with self.lock:
            self.gauges[(name, self._labels(labels))] = value

    def observe(self, name: str, value: float, **labels) -> None:
        with self.lock:
            s = self.summaries[(name, self._labels(labels))]
            s[0] += value
            s[1] += 1

    def render(self) -> str:
        lines = []

        def fmt(name, labels):
            if not labels:
                return name
            inner = ",".join(f'{k}="{v}"' for k, v in labels)
            return f"{name}{{{inner}}}"

        with self.lock:
            for kind, store in (("counter", self.counters), ("gauge", self.gauges)):
                names = sorted({n for n, _ in store})
                for n in names:
                    lines.append(f"# HELP {n} {self.help.get(n, n)}")
                    lines.append(f"# TYPE {n} {kind}")
                    for (nn, labels), v in sorted(store.items()):
                        if nn == n:
                            lines.append(f"{fmt(nn, labels)} {v}")
            names = sorted({n for n, _ in self.summaries})
            for n in names:
                lines.append(f"# HELP {n} {self.help.get(n, n)}")
                lines.append(f"# TYPE {n} summary")
                for (nn, labels), (total, count) in sorted(self.summaries.items()):
                    if nn == n:
                        lines.append(f"{fmt(n + '_sum', labels)} {total}")
                        lines.append(f"{fmt(n + '_count', labels)} {count}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()
METRICS.describe("icmdp_runs_total", "Experiment runs executed, by kind")
METRICS.describe("icmdp_intent_events_total", "Intent lifecycle outcomes (validated, rejected, hold, activated)")
METRICS.describe("icmdp_gate_interventions_total", "Proposals replaced by the runtime gate, by arm")
METRICS.describe("icmdp_violations_total", "Executed actions excluded by the scored intent, by arm")
METRICS.describe("icmdp_episode_seconds", "Wall-clock seconds per run, by kind")
METRICS.describe("icmdp_planning_seconds", "Seconds spent in planning solves, by planner")
METRICS.describe("icmdp_active_intent_version", "Version number of the active compiled rule")

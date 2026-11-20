"""
SQLite persistence: intent lifecycle (rule store with versions), runs, episodes, trace steps,
and an append-only event log for observability. Stdlib only.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    instruction TEXT NOT NULL,
    parser TEXT NOT NULL,
    raw_json TEXT,
    spec_json TEXT,
    status TEXT NOT NULL,            -- draft | rejected | hold | validated | active | retired
    reason TEXT,
    version INTEGER,                 -- assigned at activation
    empty_states INTEGER,
    created_at TEXT NOT NULL,
    activated_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    intent_id INTEGER,
    params_json TEXT NOT NULL,
    summary_json TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS episodes (
    run_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    idx INTEGER NOT NULL,
    start_key TEXT,
    return REAL,
    steps INTEGER,
    violations INTEGER,
    interventions INTEGER,
    holds INTEGER,
    operational_errors INTEGER,
    PRIMARY KEY (run_id, arm, idx)
);
CREATE TABLE IF NOT EXISTS trace_steps (
    run_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    idx INTEGER NOT NULL,
    t INTEGER NOT NULL,
    state_json TEXT,
    rule_version INTEGER,
    admissible_json TEXT,
    proposal TEXT,
    executed TEXT,
    reward REAL,
    violated INTEGER,
    event TEXT,
    kind TEXT,
    PRIMARY KEY (run_id, arm, idx, t)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    ref TEXT,
    payload_json TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # -- events -------------------------------------------------------------
    def event(self, kind: str, ref: Optional[str] = None, payload: Optional[Dict[str, Any]] = None) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO events (ts, kind, ref, payload_json) VALUES (?,?,?,?)",
                              (now(), kind, ref, json.dumps(payload or {}, default=str)))
            self.conn.commit()

    def events(self, limit: int = 200) -> List[Dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload_json"] or "{}")} for r in rows]

    # -- intents ------------------------------------------------------------
    def create_intent(self, name: Optional[str], instruction: str, parser: str, raw: Any, spec: Any,
                      status: str, reason: str = "", empty_states: Optional[int] = None) -> int:
        with self.lock:
            cur = self.conn.execute(
                """INSERT INTO intents (name, instruction, parser, raw_json, spec_json, status, reason,
                   empty_states, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (name, instruction, parser, json.dumps(raw), json.dumps(spec) if spec is not None else None,
                 status, reason, empty_states, now()))
            self.conn.commit()
            return int(cur.lastrowid)

    def _intent_row(self, r) -> Dict:
        d = dict(r)
        d["raw"] = json.loads(d.pop("raw_json") or "null")
        d["spec"] = json.loads(d.pop("spec_json") or "null")
        return d

    def intent(self, intent_id: int) -> Optional[Dict]:
        with self.lock:
            r = self.conn.execute("SELECT * FROM intents WHERE id=?", (intent_id,)).fetchone()
        return self._intent_row(r) if r else None

    def intents(self) -> List[Dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM intents ORDER BY id DESC").fetchall()
        return [self._intent_row(r) for r in rows]

    def active_intent(self) -> Optional[Dict]:
        with self.lock:
            r = self.conn.execute("SELECT * FROM intents WHERE status='active' ORDER BY version DESC LIMIT 1").fetchone()
        return self._intent_row(r) if r else None

    def set_intent_status(self, intent_id: int, status: str, reason: str = "") -> None:
        with self.lock:
            self.conn.execute("UPDATE intents SET status=?, reason=? WHERE id=?", (status, reason, intent_id))
            self.conn.commit()

    def activate_intent(self, intent_id: int) -> int:
        """Atomic activation: retire the previous active version and install this one."""
        with self.lock:
            cur = self.conn.execute("SELECT COALESCE(MAX(version),0) FROM intents").fetchone()
            version = int(cur[0]) + 1
            self.conn.execute("UPDATE intents SET status='retired' WHERE status='active'")
            self.conn.execute("UPDATE intents SET status='active', version=?, activated_at=? WHERE id=?",
                              (version, now(), intent_id))
            self.conn.commit()
        return version

    # -- runs ---------------------------------------------------------------
    def create_run(self, run_id: str, kind: str, intent_id: Optional[int], params: Dict) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO runs (id, kind, intent_id, params_json, status, created_at) VALUES (?,?,?,?,?,?)",
                              (run_id, kind, intent_id, json.dumps(params, default=str), "running", now()))
            self.conn.commit()

    def finish_run(self, run_id: str, summary: Dict, status: str = "done") -> None:
        with self.lock:
            self.conn.execute("UPDATE runs SET summary_json=?, status=?, finished_at=? WHERE id=?",
                              (json.dumps(summary, default=str), status, now(), run_id))
            self.conn.commit()

    def run(self, run_id: str) -> Optional[Dict]:
        with self.lock:
            r = self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["params"] = json.loads(d.pop("params_json") or "{}")
        d["summary"] = json.loads(d.pop("summary_json") or "null")
        return d

    def runs(self, limit: int = 50) -> List[Dict]:
        with self.lock:
            rows = self.conn.execute("SELECT id, kind, intent_id, status, created_at, finished_at FROM runs "
                                     "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def save_episodes(self, run_id: str, rows: Iterable[Dict]) -> None:
        with self.lock:
            self.conn.executemany(
                """INSERT OR REPLACE INTO episodes (run_id, arm, idx, start_key, return, steps, violations,
                   interventions, holds, operational_errors) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [(run_id, r["arm"], r["index"], r.get("start"), r["return"], r["steps"], r["violations"],
                  r["interventions"], r["holds"], r["operational_errors"]) for r in rows])
            self.conn.commit()

    def save_traces(self, run_id: str, traces: Dict[str, List[Dict]]) -> None:
        data = []
        for arm, eps in traces.items():
            for ep in eps:
                for st in ep["steps"]:
                    data.append((run_id, arm, ep["index"], st["t"], json.dumps(st["state"]),
                                 st["rule_version"], json.dumps(st["admissible"]), st["proposal"],
                                 st["executed"], st["reward"], int(bool(st["violated"])), st["event"], st["kind"]))
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO trace_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", data)
            self.conn.commit()

    def episodes(self, run_id: str, arm: str, limit: int = 200, offset: int = 0) -> List[Dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM episodes WHERE run_id=? AND arm=? ORDER BY idx LIMIT ? OFFSET ?",
                                     (run_id, arm, limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def trace(self, run_id: str, arm: str, idx: int) -> List[Dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM trace_steps WHERE run_id=? AND arm=? AND idx=? ORDER BY t",
                                     (run_id, arm, idx)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["state"] = json.loads(d.pop("state_json") or "null")
            d["admissible"] = json.loads(d.pop("admissible_json") or "[]")
            d["violated"] = bool(d["violated"])
            out.append(d)
        return out

"""Production Pattern #4 (Observability) + #3 (Idempotency) — from agents-towards-production / AI-Agents-for-Production audit 2026-11-08.

Structured run tracing for the VaR engine: every pipeline run gets a trace_id,
a deterministic idempotency key (hash of inputs), timing, and a JSON log line
that a postmortem can actually use. No external deps.
"""
from __future__ import annotations
import hashlib, json, time, uuid
from dataclasses import dataclass, field, asdict
from typing import Any

def idempotency_key(*parts: Any) -> str:
    """Deterministic key: same inputs -> same key, so a retry cannot double-write."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()[:32]

@dataclass
class RunTrace:
    run_name: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    idempotency_key: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    status: str = "running"
    events: list = field(default_factory=list)
    def event(self, name: str, **data):
        self.events.append({"event": name, "t": round(time.time()-self.started_at, 4), **data})
    def finish(self, status: str = "ok"):
        self.status = status; self.finished_at = time.time()
    @property
    def duration_s(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 4)
    def to_json(self) -> str:
        d = asdict(self); d["duration_s"] = self.duration_s
        return json.dumps(d, default=str)

def traced_run(run_name: str, *key_parts: Any) -> RunTrace:
    t = RunTrace(run_name=run_name)
    t.idempotency_key = idempotency_key(run_name, *key_parts)
    t.event("start", idempotency_key=t.idempotency_key)
    return t

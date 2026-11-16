from var_engine.observability import RunTrace, idempotency_key, traced_run
import json
def test_idempotency_key_deterministic():
    assert idempotency_key("var", 0.99, "SPY") == idempotency_key("var", 0.99, "SPY")
    assert idempotency_key("var", 0.99) != idempotency_key("var", 0.95)
def test_trace_lifecycle_and_json():
    t = traced_run("backtest", 0.99, "2026-11-08")
    assert t.status == "running" and len(t.idempotency_key) == 32
    t.event("methods_done", n=9); t.finish()
    d = json.loads(t.to_json())
    assert d["status"] == "ok" and d["duration_s"] >= 0
    assert [e["event"] for e in d["events"]] == ["start", "methods_done"]
def test_trace_ids_unique():
    assert traced_run("a").trace_id != traced_run("a").trace_id
